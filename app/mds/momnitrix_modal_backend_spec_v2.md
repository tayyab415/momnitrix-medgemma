# Momnitrix Modal Backend Spec (V2)

Last updated: 2026-02-22

## 1) Deployment Topology

This backend is split into 4 Modal services:

1. `momnitrix-api-v2` (public orchestrator API)
2. `momnitrix-core-gpu` (MedGemma + MedSigLIP)
3. `momnitrix-derm-tf` (Derm Foundation + classifier head)
4. `momnitrix-medasr` (MedASR transcription)

Primary public URL:

- `https://tayyabkhn343--momnitrix-api-v2-web.modal.run`

## 2) Service Specifications

### A) Public API: `modal_api.py`

- Modal app: `momnitrix-api-v2`
- Runtime: CPU `2`, timeout `600s`, `min_containers=1`
- Secrets: `medgemma`, `momnitrix-config`
- Endpoints:
  - `GET /health`
  - `POST /v1/triage/stream` (SSE)
  - `GET /v1/triage/{request_id}`

### B) Core GPU: `modal_core_gpu.py`

- Modal app: `momnitrix-core-gpu`
- Runtime: GPU `T4`, timeout `900s`, `min_containers=0`, `max_containers=1`
- Secrets: `medgemma-hf`, `momnitrix-config`
- Endpoints:
  - `GET /health`
  - `POST /internal/medgemma/risk_decide`
  - `POST /internal/medsiglip/infer`
- Model stack:
  - Base: `google/medgemma-1.5-4b-it`
  - Adapter: `tyb343/mamaguard-vitals-lora-p100`
  - MedSigLIP model id from env (`MOMNITRIX_MEDSIGLIP_MODEL_ID`)

### C) Derm Service: `modal_derm_tf.py`

- Modal app: `momnitrix-derm-tf`
- Runtime: GPU `T4`, timeout `900s`, `min_containers=1`, `max_containers=3`
- Secrets: `medgemma`, `momnitrix-config`
- Endpoint:
  - `POST /internal/derm/infer`
- Uses local artifacts mounted from `artifacts/derm`.

### D) MedASR Service: `modal_medasr.py`

- Modal app: `momnitrix-medasr`
- Runtime: CPU `4`, timeout `300s`, `min_containers=0`, `max_containers=4`
- Secrets: `medgemma`, `momnitrix-config`
- Endpoint:
  - `POST /internal/medasr/transcribe`

## 3) Orchestration Pipeline (End-to-End)

Entry point: `POST /v1/triage/stream`

Execution sequence:

1. Accept request and emit `request.accepted`.
2. Store raw artifacts if present:
   - `wound_image_b64` -> `raw_wound.jpg`
   - `skin_image_b64` -> `raw_skin.jpg`
   - `audio_b64` -> `raw_audio.wav`
3. Run specialist models in parallel if relevant input exists:
   - wound image -> MedSigLIP
   - skin image -> Derm TF
   - audio -> MedASR
4. Always run MedGemma decision step (`medgemma.started` -> `medgemma.completed`).
5. Compute policy floor escalation (`compute_policy_floor`).
6. Compose patient-facing output with Gemini (`gemini-3-flash-preview`).
7. Emit streamed chunks (`gemini.delta`), then `triage.final`.
8. Persist final response JSON to artifact store.

## 4) Routing Rules

Routing is modality-driven:

- `inputs.wound_image_b64` -> MedSigLIP path
- `inputs.skin_image_b64` -> Derm path
- `inputs.audio_b64` -> MedASR path
- Vitals + text + specialist outputs -> MedGemma risk reasoning
- Final message shaping -> Gemini orchestrator

No specialist input means MedGemma still runs on vitals/text only.

## 5) Safety/Fallback Behavior

1. If specialist endpoints fail, orchestrator continues and emits `model.failed`.
2. If MedGemma remote call fails/timeouts, gateway falls back to heuristic decision logic.
3. Policy floor can escalate risk regardless of model output (e.g., severe glucose/BP).
4. Gemini call failure falls back to templated patient/visit summary text.

## 6) Key Configuration Contract (`momnitrix/config.py`)

Critical env vars:

- `MOMNITRIX_GEMINI_MODEL` (default `gemini-3-flash-preview`)
- `GEMINI_API_KEY`
- `HF_TOKEN`
- `MOMNITRIX_CORE_GPU_BASE_URL`
- `MOMNITRIX_DERM_BASE_URL`
- `MOMNITRIX_MEDASR_BASE_URL`
- `MOMNITRIX_REQUEST_TIMEOUT_SEC` (default `120`)
- `MOMNITRIX_USE_REAL_MODELS`
- `MOMNITRIX_MEDGEMMA_BASE_MODEL_ID`
- `MOMNITRIX_MEDGEMMA_ADAPTER_ID`
- `MOMNITRIX_MEDGEMMA_IS_ADAPTER`
- `MOMNITRIX_MEDSIGLIP_MODEL_ID`

## 7) Public API Input/Output

### Input (`/v1/triage/stream`)

- `patient_context`
- `vitals`
- `inputs`
- optional `request_id`, `metadata`

### Stream Events

Common emitted events:

- `request.accepted`
- `artifact.uploaded` (optional)
- `model.started` / `model.completed` / `model.failed`
- `medgemma.started` / `medgemma.completed`
- `gemini.started` / `gemini.delta` / `gemini.completed`
- `triage.final`
- `triage.error` (if uncaught exception)

## 8) Frontend Attachment (Updated)

Current frontend defaults should point to:

- `https://tayyabkhn343--momnitrix-api-v2-web.modal.run`

Updated in:

- `qa_console/index.html`
- `gemini-creations/frontend/components/SimulationForm.tsx` (fallback URL)
- `kimi-frontend/README.md` (documentation default)

## 9) Operational Notes

1. T4 + `min_containers=0` on core service reduces cost but increases cold-start latency.
2. MedGemma latency on T4 can exceed 45s; API timeout default was raised to 120s.
3. If strict low-latency demo is needed, increase warm capacity or move core GPU tier up temporarily.
