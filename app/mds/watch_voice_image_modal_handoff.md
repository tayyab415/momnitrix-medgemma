# Momnitrix Watch Integration Handoff (Voice + Image + Orchestration)

This guide is for the friend implementing Apple Watch / SwiftUI client integration against the already deployed Momnitrix Modal backend.

## 1) What is deployed right now

Use only the orchestrator URL from the client app.

- Public orchestrator base URL:
  - `https://tayyabkhn343--momnitrix-api-v2-web.modal.run`
- Health endpoint:
  - `GET /health`
- Main streaming triage endpoint:
  - `POST /v1/triage/stream`

Do **not** call internal specialist endpoints from the watch app in normal flow. Send one request to orchestrator and let it route.

## 2) Modal apps and what each Python file does

### A) `modal_api.py` (App: `momnitrix-api-v2`)

Role:
- Public API gateway and SSE streaming endpoint.
- Validates request payload.
- Runs orchestration pipeline.
- Emits timeline events (`request.accepted`, `router.decision`, `model.started`, `medgemma.completed`, `triage.final`, etc.).

Key routes:
- `GET /health`
- `POST /v1/triage/stream`
- `GET /v1/triage/{request_id}`

Important behavior:
- Returns `422` if required fields are missing.
- Required fields enforced server-side:
  - `patient_context.age_years`
  - `vitals.systolic_bp`
  - `vitals.diastolic_bp`
  - `vitals.fasting_glucose_mmol_l` (alias `fasting_glucose` accepted)
  - `vitals.temp_c`
  - `vitals.hr`

### B) `modal_core_gpu.py` (App: `momnitrix-core-gpu`)

Role:
- GPU service hosting:
  - Fine-tuned MedGemma (base + LoRA adapter)
  - Fine-tuned MedSigLIP (wound image specialist)

Key routes:
- `POST /internal/medgemma/risk_decide`
- `POST /internal/medsiglip/infer`

Important behavior:
- Single MedGemma generation lock to avoid GPU contention spikes.
- Emits parse diagnostics in logs (`parse_mode`, `fallback_used`).

### C) `modal_derm_tf.py` (App: `momnitrix-derm-tf`)

Role:
- Dermatology specialist service using Derm Foundation + your classifier artifacts.

Key route:
- `POST /internal/derm/infer`

Artifacts:
- Mounted from `artifacts/derm` into `/root/artifacts/derm`.
- Paths are configured in `momnitrix/config.py`.

### D) `modal_medasr.py` (App: `momnitrix-medasr`)

Role:
- Voice transcription service (MedASR).

Key route:
- `POST /internal/medasr/transcribe`

Important behavior:
- Cleans common ASR artifacts (`<s>`, `<epsilon>`, spacing normalization).

## 3) Core orchestration files (under `momnitrix/`)

### `momnitrix/orchestration.py`

Role:
- The end-to-end pipeline coordinator (`TriageOrchestrator.run`).
- Decides specialists needed from inputs.
- Calls specialists in parallel where possible.
- Calls MedGemma for risk decision.
- Applies safety floor/guardrails.
- Optionally uses Gemini for final wording (`gemini_full`) or skips (`medgemma_first`).
- Emits SSE timeline events and final response.

### `momnitrix/gateway.py`

Role:
- HTTP client wrapper from orchestrator to specialist services.
- Methods:
  - `medasr_transcribe(...)`
  - `medsiglip_infer(...)`
  - `derm_infer(...)`
  - `medgemma_decide_with_meta(...)`

### `momnitrix/model_runtime.py`

Role:
- Actual model loading and inference runtime logic for:
  - `CoreGpuRuntime` (MedGemma + MedSigLIP)
  - `DermRuntime`
  - `MedasrRuntime`
- Builds MedGemma system prompt + user prompt.
- Parses MedGemma output (`json` or sectioned text), with fallback heuristics when needed.
- Exposes timing diagnostics (`cold_start`, `gpu_warmup_ms`, `medgemma_inference_ms`).

### `momnitrix/gemini.py`

Role:
- Gemini orchestration helper:
  - builds route/prompt task instruction for MedGemma
  - optionally composes final patient-facing text in `gemini_full` mode

### `momnitrix/config.py`

Role:
- Env-driven settings:
  - endpoint URLs
  - model IDs
  - composer mode
  - secrets/env keys

## 4) Channel routing: how voice/image are selected

Routing source of truth:
- `momnitrix/orchestration.py` `_build_router_decision(...)`

Rules (simplified):
- If `inputs.audio_b64` present -> add `medasr` specialist.
- If `inputs.wound_image_b64` present -> add `medsiglip` specialist.
- If `inputs.skin_image_b64` present -> add `derm` specialist.
- If both wound + skin are present -> use both image specialists.
- Text-only mode uses MedGemma without image/audio specialists.

Then orchestrator emits:
- `router.decision`
- `router.prompt_plan`
- specialist start/completed events
- `medgemma.completed`
- `diagnostics.inference_breakdown`
- `triage.final`

## 5) Watch client payload contract (must match exactly)

### Minimal valid JSON shape

```json
{
  "request_id": "watch-<uuid>",
  "patient_context": {
    "age_years": 29,
    "gestational_weeks": 31,
    "known_conditions": [],
    "medications": []
  },
  "vitals": {
    "systolic_bp": 130,
    "diastolic_bp": 80,
    "fasting_glucose": 5.1,
    "hr": 86,
    "temp_c": 36.7,
    "spo2": 98,
    "hrv": 42.0
  },
  "inputs": {
    "free_text": "Patient note or typed question",
    "audio_b64": null,
    "wound_image_b64": null,
    "skin_image_b64": null,
    "headache": false,
    "vision_changes": false,
    "decreased_fetal_movement": false
  },
  "metadata": {
    "ui_mode": "text",
    "composer_mode": "medgemma_first",
    "medgemma_output_style": "notebook"
  }
}
```

### Required naming notes

Use snake_case keys exactly.

Common watch mapping errors that cause `422`:
- `ageYears` instead of `age_years`
- `heartRate` instead of `hr`
- `temperatureF` sent as `temp_c` without conversion
- `fastingGlucose` missing or outside key aliases

### Required values rules

Mandatory each run:
- age
- systolic BP
- diastolic BP
- fasting glucose
- body temperature (Celsius)
- heart rate

Optional:
- `hrv`, `spo2`, `gestational_weeks`, media fields, notes.

## 6) Voice path (MedASR) integration checklist

Watch frontend must:
- Record or pick audio.
- Convert to base64 string and place in `inputs.audio_b64`.
- Set `metadata.ui_mode = "voice"` (recommended).

Backend path:
- Orchestrator -> `ModelGateway.medasr_transcribe` -> `modal_medasr.py` -> transcript in `specialist_outputs.transcript`.
- Transcript is passed into MedGemma prompt context.

If voice is not showing in output:
- Verify `audio_b64` not empty.
- Check SSE for `model.started`/`model.completed` where `model=medasr`.
- Check final JSON has `specialist_outputs.transcript`.

## 7) Image path integration checklist

### Wound image

Watch frontend:
- Base64 encode wound image into `inputs.wound_image_b64`.
- Optionally set `metadata.ui_mode = "image"`.

Backend:
- Orchestrator -> MedSigLIP via core GPU endpoint.
- Output appears under `specialist_outputs.wound_scores`.

### Skin image

Watch frontend:
- Base64 encode skin image into `inputs.skin_image_b64`.

Backend:
- Orchestrator -> Derm endpoint.
- Output appears under:
  - `specialist_outputs.skin_scores`
  - `specialist_outputs.skin_top3`

If image evidence is missing in final reasoning:
- Confirm image fields are not null.
- Confirm SSE had corresponding specialist completed events.
- Confirm `specialist_outputs` includes wound/skin structures.

## 8) Composer/orchestrator mode behavior

`composer_mode` controls who writes patient-facing final text:

- `medgemma_first`
  - MedGemma dominates final narrative.
  - Gemini compose is skipped.
  - Good when you want fine-tuned MedGemma style preserved.

- `gemini_full`
  - MedGemma still handles risk decision core.
  - Gemini rewrites final patient message + visit summary.

Recommendation for watch demo consistency:
- Use `medgemma_first` + `medgemma_output_style=notebook`.

## 9) SSE events friend should surface in watch UI

At minimum display these states:
- `request.accepted`
- `router.decision`
- `model.started` / `model.completed`
- `medgemma.started` / `medgemma.completed`
- `diagnostics.inference_breakdown`
- `triage.final`
- `triage.error` (if any)

This gives confidence that voice/image actually routed and ran.

## 10) Frontend-to-backend connection (what to wire)

### Endpoint
- `POST https://tayyabkhn343--momnitrix-api-v2-web.modal.run/v1/triage/stream`

### Headers
- `Content-Type: application/json`

### Transport
- SSE stream parsing required (`event:` + `data:` format).
- Keep-alive comments (`: keep-alive`) are normal.

### Final payload
- The `triage.final` event contains final structured response.

## 11) Fast diagnostics playbook for your friend

If run fails:

1. Check status code.
- `422`: payload schema/key mismatch or missing required fields.
- `200`: pipeline accepted; inspect SSE events for missing specialist channel.

2. Check `/health?probe=true`.
- Verify `core_gpu_reachable`, `derm_reachable`, `medasr_reachable`.

3. Check whether specialist path executed.
- Look for `model.completed` events per expected channel.

4. Check final JSON.
- `specialist_outputs.transcript` should exist for voice runs.
- `specialist_outputs.wound_scores` for wound image runs.
- `specialist_outputs.skin_top3` for skin image runs.

5. Check inference diagnostics.
- `inference_diagnostics.router` fields show selected specialists.

## 12) Practical guidance for Swift friend

Use explicit Codable keys (avoid implicit conversion bugs):
- Map exactly to backend snake_case names.
- Convert Fahrenheit to Celsius before payload (`temp_c`).
- Ensure `audio_b64` / image base64 strings are raw base64 (or data-URL accepted by backend decoder).

For parity with current web frontend behavior:
- Mirror logic in `web-app/src/adapters/media.ts`
- Mirror payload shape from `web-app/src/domain/payload.ts`
- Mirror SSE parsing from `web-app/src/domain/sse.ts`

## 13) One-line architecture summary

Watch app sends one triage payload to orchestrator -> orchestrator selects specialists (MedASR / MedSigLIP / Derm) -> aggregates specialist outputs + vitals -> MedGemma risk inference on GPU -> optional Gemini composition -> streams full timeline and final triage back to watch UI.

