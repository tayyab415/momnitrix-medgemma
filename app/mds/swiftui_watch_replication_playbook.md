# Momnitrix SwiftUI Replication Playbook

Audience: frontend/mobile engineer replicating the current web prototype into a native Watch-A-Ways SwiftUI app.

Scope:
- Build a SwiftUI app that behaves like the current web app.
- Reuse existing Modal backend.
- Do **not** change backend/model/orchestration unless explicitly approved.

---

## 1) What You Are Replicating

Current system = 4 Modal services + 1 web reference frontend.

Flow:
1. User enters vitals + text/voice/image input.
2. Public API (`/v1/triage/stream`) orchestrates specialists and MedGemma.
3. SSE events stream progress + diagnostics.
4. Final triage response arrives as `triage.final`.

Specialists:
- MedASR (audio -> transcript)
- MedSigLIP (wound image)
- Derm Foundation + classifier head (skin image)
- MedGemma 1.5 4B (+ LoRA adapter) for risk reasoning
- Gemini 3 Flash Preview for orchestration/planning and optional final composition

---

## 2) Quick Decision: Use Existing Backend or Deploy Your Own

## Option A (Recommended for Swift replication now): Use existing deployed backend
- You only need the public API base URL from Tayyab.
- No Gemini/HF/API keys are required in the iOS/watch app.
- App calls HTTPS endpoints only.

## Option B: Deploy your own Modal stack
- Needed only if you must run an isolated environment.
- Requires your own Modal account + secrets + model access.

---

## 3) Backend File Map (What each file does)

## Modal entrypoints
- `modal_api.py`
  - Public orchestrator API.
  - Endpoints:
    - `GET /health`
    - `POST /v1/triage/stream` (SSE)
    - `GET /v1/triage/{request_id}`
  - Validates required vitals and streams events.

- `modal_core_gpu.py`
  - Core GPU service (MedGemma + MedSigLIP).
  - Endpoints:
    - `GET /health`
    - `POST /internal/medgemma/risk_decide`
    - `POST /internal/medsiglip/infer`
  - Uses A100 in current code.

- `modal_derm_tf.py`
  - Derm Foundation + classifier service.
  - Endpoint:
    - `POST /internal/derm/infer`
  - Mounts local artifacts from `artifacts/derm` to `/root/artifacts/derm`.

- `modal_medasr.py`
  - MedASR transcription service.
  - Endpoint:
    - `POST /internal/medasr/transcribe`

## Shared orchestration/runtime modules
- `momnitrix/config.py`
  - All env-driven settings (model IDs, service URLs, timeouts, tokens, composer mode).

- `momnitrix/schemas.py`
  - Contract models for request/response and internal service payloads.
  - Important aliases: fasting glucose accepts keys like `fasting_glucose`, `fasting_plasma_glucose`, etc.

- `momnitrix/orchestration.py`
  - Main triage pipeline:
    - routing decision
    - specialist fan-out
    - MedGemma inference
    - policy-floor escalation
    - optional Gemini composition
    - diagnostics + final response emission

- `momnitrix/gateway.py`
  - HTTP client for core/derm/medasr services.
  - Has deterministic fallback paths if upstream unavailable.

- `momnitrix/gemini.py`
  - Gemini planning/composition client.
  - Builds task instruction used to refine MedGemma prompt strategy.

- `momnitrix/model_runtime.py`
  - Actual model loading/inference wrappers:
    - `CoreGpuRuntime` (MedGemma + MedSigLIP)
    - `DermRuntime`
    - `MedasrRuntime`
  - MedGemma prompt profiles (`text_only`, `voice_vitals`, `wound_image`, `derma_image`, `multimodal_image`).

- `momnitrix/risk.py`
  - Safety policy floor + deterministic heuristic fallback.
  - Includes pregnancy-specific fasting glucose thresholds.

- `momnitrix/storage.py`
  - Artifact/event persistence (S3 if configured, local fallback otherwise).

- `momnitrix/sse.py`
  - SSE formatting helper.

- `momnitrix/utils.py`
  - Risk ranking, chunking, time helpers.

---

## 4) Frontend File Map (Reference implementation to replicate)

Canonical replicated frontend now lives in:
- `web-app/` (same architecture as `qa_console/`)

## Core static files
- `web-app/index.html`
  - UI structure:
    - Input mode cards (Text/Voice/Image)
    - Input values sheet
    - Interaction sheet
    - Results sheet with card carousel
    - SSE timeline

- `web-app/styles.css`
  - Watch-like visual system + sheet/card transitions.

- `web-app/app.js`
  - Bundled output. Do not edit directly.

## TypeScript source
- `web-app/src/main.ts`
  - Startup entrypoint.

- `web-app/src/ui/dom.ts`
  - UI orchestrator (the only module touching DOM APIs).
  - Handles:
    - mode switching
    - sheet open/close
    - validation gate
    - submit + SSE stream handling
    - render results/diagnostics
    - run log export

- `web-app/src/domain/types.ts`
  - Typed contracts (includes Swift mirror comments).

- `web-app/src/domain/constants.ts`
  - Central constants (required rules, event labels, randomizer modes, metadata constants).

- `web-app/src/domain/validation.ts`
  - Pure required/optional input validation.

- `web-app/src/domain/payload.ts`
  - Pure payload composition for `/v1/triage/stream`.

- `web-app/src/domain/sse.ts`
  - Pure SSE parser.

- `web-app/src/domain/formatters.ts`
  - Pure result/timeline formatting.

- `web-app/src/domain/randomizer.ts`
  - Random clinical profile generation.

- `web-app/src/adapters/base64.ts`
  - File/Blob -> base64 helper.

- `web-app/src/adapters/audio.ts`
  - Audio conversion helpers (WAV conversion).

- `web-app/src/adapters/media.ts`
  - Media attachment into triage payload.

- `web-app/src/adapters/streaming.ts`
  - Fetch body streaming adapter.

## Tests and docs
- `web-app/src/__tests__/*`
  - Domain-level test coverage for validation/payload/sse/formatters/randomizer.

- `web-app/README.md`
  - Build/test/run commands.

- `mds/frontend_to_swiftui_mapping.md`
  - Existing TS -> SwiftUI mapping reference.

---

## 5) API Contract You Must Preserve in SwiftUI

Public endpoint to call:
- `POST {API_BASE_URL}/v1/triage/stream`

Health endpoint:
- `GET {API_BASE_URL}/health?probe=1`

Optional retrieval:
- `GET {API_BASE_URL}/v1/triage/{request_id}`

### Required clinical inputs (enforced server-side)
- `patient_context.age_years`
- `vitals.systolic_bp`
- `vitals.diastolic_bp`
- `vitals.fasting_glucose_mmol_l` (or accepted aliases)
- `vitals.temp_c`
- `vitals.hr`

### Request payload shape (minimum)
```json
{
  "request_id": "optional-string",
  "patient_context": {
    "age_years": 25,
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
    "hrv": 42
  },
  "inputs": {
    "headache": false,
    "vision_changes": false,
    "decreased_fetal_movement": false,
    "free_text": "text or transcript context",
    "wound_image_b64": null,
    "skin_image_b64": null,
    "audio_b64": null
  },
  "metadata": {
    "ui_mode": "text",
    "composer_mode": "medgemma_first",
    "medgemma_output_style": "notebook"
  }
}
```

### SSE format
Each event block:
```text
event: <event_name>
data: <json>

```
Also keep-alive comments may be sent:
```text
: keep-alive

```

Key events:
- `request.accepted`
- `router.decision`
- `router.prompt_plan`
- `model.started` / `model.completed` / `model.failed`
- `medgemma.started` / `medgemma.completed` / `medgemma.delta`
- `gemini.started` / `gemini.completed` / `gemini.skipped`
- `diagnostics.inference_breakdown`
- `triage.final`
- `triage.error`

---

## 6) Model Setup Details (if deploying backend)

Current intended model wiring:
- MedGemma base: `google/medgemma-1.5-4b-it`
- MedGemma adapter (LoRA): `tyb343/mamaguard-vitals-lora-p100`
- MedSigLIP fine-tuned model: `tyb343/medsiglip-448-momnitrix-wound`
- Derm model: `google/derm-foundation` + local classifier artifacts
- MedASR model: `google/medasr`
- Gemini planner/composer: `gemini-3-flash-preview`

Important runtime switch:
- `MOMNITRIX_USE_REAL_MODELS=true` to use real models.
- `false` enables deterministic stubs/fallback behavior for dry runs.

---

## 7) Modal Setup (Option B only)

## Prerequisites
```bash
python3 -m pip install --upgrade modal
modal setup
```

## Required secrets

1) `medgemma`
- `GEMINI_API_KEY=<your_gemini_key>`
- optional `HF_TOKEN=<your_hf_token>`

2) `medgemma-hf`
- `HF_TOKEN=<your_hf_token>`

3) `momnitrix-config`
- all env config values used by `momnitrix/config.py`.

Create secrets (example):
```bash
modal secret create medgemma \
  GEMINI_API_KEY="YOUR_GEMINI_API_KEY" \
  HF_TOKEN="YOUR_HF_TOKEN"

modal secret create medgemma-hf \
  HF_TOKEN="YOUR_HF_TOKEN"
```

Create `momnitrix-config` (phase 1, before service URLs are known):
```bash
modal secret create momnitrix-config \
  MOMNITRIX_USE_REAL_MODELS="true" \
  MOMNITRIX_GEMINI_MODEL="gemini-3-flash-preview" \
  MOMNITRIX_RESPONSE_COMPOSER_MODE="medgemma_first" \
  MOMNITRIX_MEDGEMMA_BASE_MODEL_ID="google/medgemma-1.5-4b-it" \
  MOMNITRIX_MEDGEMMA_ADAPTER_ID="tyb343/mamaguard-vitals-lora-p100" \
  MOMNITRIX_MEDGEMMA_IS_ADAPTER="true" \
  MOMNITRIX_MEDSIGLIP_MODEL_ID="tyb343/medsiglip-448-momnitrix-wound" \
  MOMNITRIX_DERM_CLASSIFIER_PATH="/root/artifacts/derm/derm_classifier.pkl" \
  MOMNITRIX_DERM_SCALER_PATH="/root/artifacts/derm/derm_scaler.pkl" \
  MOMNITRIX_DERM_LABELS_PATH="/root/artifacts/derm/derm_labels.json" \
  MOMNITRIX_REQUEST_TIMEOUT_SEC="120" \
  MOMNITRIX_MEDGEMMA_REQUEST_TIMEOUT_SEC="300"
```

## Deploy order
```bash
modal deploy modal_core_gpu.py
modal deploy modal_derm_tf.py
modal deploy modal_medasr.py
```

Get these 3 deployed URLs, then update `momnitrix-config` with:
- `MOMNITRIX_CORE_GPU_BASE_URL`
- `MOMNITRIX_DERM_BASE_URL`
- `MOMNITRIX_MEDASR_BASE_URL`

Then deploy API:
```bash
modal deploy modal_api.py
```

Verify:
```bash
curl -sS "<API_URL>/health?probe=1"
```

### Notes about access/keys
- Client apps (SwiftUI/watchOS) should never hold Gemini/HF/Modal keys.
- Keys live only in Modal secrets.
- Friend only needs API base URL to integrate frontend.

---

## 8) SwiftUI Replication Plan (Practical)

## Architecture target (recommended)
- `Models/` (Codable structs mirroring `web-app/src/domain/types.ts`)
- `Domain/` (pure logic port of validation/payload/sse/formatters/randomizer)
- `Services/`
  - `TriageAPIClient` (HTTP + SSE)
  - `AudioService` (record/transcode)
  - `ImageService` (picker + base64)
- `ViewModels/`
  - `TriageViewModel` (equivalent of `ui/dom.ts` orchestration)
- `Views/`
  - Mode selection
  - Input sheet
  - Interaction sheet
  - Results card carousel
  - Diagnostics timeline

## 1:1 mapping guide
- Use `mds/frontend_to_swiftui_mapping.md` as the primary type/function mapping source.

Core ports:
- `validation.ts` -> `ValidationService.swift`
- `payload.ts` -> `PayloadComposer.swift`
- `sse.ts` -> `SSEParser.swift` (using `URLSession.bytes`)
- `randomizer.ts` -> `Randomizer.swift`
- `formatters.ts` -> `TriageFormatter.swift`

## Important behavior to preserve
- Diagnose button disabled until required inputs are valid.
- Mode-first UX (text/voice/image).
- Input values in a dedicated sheet.
- SSE timeline visible for diagnostics.
- Final triage in card-by-card navigation.
- Download/share run log equivalent (JSON export/share sheet).

## Native replacements for web APIs
- MediaRecorder -> `AVAudioRecorder`
- File input -> `PhotosPicker` / `UIDocumentPicker`
- Fetch stream -> `URLSession.bytes`
- DOM state -> `@State` + `ObservableObject`

---

## 9) Recommended Swift Networking Skeleton

For SSE:
1. `POST /v1/triage/stream`
2. Read bytes incrementally.
3. Buffer text until `\n\n`.
4. Parse `event:` + `data:`.
5. Update view model per event.

Pseudo pattern:
- Maintain `events: [SSEEvent]`
- Maintain `finalResponse: TriageFinalResponse?`
- Maintain diagnostics:
  - cold start
  - warmup ms
  - medgemma inference ms
  - total latency

---

## 10) Validation/Test Checklist Before Handoff

## Backend smoke
1. `GET /health?probe=1` returns configured + reachable status.
2. `POST /v1/triage/stream` returns SSE and ends with `triage.final`.

## SwiftUI parity checks
1. Text-only run works.
2. Voice run includes transcript in specialist outputs.
3. Wound image run includes wound specialist evidence.
4. Skin image run includes derm specialist evidence.
5. Required field gate works.
6. Results carousel and timeline update correctly.

## Regression checks
1. No backend contract key changes.
2. No client-side secrets.
3. No direct calls from app to `/internal/*` endpoints.

---

## 11) Common Pitfalls

1. Sending temperature as degF in payload:
- Backend expects `temp_c`.
- Convert before sending.

2. Missing required aliases:
- Prefer sending `fasting_glucose` (accepted by schema alias).

3. Assuming all responses are only final JSON:
- This endpoint is streaming SSE; parse incrementally.

4. App-side key leakage:
- Never ship Gemini/HF token in Swift code.

5. Misinterpreting model ownership:
- Risk/actions from MedGemma + policy floor.
- Final patient copy may be Gemini (if composer mode enabled).
- Use diagnostics event to inspect model share.

---

## 12) What to Ask Tayyab Before Starting

1. Current public API URL to use.
2. Desired composer mode for demo (`medgemma_first` or `gemini_full`).
3. Whether friend should use shared deployed backend or their own Modal deployment.
4. Whether to expose diagnostics UI in final watch demo build or hide under debug mode.

---

## 13) Reference Docs

- `README.md` (root backend setup and deploy flow)
- `web-app/README.md` (frontend TS build/test/run)
- `mds/frontend_to_swiftui_mapping.md` (TS -> Swift mapping)
- `mds/frontend_agent_handoff.md` (frontend architecture context)
- `mds/momnitrix_modal_backend_spec_v2.md` (system topology overview)

