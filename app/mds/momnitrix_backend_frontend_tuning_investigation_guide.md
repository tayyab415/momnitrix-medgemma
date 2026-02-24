# Momnitrix Backend + Frontend Tuning Investigation Guide

Last updated: 2026-02-22

## 1) Purpose and Audience

This document is for:

- codebase analysis
- incident investigation
- audit and troubleshooting by humans and AI agents

It explains:

- what was changed and tuned in backend and frontend
- how request routing works under the hood
- which files matter and how they integrate
- incidents observed, fixes applied, and remaining risk areas
- guidance questions for follow-up review

## 2) Current System Snapshot

Project identity:

- Product name: `Momnitrix`
- Core use case: maternal triage assistant with multimodal inputs (structured vitals, text, wound image, skin image, audio)

Live Modal services:

- Public API: `momnitrix-api-v2` (`modal_api.py`)
- Core GPU: `momnitrix-core-gpu` (`modal_core_gpu.py`)
- Derm service: `momnitrix-derm-tf` (`modal_derm_tf.py`)
- MedASR service: `momnitrix-medasr` (`modal_medasr.py`)

As configured in code now:

- `modal_core_gpu.py`: GPU `L4`, `min_containers=0`, `max_containers=1`
- `modal_api.py`: CPU `2`, `min_containers=1`
- `modal_derm_tf.py`: GPU `T4`, `min_containers=1`
- `modal_medasr.py`: CPU `4`, `min_containers=0`

## 3) High-Level Request Flow (Under the Hood)

Entry endpoint:

- `POST /v1/triage/stream` in `modal_api.py`

Pipeline:

1. Validate payload to `TriageStreamRequest` (`momnitrix/schemas.py`).
2. Emit SSE `request.accepted`.
3. Persist raw artifacts (if present) using `ArtifactStore` (`momnitrix/storage.py`).
4. Dispatch specialists in parallel from orchestrator (`momnitrix/orchestration.py`):
   - wound image -> `medsiglip`
   - skin image -> `derm`
   - audio -> `medasr`
5. Call MedGemma decision step (`/internal/medgemma/risk_decide`) through gateway.
6. Compute deterministic policy floor (`momnitrix/risk.py`).
7. Merge deterministic guardrail reasons/actions with MedGemma output.
8. Compose patient message and visit summary with Gemini (`momnitrix/gemini.py`).
9. Stream `gemini.delta` chunks to frontend.
10. Emit and persist final `triage.final` JSON.

Core property:

- Specialist model failures do not terminate whole triage. The pipeline continues, records failed branch event, and still returns a final response.

## 4) What Was Tuned (Backend)

This section describes material changes made during investigation and hardening.

### 4.1 Safety and Risk Classification

Files:

- `momnitrix/schemas.py`
- `momnitrix/risk.py`

Changes:

- Added glucose aliases so varied payload keys still parse:
  - `fasting_glucose`
  - `fasting_plasma_glucose`
  - `fasting_glucose_mmol`
  - `fasting_glucose_mmol_l`
- Added pregnancy-specific glucose policy logic:
  - target: `< 5.3 mmol/L`
  - diabetes-range escalation at `>= 7.0 mmol/L`
  - severe escalation at `>= 10.0 mmol/L`
- Added free-text glucose extraction as backup.

Impact:

- Glucose-heavy scenarios no longer get masked by "no acute flags" behavior.
- Risk floor can escalate independently of model prose quality.

### 4.2 Orchestration Guardrail Merge

File:

- `momnitrix/orchestration.py`

Changes:

- Added merging function to prioritize deterministic floor + heuristic findings.
- De-duplicates reasons/actions and keeps them bounded.
- Excludes repetitive "safety floor escalation" strings from polluted outputs.

Impact:

- Final message rationale is less dependent on noisy generative sections.
- High-risk vitals are surfaced first.

### 4.3 MedGemma Runtime and Parsing Cleanup

File:

- `momnitrix/model_runtime.py`

Changes:

- Added output sanitization helpers:
  - strips turn artifacts (`<start_of_turn>`, `<end_of_turn>`)
  - strips unused tokens (`<unusedXX>`)
  - removes heading-only fragments
  - filters malformed unit/value fragments
- Improved section parsing and list extraction from generated text.
- Kept fallback parser for both JSON-like and sectioned text outputs.
- Enabled inference cache:
  - `self._medgemma_model.config.use_cache = True`
- Generation set to deterministic mode:
  - `do_sample=False`
  - bounded `max_new_tokens`
  - `repetition_penalty`, `no_repeat_ngram_size`

Impact:

- Lower garbage token leakage in `action_items` and `medgemma_reasons`.
- More stable extraction of actionable fields.

### 4.4 Gateway Resilience

File:

- `momnitrix/gateway.py`

Changes:

- Dedicated MedGemma timeout uses `MOMNITRIX_MEDGEMMA_REQUEST_TIMEOUT_SEC`.
- Added redirect handling:
  - `follow_redirects=True` in HTTP client.

Incident addressed:

- Intermittent `303 See Other` from Modal function URL caused fallback path previously.

### 4.5 Gemini Integration and Fallback Behavior

Files:

- `momnitrix/gemini.py`
- `momnitrix/config.py`
- `modal_api.py`

Changes:

- Model default uses `gemini-3-flash-preview`.
- Retry to `gemini-3-flash` on `401/403/404`.
- Safer fallback logging avoids printing full key-bearing URL in current code path.
- Compact fallback response formatter if Gemini fails.
- Config now accepts multiple key env names:
  - `GEMINI_API_KEY`
  - `Gemini_API_Key`
  - `GEMINI_API_Key`
  - `gemini_api_key`
- API health includes `gemini_key_configured` boolean.

Impact:

- Reduced configuration fragility for secret naming differences.
- Better operational visibility from `/health`.

### 4.6 Infra and Deployment Tuning

Files:

- `modal_core_gpu.py`
- `modal_api.py`

Changes:

- Core GPU upgraded from `T4` to `L4`.
- API remains warm (`min_containers=1`) to reduce orchestration cold-path startup.

Tradeoff:

- Core currently remains `min_containers=0` for cost control, so cold starts still happen.

## 5) What Was Tuned (Frontend)

### 5.1 QA Console

Files:

- `qa_console/index.html`
- `qa_console/app.js`
- `qa_console/README.md`
- `qa_console/INPUT_RANGE_BASIS.md`

Changes and behavior:

- Default backend URL points to `momnitrix-api-v2`.
- Manual entry + bounded randomization for realistic watch simulation.
- Payload includes:
  - structured vitals
  - symptom flags
  - free-text context (age, G/P, BMI)
  - optional image/audio base64
- SSE timeline parser records full event stream for audit.
- Downloadable run log JSON includes request, event stream, final response.
- Temperature input in `degF` converted client-side to `temp_c`.

### 5.2 Kimi Frontend

Files:

- `kimi-frontend/index.html`
- `kimi-frontend/app.js`
- `kimi-frontend/README.md`

Behavior:

- Similar watch-simulation UX with bounded ranges and multimodal upload.
- Streams orchestrator SSE and displays timeline + final JSON.

### 5.3 Next.js Simulation (gemini-creations)

Files:

- `gemini-creations/frontend/components/SimulationForm.tsx`
- `gemini-creations/backend/api_client.ts`
- `gemini-creations/orchestrator/router.ts`

Behavior:

- Direct SSE submission to `/v1/triage/stream`.
- Includes local pre-flight edge logic and modality route map helpers.
- Uses `NEXT_PUBLIC_MODAL_API_URL` with fallback to deployed API URL.

## 6) File Integration Map

### 6.1 Entry and Wiring

- `modal_api.py`
  - public API surface
  - SSE streaming
  - request validation and orchestrator invocation
  - CORS allowlist for localhost ports
- `momnitrix/config.py`
  - central env settings contract
- `momnitrix/schemas.py`
  - strict runtime request/response model validation

### 6.2 Orchestration Core

- `momnitrix/orchestration.py`
  - parallel specialist fan-out
  - MedGemma call
  - policy-floor merge
  - Gemini compose
  - final response assembly
- `momnitrix/gateway.py`
  - service-to-service HTTP bridge
  - fallback behavior if internal service unavailable
- `momnitrix/risk.py`
  - deterministic clinical floor and heuristic backup

### 6.3 Model Runtime

- `modal_core_gpu.py`
  - serves MedGemma + MedSigLIP internal endpoints
- `momnitrix/model_runtime.py`
  - model loading, generation, parsing, sanitization
- `modal_derm_tf.py`
  - derm model wrapper endpoint
- `modal_medasr.py`
  - ASR wrapper endpoint

### 6.4 Persistence and SSE

- `momnitrix/storage.py`
  - local fallback persistence
  - optional S3 writes when configured
- `momnitrix/sse.py`
  - SSE frame formatting

### 6.5 Frontend and Simulation

- `qa_console/*`
  - primary local audit console for runs and event timeline
- `kimi-frontend/*`
  - alternate static UI for simulation
- `gemini-creations/*`
  - Next.js simulation path and local route helpers

### 6.6 Tests

- `tests/test_risk.py`
  - glucose and policy-floor regression tests
- `tests/test_medgemma_parsing.py`
  - parsing/sanitization regressions for MedGemma outputs
- `tests/test_orchestration.py`
  - orchestration flow and guardrail priority checks
- `tests/test_smoke_unittest.py`
  - smoke-style orchestration behavior

## 7) Incidents Faced and Fixes

### Incident A: Glucose severity was under-reflected

Symptoms:

- Cases with high fasting glucose were being marked too low.

Root cause:

- Input key mismatch + insufficient deterministic glucose floor.

Fixes:

- Added schema aliases and explicit pregnancy glucose policy thresholds.
- Guardrail merge now prioritizes floor findings.

### Incident B: Noisy MedGemma outputs polluted fields

Symptoms:

- Duplicated sections, turn markers, `<unusedXX>`, malformed line fragments.

Root cause:

- Section parser accepted raw generated artifacts too permissively.

Fixes:

- Added text sanitation, heading filtering, dedupe, artifact stripping.

### Incident C: Gemini fallback and auth confusion

Symptoms:

- Frequent fallback messages due key/model auth path issues.

Root cause:

- Secret key naming inconsistency and model access mismatch in some runs.

Fixes:

- Support multiple key env names.
- Retry from preview model to non-preview model.
- Exposed `gemini_key_configured` in health.

### Incident D: Gateway failed on redirect response

Symptoms:

- API logged fallback due `303 See Other` from internal core URL.

Root cause:

- HTTP client not following redirects.

Fix:

- Enabled `follow_redirects=True` in gateway client.

### Incident E: Cold-start latency spikes

Symptoms:

- Response times ranging from fast warm runs (~20-35s) to long cold runs (~80-150s).

Root cause:

- Core GPU service set to `min_containers=0`, causing model fetch/load on cold starts.

Status:

- Not fully solved. This is an explicit cost/latency tradeoff.

## 8) Current Gaps and Refinement Targets

### 8.1 Major Operational Risks

1. Latency variability remains high because core container is not always warm.
2. Some generated action items can still be truncated in edge cases.
3. S3 persistence is not configured in active health status (`s3_configured=false`), so durability is limited.
4. There are stale docs in repo with outdated infra values (for example old `T4` references).

### 8.2 Clinical Output Quality Risks

1. Model prose can still drift from perfect unit rigor.
2. Long generation can inject non-essential text; deterministic floor helps but does not eliminate all verbosity artifacts.

### 8.3 Suggested Next Technical Improvements

1. Set `min_containers=1` on core L4 for stable latency.
2. Add final post-processor to drop incomplete sentence fragments in `action_items`.
3. Add structured validation rule to reject or trim suspiciously short trailing action strings.
4. Wire S3 for persistent audit trail storage.
5. Add p95 latency telemetry and event-level error counters.

## 9) Troubleshooting Playbook

### 9.1 Quick Health Check

Use:

- `GET /health` on API, core, derm, medasr.

Verify:

- `gemini_key_configured=true` on API
- `real_models=true` on model services
- configured base URLs are non-empty from API health

### 9.2 Stream Flow Check

Expected event order:

1. `request.accepted`
2. optional `artifact.uploaded`
3. optional specialist `model.started` and `model.completed`
4. `medgemma.started`
5. `medgemma.completed`
6. `gemini.started`
7. one or more `gemini.delta`
8. `gemini.completed`
9. `triage.final`

If missing `triage.final`:

- inspect `triage.error` event payload
- inspect API logs for upstream timeout/fallback

### 9.3 If Risk Looks Wrong

Check in order:

1. request payload key names for glucose field aliasing
2. `policy_floor` in final JSON
3. `medgemma_reasons` ordering
4. `risk.py` threshold logic and request units

### 9.4 If Multimodal Branch Not Triggering

Check:

1. base64 exists in `inputs` fields
2. SSE events include `model.started` for expected specialists
3. `specialist_outputs` contains wound/skin/transcript objects

## 10) Questions for AI Agents and Reviewers

Use these prompts when asking for guided help:

1. "Given this `triage.final` JSON, identify whether final risk was model-driven or policy-floor-driven."
2. "Trace this request_id end-to-end and summarize every event with latency."
3. "Compare input vitals against `risk.py` thresholds and report any mismatch in returned risk."
4. "Check whether `medgemma_reasons` are clinically coherent and free of unit hallucinations."
5. "Audit whether specialist model outputs were actually consumed by MedGemma prompt assembly."
6. "Find stale docs/config references that no longer match deployed runtime."
7. "List top 3 code changes to reduce p95 latency without changing clinical logic."

## 11) Known Documentation Inconsistencies to Keep in Mind

1. Some existing markdown/docs still reference older runtime values (for example core `T4`), while code is now `L4`.
2. Some historic notes mention model naming states that were later changed.
3. This file should be treated as the primary investigation baseline for current code.

## 12) Related Files for Deeper Review

- `modal_api.py`
- `modal_core_gpu.py`
- `modal_derm_tf.py`
- `modal_medasr.py`
- `momnitrix/orchestration.py`
- `momnitrix/model_runtime.py`
- `momnitrix/risk.py`
- `momnitrix/gateway.py`
- `momnitrix/gemini.py`
- `momnitrix/config.py`
- `momnitrix/storage.py`
- `momnitrix/schemas.py`
- `qa_console/app.js`
- `qa_console/INPUT_RANGE_BASIS.md`
- `kimi-frontend/app.js`
- `gemini-creations/frontend/components/SimulationForm.tsx`
- `tests/test_risk.py`
- `tests/test_medgemma_parsing.py`
- `tests/test_orchestration.py`

