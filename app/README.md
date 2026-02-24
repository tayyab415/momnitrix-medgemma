# Momnitrix Backend (Modal)

This repository now implements a production-shaped Momnitrix backend with:

- `modal_api.py`: public orchestration API (`/v1/triage/stream`)
- `modal_core_gpu.py`: fine-tuned MedGemma 1.5 4B + fine-tuned MedSigLIP container
- `modal_derm_tf.py`: Derm Foundation + sklearn classifier container
- `modal_medasr.py`: MedASR container
- `momnitrix/`: shared orchestration, schema, risk, storage, and gateway logic
- `tests/`: unit/integration-style orchestration tests

Gemini orchestration model defaults to `gemini-3-flash-preview`.

## Investigation Guide

For architecture analysis, tuning history, incident fixes, and audit/troubleshooting prompts:

- `mds/momnitrix_backend_frontend_tuning_investigation_guide.md`

## 1) Prerequisites

```bash
python3 -m pip install --upgrade modal
modal setup
```

## 2) Environment

Set these for deployed wiring:

```bash
export MOMNITRIX_MEDGEMMA_BASE_MODEL_ID="google/medgemma-1.5-4b-it"
export MOMNITRIX_MEDGEMMA_ADAPTER_ID="tyb343/mamaguard-vitals-lora-p100"
export MOMNITRIX_MEDGEMMA_IS_ADAPTER="true"
export MOMNITRIX_MEDSIGLIP_MODEL_ID="tyb343/medsiglip-448-momnitrix-wound"
export MOMNITRIX_USE_REAL_MODELS="true"
export MOMNITRIX_RESPONSE_COMPOSER_MODE="gemini_full"  # or medgemma_first

export MOMNITRIX_CORE_GPU_BASE_URL="<core-gpu-url>"
export MOMNITRIX_DERM_BASE_URL="<derm-url>"
export MOMNITRIX_MEDASR_BASE_URL="<medasr-url>"

export MOMNITRIX_S3_BUCKET="<bucket-name>"
export MOMNITRIX_S3_REGION="us-east-1"
export MOMNITRIX_S3_PREFIX="momnitrix"

export MOMNITRIX_DERM_CLASSIFIER_PATH="artifacts/derm/derm_classifier.pkl"
export MOMNITRIX_DERM_SCALER_PATH="artifacts/derm/derm_scaler.pkl"
export MOMNITRIX_DERM_LABELS_PATH="artifacts/derm/derm_labels.json"
```

You can also override composer mode per request without redeploying:

```json
{
  "metadata": {
    "composer_mode": "medgemma_first"
  }
}
```

Create the Modal secret consumed by all services:

```bash
modal secret create medgemma \
  GEMINI_API_KEY="<your-gemini-key>" \
  HF_TOKEN="<optional-hf-token>"
```

## 3) Deploy services

```bash
modal deploy modal_core_gpu.py
modal deploy modal_derm_tf.py
modal deploy modal_medasr.py
modal deploy modal_api.py
```

For local dry-run behavior without real model loading:

```bash
export MOMNITRIX_USE_REAL_MODELS="false"
modal serve modal_api.py
```

## 4) Test the streaming endpoint

```bash
curl -N -X POST "<MOMNITRIX_API_URL>/v1/triage/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_context": {"gestational_weeks": 34},
    "vitals": {"systolic_bp": 148, "diastolic_bp": 94},
    "inputs": {
      "headache": true,
      "vision_changes": true,
      "wound_image_b64": "ZmFrZQ=="
    }
  }'
```

SSE now includes an explicit attribution event:

- `diagnostics.inference_breakdown` with:
- `model=medgemma:<x>%|gemini:<y>%` (latency share)
- `medgemma_engine` (`medgemma.core_gpu` vs heuristic fallback)
- `field_authorship` (which model authored risk/actions/message fields)
- `gpu_warmup_ms` (cold-start model load time)
- `medgemma_inference_ms` (actual MedGemma infer+parse path time after warmup)

## 5) Run tests

```bash
python3 -m unittest discover -s tests -v
```

## 6) Benchmark Gemini vs MedGemma-first paths

Run A/B benchmarks on the same cases and emit a report:

```bash
python3 scripts/benchmark_composer_paths.py \
  --api-base-url "https://<your-api>.modal.run" \
  --maternal-csv "/Users/tayyabkhan/Downloads/medgemma/Maternal Health Risk Data Set.csv" \
  --limit 30 \
  --lora-core-url "https://<your-core-lora>.modal.run"
```

Outputs:

- `reports/benchmark_path_report.json`
- `reports/benchmark_path_report.md`

## 7) Local QA Console (watch-simulation frontend)

The local QA frontend lives in `qa_console/` and supports:

- manual profile + vitals entry
- realistic bounded randomization (normal/borderline/red-flag profiles)
- wound + skin image upload
- audio upload and browser mic recording
- SSE event timeline and final triage viewer
- downloadable JSON run logs for audit/debugging

Run it locally:

```bash
cd qa_console
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Make sure your backend URL is set in the UI (defaults to the deployed Modal endpoint).

## 8) Dataset/Input Audit (CSV + JSON)

Audit input files against `qa_console/input_contract.json`:

```bash
python3 scripts/audit_inputs.py \
  --csv /path/to/inputs.csv \
  --json /path/to/scenarios.json
```

Outputs:

- `reports/input_audit_report.md`
- `reports/input_audit_report.json`

Use this report to tighten randomizer bounds and validate dataset realism before demos.

For documented range choices and references:

- `qa_console/INPUT_RANGE_BASIS.md`

## Legacy

`modal_sandbox.py` is kept only as an early prototype and is no longer the primary API.
