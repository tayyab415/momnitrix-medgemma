# MamaGuard & Momnitrix Project

## Project Overview

This project encompasses **MamaGuard**, a maternal health risk assessment AI system built by fine-tuning Google's MedGemma-4B-IT model, and **Momnitrix**, its production-shaped backend deployed on Modal. 

The system takes pregnancy vital signs (blood pressure, glucose, temperature, heart rate) and patient context as input to generate structured clinical risk assessments (LOW/MID/HIGH), clinical reasoning, potential complications, and recommended actions. It also integrates a Derm Foundation classifier and MedASR capabilities.

### Key Components

1. **Data Preparation Pipeline** (`prepare_training_data.py`): Converts raw maternal health CSV data into instruction-tuning format with synthetic pregnancy context. Optionally uses Google Gemini Flash for high-quality response generation.
2. **Fine-Tuning Notebooks**: Several Jupyter notebooks (`medgemma_lora_finetune*.ipynb`) for LoRA fine-tuning experiments across different environments (local, Kaggle).
3. **Surgical Wound Extractor** (`scripts/extract_surgwound.py`): Extracts and processes the SurgWound dataset for multi-modal extensions.
4. **Momnitrix Backend** (`app/`): A Modal-based microservices architecture containing:
   - `modal_api.py`: Public orchestration API endpoint (`/v1/triage/stream`).
   - `modal_core_gpu.py`: Container running the fine-tuned MedGemma 1.5 4B and MedSigLIP.
   - `modal_derm_tf.py`: Container for Derm Foundation + sklearn classifier.
   - `modal_medasr.py`: Container for MedASR.

## Technology Stack

- **Base Models**: Google MedGemma-4B-IT, MedSigLIP, Gemini Flash (Teacher model)
- **Fine-Tuning**: LoRA (PEFT), HuggingFace Transformers, TRL, bitsandbytes
- **Backend Infrastructure**: Modal (Serverless GPU containers)
- **Language**: Python 3.12
- **Data Tools**: pandas, datasets, torch

## Building and Running

### 1. Data Preparation

Generate training data from the source CSV (`Maternal Health Risk Data Set.csv`):

```bash
# Template Mode
python prepare_training_data.py --input "Maternal Health Risk Data Set.csv" --output-dir ./output --seed 42

# Gemini Enrichment Mode (Requires GEMINI_API_KEY)
python prepare_training_data.py --input "Maternal Health Risk Data Set.csv" --output-dir ./output_gemini_high --use-gemini --gemini-model gemini-3-flash-preview --thinking-level HIGH
```

### 2. Backend Deployment (Modal)

Prerequisites: Install Modal and configure the environment variables as specified in `app/README.md`.

```bash
# Install Modal
python3 -m pip install --upgrade modal
modal setup

# Create necessary secrets
modal secret create medgemma GEMINI_API_KEY="<your-gemini-key>" HF_TOKEN="<optional-hf-token>"

# Deploy the services
cd app
modal deploy modal_core_gpu.py
modal deploy modal_derm_tf.py
modal deploy modal_medasr.py
modal deploy modal_api.py
```

To run a local dry-run without loading real models:
```bash
export MOMNITRIX_USE_REAL_MODELS="false"
modal serve modal_api.py
```

### 3. Testing

Run the orchestration unit tests:
```bash
cd app
python3 -m unittest discover -s tests -v
```

Test the streaming endpoint via curl:
```bash
curl -N -X POST "<MOMNITRIX_API_URL>/v1/triage/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_context": {"gestational_weeks": 34},
    "vitals": {"systolic_bp": 148, "diastolic_bp": 94},
    "inputs": {"headache": true, "vision_changes": true}
  }'
```

## Development Conventions

- **Typing & Docs**: Use Python 3.9+ type hints (`list[dict[str, Any]]`) and Google-style docstrings.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- **Data Structures**: Utilize `@dataclass` for structured data representation.
- **Error Handling**: Implement explicit exception types, log appropriately (`[WARN]`, `[ERROR]`), and use exponential backoff for external API calls (e.g., Gemini).
- **Paths**: Use `pathlib.Path` instead of raw string paths.
