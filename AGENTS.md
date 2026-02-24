# MamaGuard: MedGemma Fine-Tuning for Maternal Health Risk Assessment

## Project Overview

This project implements **MamaGuard**, a maternal health risk assessment AI system built by fine-tuning Google's MedGemma-4B-IT model. The system takes pregnancy vital signs (blood pressure, glucose, temperature, heart rate) as input and generates structured clinical risk assessments with risk level classification (LOW/MID/HIGH), clinical reasoning, potential complications, recommended actions, and warning signs.

### Key Components

1. **Data Preparation Pipeline** (`prepare_training_data.py`): Converts raw maternal health CSV data into instruction-tuning format with synthetic pregnancy context and clinical responses
2. **Fine-Tuning Notebooks**: Multiple Jupyter notebooks for LoRA fine-tuning experiments on different platforms (local, Kaggle, Colab)
3. **Surgical Wound Dataset Extractor** (`scripts/extract_surgwound.py`): Utility to extract and process the SurgWound dataset for potential multi-modal extensions

## Technology Stack

- **Base Model**: Google MedGemma-4B-IT (4B parameter instruction-tuned medical LLM)
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation) via PEFT library
- **Training Framework**: HuggingFace Transformers + TRL
- **Data Processing**: pandas, datasets, tqdm
- **Optional Teacher Model**: Google Gemini Flash (for high-quality response generation)
- **Environment**: Python 3.12 with virtual environment (`.venv`)

### Key Dependencies

```
transforms>=4.46.0
peft>=0.13.0
trl>=0.12.0
bitsandbytes>=0.44.0
datasets
accelerate
google-genai
huggingface_hub
pandas
torch
pillow
```

## Project Structure

```
.
├── prepare_training_data.py          # Main data preparation script
├── _check_lengths.py                 # Utility to validate response lengths
├── scripts/
│   └── extract_surgwound.py          # SurgWound dataset extractor
├── data/
│   └── surgwound/                    # Extracted surgical wound data
├── Maternal Health Risk Data Set.csv # Source maternal health dataset (~1014 rows)
├── mamaguard_train.jsonl             # Training data (JSONL format)
├── mamaguard_eval.jsonl              # Evaluation data (JSONL format)
├── medgemma_lora_finetune.ipynb      # Main fine-tuning notebook (local)
├── medgemma_lora_finetune_kaggle.ipynb # Kaggle-optimized notebook
├── medgemma-lora-finetune-kaggle-fixed-outputs.ipynb # Fixed Kaggle version
├── medgemma-lora-fine-tuning-with-gemini-as-a-teacher.ipynb # Gemini teacher notebook
├── starter-nb-fine-tune-medgemma-1-5-for-detection.ipynb # Detection task starter
├── medgemma-1-5-fine-tuning-for-covid-19-cough-detect.ipynb # COVID detection variant
├── output*/                           # Generated training data outputs
│   ├── maternal_health_dataset/      # HuggingFace dataset format
│   ├── maternal_health_train.jsonl
│   ├── maternal_health_eval.jsonl
│   └── data_summary.json
└── enrichment_checkpoint*.json       # Resume checkpoints for Gemini enrichment
```

## Data Pipeline

### Input Data Format

Source: `Maternal Health Risk Data Set.csv` (UCI Machine Learning Repository)

Columns:
- `Age`: Patient age in years
- `SystolicBP`: Systolic blood pressure (mmHg)
- `DiastolicBP`: Diastolic blood pressure (mmHg)
- `BS`: Blood sugar/glucose (mmol/L)
- `BodyTemp`: Body temperature (°F)
- `HeartRate`: Heart rate (bpm)
- `RiskLevel`: Target label (low risk/mid risk/high risk)

### Data Preparation Pipeline

1. **Load and Validate**: Reads CSV, validates required columns, checks class distribution
2. **Add Synthetic Context**: Generates realistic pregnancy metadata:
   - Gestational week (8-40)
   - Trimester derivation
   - Gravida/Para (pregnancy history)
   - BMI category based on age
3. **Build Instruction Prompts**: Creates varied clinical instruction templates (4 different formats)
4. **Generate Responses**: 
   - **Gemini Mode** (optional): Uses Gemini Flash with retries and quality validation
   - **Template Mode** (default): Rule-based fallback with clinical reasoning logic
5. **Format for Gemma**: Applies Gemma 3 chat template (`<start_of_turn>user/model<end_of_turn>`)
6. **Split and Export**: 90/10 train/test split, outputs HuggingFace dataset + JSONL

### Response Format Requirements

All generated responses MUST contain:
- First line: `RISK LEVEL: {LOW|MID|HIGH}`
- Section: `CLINICAL REASONING` or `CLINICAL ASSESSMENT`
- Section: `POTENTIAL COMPLICATIONS`
- Section: `RECOMMENDED ACTIONS` (with bullet points)
- Section: `WARNING SIGNS`

Quality gates:
- Minimum 800 characters
- Maximum 2000 characters
- At least 3 bullet-point actions
- No AI disclaimers or "consult your doctor" phrases

## Build and Run Commands

### Data Preparation (Template Mode)

```bash
# Basic usage - template-based response generation
python prepare_training_data.py \
    --input "Maternal Health Risk Data Set.csv" \
    --output-dir ./output \
    --seed 42
```

### Data Preparation with Gemini Enrichment

```bash
# Requires GEMINI_API_KEY in .env or environment
python prepare_training_data.py \
    --input "Maternal Health Risk Data Set.csv" \
    --output-dir ./output_gemini_high \
    --use-gemini \
    --gemini-model gemini-3-flash-preview \
    --thinking-level HIGH \
    --checkpoint ./enrichment_checkpoint_gemini_high.json \
    --min-response-chars 800 \
    --max-response-chars 2000
```

### SurgWound Dataset Extraction

```bash
python scripts/extract_surgwound.py --output_dir data/surgwound

# Smoke test with limited images
python scripts/extract_surgwound.py --output_dir data/surgwound --limit 50
```

### Response Length Validation

```bash
python _check_lengths.py
```

## Fine-Tuning Configuration

### LoRA Hyperparameters

```python
LORA_CONFIG = {
    "r": 16,                    # LoRA rank (low = faster, high = more expressive)
    "lora_alpha": 32,           # Scaling factor (typically 2*r)
    "lora_dropout": 0.05,       # Dropout for regularization
    "target_modules": [         # Which layers to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}
```

### Training Hyperparameters

```python
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "fp16": True,               # Mixed precision training
    "optim": "paged_adamw_8bit", # Memory-efficient optimizer
}
```

### Platform-Specific Configurations

The project includes auto-detection for different GPU environments:

| GPU | Batch Size | Grad Accum | Max Length | Flash Attention |
|-----|------------|------------|------------|-----------------|
| A100 (40GB+) | 4 | 2 | 2048 | Yes |
| L4/A10/P100 | 2 | 4 | 2048 | Yes (except P100) |
| T4/Default | 1 | 8 | 1536 | No |

## Code Style Guidelines

### Python Code Style

- **Type hints**: Use Python 3.9+ type hints (`list[dict[str, Any]]`, `str | None`)
- **Docstrings**: Google-style docstrings for functions and classes
- **Constants**: UPPER_SNAKE_CASE for module-level constants (e.g., `RISK_CANONICAL`, `REQUIRED_RESPONSE_SECTIONS`)
- **Dataclasses**: Use `@dataclass` for structured data (e.g., `SampleRecord`, `SyntheticContext`)
- **Path handling**: Use `pathlib.Path` instead of string paths
- **Random seeding**: Derive per-row RNG from base seed: `random.Random(seed * 1_000_003 + row_index)`

### Naming Conventions

- Functions: `snake_case`
- Classes: `PascalCase`
- Private/internal: `_leading_underscore` or module-level prefix
- Checkpoint keys: Use string representation of row indices for JSON compatibility

### Error Handling

- Use explicit exception types in `try/except` blocks
- Log warnings with `[WARN]` prefix, errors with `[ERROR]`
- Implement exponential backoff for API calls (2s, 4s, 8s)
- Graceful degradation: Template fallback when Gemini fails

## Testing and Validation

### Quality Validation (Automatic)

The `validate_outputs()` function automatically checks:

1. **Prefix validation**: All responses start with "RISK LEVEL: {LOW|MID|HIGH}"
2. **Length validation**: Responses between 100-2000 characters
3. **Disallowed phrases**: No AI disclaimers
4. **Class distribution**: Train/test splits match overall distribution (±10% tolerance)
5. **Section presence**: Required sections present in response

### Manual Review

Sample examples are automatically exported to `output/sample_examples.txt` with 3 examples per risk level for human review.

### Checkpoint Resume

The data preparation script supports resumable processing:
- Saves checkpoint every 100 rows
- Validates cached entries against quality gates
- Allows switching between Gemini and template modes while preserving progress

## Environment Configuration

### Required Environment Variables

Create `.env` file in project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt  # If available
# Or install core packages:
pip install transformers peft trl bitsandbytes datasets accelerate google-genai pandas torch pillow
```

## Security Considerations

1. **API Keys**: Never commit `.env` file. The existing `.env` in repo contains sample keys that should be rotated.
2. **Model Outputs**: Generated medical responses are for research purposes only; not for clinical use without expert validation.
3. **Data Privacy**: The maternal health dataset is from UCI ML Repository (public domain), but treat any real patient data with HIPAA/GDPR compliance.
4. **Disallowed Phrases**: The system explicitly filters out AI disclaimers to prevent false reassurance in medical contexts.

## Common Tasks

### Regenerate Training Data from Scratch

```bash
rm -f enrichment_checkpoint.json
python prepare_training_data.py --output-dir ./output --seed 42
```

### Convert JSONL to CSV for Analysis

```python
import pandas as pd
import json

data = [json.loads(line) for line in open('mamaguard_train.jsonl')]
df = pd.DataFrame(data)
df.to_csv('mamaguard_train.csv', index=False)
```

### Merge LoRA Weights for Deployment

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")

# Load and merge LoRA weights
model = PeftModel.from_pretrained(base_model, "./medgemma-lora-maternal-health")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./medgemma-maternal-health-merged")
```

## Troubleshooting

### Out of Memory During Training

- Reduce `MAX_SEQ_LENGTH` to 1536 or 1024
- Increase `gradient_accumulation_steps`, reduce `per_device_train_batch_size`
- Enable 8-bit quantization: `load_in_8bit=True`

### Gemini API Failures

- Check `GEMINI_API_KEY` is set correctly
- Verify `google-genai` is installed: `pip install google-genai`
- The script automatically falls back to template mode if Gemini is unavailable

### Checkpoint Corruption

If checkpoint file becomes corrupted:
```bash
rm enrichment_checkpoint.json
# Re-run the script - it will regenerate from scratch
```

## References

- **MedGemma**: https://developers.google.com/google-for-health/ai-solutions
- **Maternal Health Dataset**: UCI Machine Learning Repository
- **SurgWound Dataset**: https://huggingface.co/datasets/xuxuxuxuxu/SurgWound
- **LoRA Paper**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
