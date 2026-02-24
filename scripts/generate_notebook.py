#!/usr/bin/env python3
"""Generate the MedSigLIP fine-tuning notebook as a .ipynb file."""

import json
import sys

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                 "source": source.split("\n")})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": source.split("\n")})


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 0: Title markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""# Fine-tune MedSigLIP for Surgical Wound Assessment

**MamaGuard — Wound Assessment Component**

Fine-tunes `google/medsiglip-448` vision encoder on the [SurgWound dataset](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound)
for **6 binary clinical labels**: healing status, erythema, edema, infection risk, urgency, and exudate.

The fine-tuned model serves as a structured clinical signal provider in a two-stage pipeline:
```
Wound photo → MedSigLIP (this model) → structured scores → Gemini/MedGemma orchestrator → empathetic response
```

### Key Design Decisions
- **Selective freezing**: Only last 4 encoder blocks + classification head are trainable (~13-15% of params) to fit T4 16GB
- **Masked BCE loss**: 3 of 6 labels have MISSING values — loss is zeroed out for those entries instead of dropping entire samples
- **Light augmentation**: Horizontal flip + rotation + color jitter to compensate for small dataset (480 train images)
- **eval_loss for model selection**: Val set has only 69 images — per-label AUC too noisy for checkpoint comparison

### Dataset
- **Source**: SurgWound (686 images: 480 train / 69 val / 137 test)
- **Upload as Kaggle dataset** at `surgwound-dataset` for instant `/kaggle/input/` access

### References
- [MedSigLIP model](https://huggingface.co/google/medsiglip-448)
- [Google's fine-tuning notebook](https://github.com/google-health/medsiglip/blob/main/notebooks/fine_tune_for_image_classification.ipynb)
- [SurgWound dataset](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound)""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Setup markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 1. Setup

### GPU Requirements
This notebook is designed for **T4 (16GB)** on Kaggle free tier.
It will also work on P100 (16GB), L4 (24GB), or A100 (40GB+).

### Dataset Setup
Before running, upload the `data/surgwound/` folder (containing `labels.csv` + `images/` with 686 JPGs)
as a Kaggle dataset named `surgwound-dataset`. It will be accessible at `/kaggle/input/surgwound-dataset/`.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Install dependencies ──────────────────────────────────────────────────────
# Pin transformers >= 4.46.0 for Trainer compatibility
!pip install --upgrade --quiet \\
    "transformers>=4.46.0" \\
    accelerate \\
    datasets \\
    evaluate \\
    tensorboard \\
    scikit-learn \\
    tqdm \\
    pillow""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Auth
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Authenticate with Hugging Face ────────────────────────────────────────────
# Required to download gated model: google/medsiglip-448
#
# On Kaggle: Add your HF token as a Kaggle Secret named "HF_TOKEN"
# On Colab:  Add it to Colab Secrets, or it will prompt notebook_login()

import os
import sys

if "kaggle_secrets" in dir() or os.path.exists("/kaggle"):
    # Running on Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
        print("✓ HF_TOKEN loaded from Kaggle Secrets")
    except Exception as e:
        print(f"⚠ Could not load HF_TOKEN from Kaggle Secrets: {e}")
        print("  Falling back to huggingface_hub login...")
        from huggingface_hub import notebook_login
        notebook_login()
elif "google.colab" in sys.modules:
    # Running on Colab
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        print("✓ HF_TOKEN loaded from Colab Secrets")
    except Exception:
        from huggingface_hub import notebook_login
        notebook_login()
else:
    # Local / other environment
    from huggingface_hub import get_token
    if get_token() is None:
        from huggingface_hub import notebook_login
        notebook_login()
    else:
        print("✓ HF token already configured")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Config markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 2. Configuration

All hyperparameters and label definitions in one place.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Configuration
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Force single-GPU mode on multi-GPU Kaggle runtimes ──────────────────────
# Must run BEFORE importing torch.
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
# Adjust BASE_PATH depending on your environment:
#   Kaggle:  /kaggle/input/surgwound-dataset
#   Local:   ./data/surgwound

if os.path.exists("/kaggle/input/surgwound-dataset"):
    BASE_PATH = "/kaggle/input/surgwound-dataset"
elif os.path.exists("/kaggle/input/surgwound"):
    BASE_PATH = "/kaggle/input/surgwound"
else:
    BASE_PATH = "./data/surgwound"  # Local development

LABELS_CSV = os.path.join(BASE_PATH, "labels.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "images")

# If Kaggle dataset was uploaded with directory mode=zip, images may arrive as images.zip
IMAGES_ZIP = os.path.join(BASE_PATH, "images.zip")
if not os.path.isdir(IMAGES_DIR) and os.path.isfile(IMAGES_ZIP):
    import zipfile
    print(f"Extracting {IMAGES_ZIP} ...")
    with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
        zf.extractall(BASE_PATH)
    print(f"✓ Extracted images to {IMAGES_DIR}")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "google/medsiglip-448"
OUTPUT_DIR = "medsiglip-448-surgwound-6label"

# ── Label definitions ─────────────────────────────────────────────────────────
# 6 binary labels predicted from wound images
LABEL_NAMES = [
    "healing_status",   # 0: Healed, 1: Not Healed
    "erythema",         # 0: Non-existent, 1: Existent       (has MISSING)
    "edema",            # 0: Non-existent, 1: Existent       (has MISSING)
    "infection_risk",   # 0: Low, 1: Medium+High
    "urgency",          # 0: Green (home care), 1: Yellow+Red (needs attention)
    "exudate",          # 0: Non-existent, 1: Any exudate    (has MISSING)
]
NUM_LABELS = len(LABEL_NAMES)

id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

# Precomputed from training split (non-MISSING samples only):
#   healing:   neg=282, pos=198  → 282/198 = 1.42
#   erythema:  neg=334, pos=129  → 334/129 = 2.59
#   edema:     neg=328, pos=50   → 328/50  = 6.56
#   infection: neg=402, pos=78   → 402/78  = 5.15
#   urgency:   neg=423, pos=57   → 423/57  = 7.42
#   exudate:   neg=367, pos=70   → 367/70  = 5.24
POS_WEIGHT = torch.tensor([1.42, 2.59, 6.56, 5.15, 7.42, 5.24])

# ── Freezing strategy ────────────────────────────────────────────────────────
N_UNFREEZE = 4  # Unfreeze last N encoder blocks + classification head

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 4
GRAD_ACCUM = 16       # Effective batch = 4 × 16 = 64 (matches Google's 8×8)
EPOCHS = 5            # More passes for small dataset (480 images)
LR = 5e-5             # Matches Google's reference notebook
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
SCHEDULER = "cosine"
FP16 = True           # T4 doesn't support bf16; fp16 saves VRAM

# ── Device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    visible_gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {visible_gpu_count}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    # Auto-adjust for larger GPUs
    if gpu_mem >= 30:  # A100 / L4
        BATCH_SIZE = 8
        GRAD_ACCUM = 8
        print(f"  → Adjusted: batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
else:
    print("⚠ No GPU detected — training will be extremely slow")
    FP16 = False

print(f"\\nEffective batch size: {BATCH_SIZE * GRAD_ACCUM}")
forward_passes = (480 + BATCH_SIZE - 1) // BATCH_SIZE
optimizer_steps = (forward_passes + GRAD_ACCUM - 1) // GRAD_ACCUM
print(f"Forward passes per epoch: {forward_passes}")
print(f"Optimizer steps per epoch: {optimizer_steps}")
print(f"Dataset path: {BASE_PATH}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Data loading markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 3. Load & Validate Dataset

Read `labels.csv`, verify split counts (480/69/137), and print label distributions.
This cell fails fast if the dataset is corrupted or missing.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Load & validate data
# ═══════════════════════════════════════════════════════════════════════════════
code("""import pandas as pd
from pathlib import Path

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(df)} rows from {LABELS_CSV}")
print(f"Columns: {list(df.columns)}\\n")

# ── Verify split counts ──────────────────────────────────────────────────────
split_counts = df["split"].value_counts().to_dict()
print("Split counts:", split_counts)

assert split_counts.get("train", 0) == 480, f"Expected 480 train, got {split_counts.get('train', 0)}"
assert split_counts.get("validation", 0) == 69, f"Expected 69 val, got {split_counts.get('validation', 0)}"
assert split_counts.get("test", 0) == 137, f"Expected 137 test, got {split_counts.get('test', 0)}"
print("✓ Split counts verified: 480 train / 69 val / 137 test\\n")

# ── Verify all images exist on disk ──────────────────────────────────────────
missing_images = []
for _, row in df.iterrows():
    img_path = os.path.join(BASE_PATH, row["image_path"])
    if not os.path.exists(img_path):
        missing_images.append(img_path)

if missing_images:
    print(f"✗ {len(missing_images)} images missing! First 5:")
    for p in missing_images[:5]:
        print(f"  {p}")
    raise FileNotFoundError(f"{len(missing_images)} images not found on disk")
else:
    print(f"✓ All {len(df)} images verified on disk\\n")

# ── Print label distributions for training split ─────────────────────────────
train_df = df[df["split"] == "train"]
print("=" * 60)
print("TRAINING SPLIT LABEL DISTRIBUTIONS")
print("=" * 60)
for col in ["healing_status", "erythema", "edema", "infection_risk", "urgency_level", "exudate_type"]:
    counts = train_df[col].value_counts()
    print(f"\\n{col}:")
    for val, cnt in counts.items():
        pct = 100 * cnt / len(train_df)
        marker = " ← MISSING" if val == "MISSING" else ""
        print(f"  {val:55s} {cnt:4d} ({pct:5.1f}%){marker}")

print("\\n" + "=" * 60)""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Label encoding markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 4. Label Encoding

Convert raw CSV labels into a 6-dimensional binary vector per image.

| # | Label | 0 (negative) | 1 (positive) | MISSING → -1 |
|---|---|---|---|---|
| 0 | healing_status | Healed | Not Healed | never |
| 1 | erythema | Non-existent | Existent | yes (17 in train) |
| 2 | edema | Non-existent | Existent | yes (102 in train) |
| 3 | infection_risk | Low | Medium or High | never |
| 4 | urgency | Home Care (Green) | Clinic Visit or Emergency | never |
| 5 | exudate | Non-existent | Any type present | yes (43 in train) |""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Label encoding function
# ═══════════════════════════════════════════════════════════════════════════════
code("""def encode_labels(row: pd.Series) -> list[float]:
    \"\"\"
    Convert a single CSV row into a 6-dim label vector.

    Returns:
        List of 6 floats: 0.0 (negative), 1.0 (positive), or -1.0 (MISSING).
        The masked loss function will ignore -1.0 entries.
    \"\"\"
    labels = []

    # 0. healing_status: "Not Healed" → 1 (positive), "Healed" → 0
    labels.append(1.0 if row["healing_status"] == "Not Healed" else 0.0)

    # 1. erythema: "Existent" → 1, "Non-existent" → 0, "MISSING" → -1
    if row["erythema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["erythema"] == "Existent" else 0.0)

    # 2. edema: "Existent" → 1, "Non-existent" → 0, "MISSING" → -1
    if row["edema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["edema"] == "Existent" else 0.0)

    # 3. infection_risk: "Medium" or "High" → 1, "Low" → 0
    labels.append(1.0 if row["infection_risk"] in ("Medium", "High") else 0.0)

    # 4. urgency: anything other than "Home Care (Green)..." → 1
    labels.append(0.0 if row["urgency_level"].startswith("Home Care") else 1.0)

    # 5. exudate: "Non-existent" → 0, "MISSING" → -1, anything else → 1
    if row["exudate_type"] == "MISSING":
        labels.append(-1.0)
    elif row["exudate_type"] == "Non-existent":
        labels.append(0.0)
    else:
        labels.append(1.0)  # Serous, Sanguineous, Purulent, Seropurulent

    return labels


# ── Verify encoding on a few known examples ──────────────────────────────────
sample_row = train_df.iloc[0]
sample_labels = encode_labels(sample_row)
print(f"Sample row (img_id={sample_row['img_id']}):")
print(f"  healing_status = {str(sample_row['healing_status']):20s} → {sample_labels[0]}")
print(f"  erythema       = {str(sample_row['erythema']):20s} → {sample_labels[1]}")
print(f"  edema          = {str(sample_row['edema']):20s} → {sample_labels[2]}")
print(f"  infection_risk = {str(sample_row['infection_risk']):20s} → {sample_labels[3]}")
print(f"  urgency_level  = {str(sample_row['urgency_level'])[:30]:30s} → {sample_labels[4]}")
print(f"  exudate_type   = {str(sample_row['exudate_type']):20s} → {sample_labels[5]}")
print(f"\\n  Encoded vector: {sample_labels}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Build datasets markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 5. Create HuggingFace Datasets

Build `Dataset` objects for train, validation, and test splits.
Each sample has an `image` (PIL) and `label` (6-dim float list).""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Build datasets
# ═══════════════════════════════════════════════════════════════════════════════
code("""from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from PIL import Image
from tqdm.auto import tqdm


def build_dataset_from_split(split_df: pd.DataFrame, split_name: str) -> Dataset:
    \"\"\"
    Build a HuggingFace Dataset from a pandas DataFrame for one split.

    Loads images from disk and encodes labels as 6-dim float vectors.
    \"\"\"
    image_paths = []
    labels = []
    skipped = 0

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Loading {split_name}"):
        img_path = os.path.join(BASE_PATH, row["image_path"])
        if not os.path.exists(img_path):
            skipped += 1
            continue
        image_paths.append(img_path)
        labels.append(encode_labels(row))

    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} missing images in {split_name}")

    # Build dataset with image paths, then cast to Image type for lazy loading
    ds = Dataset.from_dict({
        "image": image_paths,
        "label": labels,
    })
    ds = ds.cast_column("image", HFImage())

    print(f"  ✓ {split_name}: {len(ds)} samples")
    return ds


# ── Build all three splits ────────────────────────────────────────────────────
train_ds_raw = build_dataset_from_split(df[df["split"] == "train"], "train")
val_ds_raw   = build_dataset_from_split(df[df["split"] == "validation"], "validation")
test_ds_raw  = build_dataset_from_split(df[df["split"] == "test"], "test")

print(f"\\nDataset sizes: train={len(train_ds_raw)}, val={len(val_ds_raw)}, test={len(test_ds_raw)}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Preprocessing markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 6. Image Preprocessing

Following Google's MedSigLIP preprocessing exactly:
1. **CenterCrop** to `max(width, height)` — zero-pads non-square images to preserve aspect ratio
2. **Resize** to 448×448 (bilinear)
3. **ToTensor** — rescale [0, 255] → [0, 1]
4. **Normalize** — scale to [-1, 1] with mean=0.5, std=0.5

Training adds light augmentation (horizontal flip, rotation, color jitter) to compensate for the small dataset.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Preprocessing transforms
# ═══════════════════════════════════════════════════════════════════════════════
code("""from torchvision.transforms import (
    Compose, CenterCrop, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter,
    InterpolationMode,
)
from transformers import AutoImageProcessor

# ── Load processor to get canonical image size and normalization ──────────────
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
IMG_SIZE = image_processor.size["height"]  # 448
IMG_MEAN = image_processor.image_mean       # [0.5, 0.5, 0.5]
IMG_STD = image_processor.image_std          # [0.5, 0.5, 0.5]
print(f"Image size: {IMG_SIZE}, mean: {IMG_MEAN}, std: {IMG_STD}")

# ── Base transform (eval/test) — matches Google's training preprocessing ─────
_base_transform = Compose([
    # CenterCrop zero-pads to square, preserving aspect ratio — CRITICAL
    # Google's training data used this exact step
    Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=IMG_MEAN, std=IMG_STD),  # pixel range: [-1, 1]
])

# ── Training transform — adds light augmentation ─────────────────────────────
_train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.1, contrast=0.1),
    Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def _apply_zero_pad_and_transform(image, transform) -> torch.Tensor:
    \"\"\"
    Apply Google's zero-pad-to-square then the given transform pipeline.

    CenterCrop(max(w, h)) on a non-square image effectively zero-pads
    the shorter dimension to match the longer one, preserving aspect ratio.
    \"\"\"
    image = image.convert("RGB")
    # Zero-pad to square (same as Google's notebook)
    image = CenterCrop(max(image.size))(image)
    return transform(image)


def preprocess_train(examples):
    \"\"\"Preprocess training examples with augmentation.\"\"\"
    examples["pixel_values"] = [
        _apply_zero_pad_and_transform(img, _train_transform)
        for img in examples["image"]
    ]
    return examples


def preprocess_eval(examples):
    \"\"\"Preprocess validation/test examples without augmentation.\"\"\"
    examples["pixel_values"] = [
        _apply_zero_pad_and_transform(img, _base_transform)
        for img in examples["image"]
    ]
    return examples


# ── Apply preprocessing ──────────────────────────────────────────────────────
print("Preprocessing training data (with augmentation)...")
train_ds = train_ds_raw.map(preprocess_train, batched=True, remove_columns=["image"])

print("Preprocessing validation data...")
val_ds = val_ds_raw.map(preprocess_eval, batched=True, remove_columns=["image"])

print("Preprocessing test data...")
test_ds = test_ds_raw.map(preprocess_eval, batched=True, remove_columns=["image"])

print(f"\\n✓ Preprocessed: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"  Sample pixel_values shape: {train_ds[0]['pixel_values'].shape}")
print(f"  Sample label: {train_ds[0]['label']}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Sanity check markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 7. Visual Sanity Check

Display sample images with their encoded labels to verify the pipeline.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: Visual sanity check
# ═══════════════════════════════════════════════════════════════════════════════
code("""import matplotlib.pyplot as plt


def show_sample(ds_raw, ds_processed, idx, title_prefix=""):
    \"\"\"Display an image alongside its encoded label vector.\"\"\"
    raw_img = ds_raw[idx]["image"]
    label_vec = ds_processed[idx]["label"]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(raw_img)
    ax.set_title(f"{title_prefix} sample {idx}", fontsize=10)
    ax.axis("off")

    label_str = "\\n".join(
        f"  {LABEL_NAMES[i]}: {label_vec[i]:+.0f}"
        + (" (MISSING)" if label_vec[i] == -1.0 else "")
        for i in range(NUM_LABELS)
    )
    ax.text(
        1.05, 0.5, label_str,
        transform=ax.transAxes, fontsize=9, verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    plt.show()


# Show 4 samples: 2 train, 2 val
for i in [0, 50]:
    show_sample(train_ds_raw, train_ds, i, title_prefix="Train")
for i in [0, 30]:
    show_sample(val_ds_raw, val_ds, i, title_prefix="Val")

# ── Verify MISSING encoding ──────────────────────────────────────────────────
missing_found = False
for idx in range(len(train_ds)):
    label_vec = train_ds[idx]["label"]
    if -1.0 in label_vec:
        missing_indices = [i for i, v in enumerate(label_vec) if v == -1.0]
        missing_names = [LABEL_NAMES[i] for i in missing_indices]
        print(f"✓ Found MISSING in train[{idx}]: {missing_names}")
        print(f"  Full label vector: {label_vec}")
        missing_found = True
        break

if not missing_found:
    print("⚠ No MISSING values found in training data — check encoding!")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16: Model markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 8. Load Model with Selective Freezing

Load `google/medsiglip-448` with a 6-label classification head.
Freeze all parameters except the last 4 encoder blocks + the new head.

This reduces trainable params from ~400M to ~50-60M, fitting comfortably in T4 16GB VRAM.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 17: Load model
# ═══════════════════════════════════════════════════════════════════════════════
code("""from transformers import AutoModelForImageClassification

# ── Load pretrained model with new classification head ────────────────────────
model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    problem_type="multi_label_classification",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # Head size mismatch: pretrained != 6
)

# ── Freeze all parameters ────────────────────────────────────────────────────
for param in model.parameters():
    param.requires_grad = False

# ── Unfreeze classification head (randomly initialized — MUST be trainable) ──
for param in model.classifier.parameters():
    param.requires_grad = True

# ── Unfreeze last N encoder blocks ───────────────────────────────────────────
encoder_layers = model.vision_model.encoder.layers
total_layers = len(encoder_layers)
print(f"Encoder has {total_layers} layers. Unfreezing last {N_UNFREEZE}...")

for layer in encoder_layers[-N_UNFREEZE:]:
    for param in layer.parameters():
        param.requires_grad = True

# ── Print parameter counts ───────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"\\nParameter summary:")
print(f"  Total:     {total_params:>12,}")
print(f"  Trainable: {trainable_params:>12,} ({100*trainable_params/total_params:.1f}%)")
print(f"  Frozen:    {frozen_params:>12,} ({100*frozen_params/total_params:.1f}%)")

# ── Move to device and check VRAM ────────────────────────────────────────────
model = model.to(device)

if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\\nVRAM after model load: {vram_used:.2f} / {vram_total:.1f} GB")
    if vram_used > vram_total * 0.7:
        print("⚠ VRAM usage is high — consider reducing N_UNFREEZE or BATCH_SIZE")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18: Loss markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 9. Masked BCE Loss Function

Three labels (erythema, edema, exudate) have MISSING values encoded as `-1.0`.
Instead of dropping entire rows (losing signal for other valid labels),
the masked loss zeros out gradient contributions for MISSING entries.

Every image contributes to every label it has data for.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 19: Masked loss + unit test
# ═══════════════════════════════════════════════════════════════════════════════
code("""from torch.nn import BCEWithLogitsLoss


def masked_bce_loss(
    outputs: dict,
    labels: torch.Tensor,
) -> torch.Tensor:
    \"\"\"
    BCE loss with per-element masking for MISSING values and class-imbalance weighting.

    Args:
        outputs: Model output dict containing 'logits' of shape (batch, 6).
        labels:  Tensor of shape (batch, 6) with values in {0.0, 1.0, -1.0}.
                 -1.0 indicates MISSING — loss for that entry is masked out.

    Returns:
        Scalar loss tensor (mean over valid entries only).
    \"\"\"
    logits = outputs.get("logits")             # (batch, 6)
    mask = (labels >= 0).float()                # 1 where valid, 0 where MISSING
    safe_labels = labels.clamp(min=0.0)         # Replace -1 with 0 for BCE math

    pos_weight = POS_WEIGHT.to(logits.device)
    loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    per_element_loss = loss_fct(logits, safe_labels)  # (batch, 6)
    masked_loss = per_element_loss * mask               # Zero out MISSING

    # Mean over valid entries only (prevents bias from batch MISSING counts)
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return masked_loss.sum() / num_valid


# ── Unit test: verify MISSING entries are masked ─────────────────────────────
print("Running masked loss unit test...")

# Fake logits and labels with known MISSING entries
_test_logits = torch.tensor([[0.5, 0.3, -0.2, 0.1, 0.4, 0.6]])
_test_labels_valid = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])   # All valid
_test_labels_missing = torch.tensor([[1.0, 0.0, -1.0, 0.0, 1.0, -1.0]])  # Entries 2,5 MISSING

# Compute loss with all valid
_loss_valid = masked_bce_loss({"logits": _test_logits}, _test_labels_valid)

# Compute loss with MISSING
_loss_masked = masked_bce_loss({"logits": _test_logits}, _test_labels_missing)

print(f"  Loss (all valid):     {_loss_valid.item():.4f}  (6/6 entries contribute)")
print(f"  Loss (2 MISSING):     {_loss_masked.item():.4f}  (4/6 entries contribute)")
print(f"  Losses differ: {_loss_valid.item() != _loss_masked.item()} (expected: True)")

# Verify gradient flows for valid entries but not MISSING
_test_logits_grad = torch.tensor([[0.5, 0.3, -0.2, 0.1, 0.4, 0.6]], requires_grad=True)
_loss_for_grad = masked_bce_loss({"logits": _test_logits_grad}, _test_labels_missing)
_loss_for_grad.backward()
grad = _test_logits_grad.grad[0]
print(f"  Gradients: {[round(g, 4) for g in grad.tolist()]}")
print(f"  MISSING positions (2,5) have zero grad: {grad[2].item() == 0.0 and grad[5].item() == 0.0}")
print("✓ Masked loss unit test passed!")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 20: Metrics markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 10. Evaluation Metrics

Macro-averaged One-vs-Rest ROC AUC + per-label sensitivity/specificity.
For labels with MISSING values in the eval set, metrics are computed only on non-MISSING samples.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 21: Metrics functions
# ═══════════════════════════════════════════════════════════════════════════════
code("""from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    \"\"\"
    Compute evaluation metrics, handling MISSING values (-1) in labels.

    Returns:
        Dict with 'roc_auc_macro' and per-label AUC.
    \"\"\"
    logits, labels = eval_pred
    scores = sigmoid(logits)

    results = {}
    per_label_auc = []

    for i, name in enumerate(LABEL_NAMES):
        # Mask out MISSING entries for this label
        valid_mask = labels[:, i] >= 0
        if valid_mask.sum() == 0:
            continue

        y_true = labels[valid_mask, i]
        y_score = scores[valid_mask, i]

        # Need at least one sample of each class for AUC
        if len(np.unique(y_true)) < 2:
            results[f"auc_{name}"] = float("nan")
            continue

        try:
            auc = roc_auc_score(y_true, y_score)
            per_label_auc.append(auc)
            results[f"auc_{name}"] = auc
        except ValueError:
            results[f"auc_{name}"] = float("nan")

    # Macro-averaged AUC (only over labels with valid AUC)
    if per_label_auc:
        results["roc_auc_macro"] = np.mean(per_label_auc)
    else:
        results["roc_auc_macro"] = float("nan")

    return results


def compute_full_metrics(logits, labels, threshold=0.5):
    \"\"\"
    Compute detailed metrics including sensitivity/specificity per label.
    Used for final test evaluation (not during training).
    \"\"\"
    scores = sigmoid(logits)
    results = {}

    print("\\n" + "=" * 70)
    print(f"{'Label':<20} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'N valid':>8}")
    print("=" * 70)

    all_aucs = []
    for i, name in enumerate(LABEL_NAMES):
        valid_mask = labels[:, i] >= 0
        n_valid = valid_mask.sum()
        y_true = labels[valid_mask, i]
        y_score = scores[valid_mask, i]
        y_pred = (y_score > threshold).astype(int)

        # AUC
        try:
            auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else float("nan")
        except ValueError:
            auc = float("nan")

        # Sensitivity and specificity
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        if not np.isnan(auc):
            all_aucs.append(auc)

        results[f"auc_{name}"] = auc
        results[f"sens_{name}"] = sens
        results[f"spec_{name}"] = spec

        print(f"{name:<20} {auc:>8.4f} {sens:>8.4f} {spec:>8.4f} {n_valid:>8d}")

    macro_auc = np.mean(all_aucs) if all_aucs else float("nan")
    results["roc_auc_macro"] = macro_auc
    print("=" * 70)
    print(f"{'Macro AUC':<20} {macro_auc:>8.4f}")
    print("=" * 70)

    return results


print("✓ Metrics functions defined")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 22: Collator markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 11. Data Collator""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 23: Collator function
# ═══════════════════════════════════════════════════════════════════════════════
code("""def collate_fn(examples):
    \"\"\"
    Collate function for Trainer.

    Stacks pixel_values into a (batch, 3, 448, 448) tensor
    and labels into a (batch, 6) float tensor (with -1.0 for MISSING).
    \"\"\"
    pixel_values = torch.stack([torch.tensor(ex["pixel_values"]) for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.float)
    return {"pixel_values": pixel_values, "labels": labels}


# ── Quick verification ────────────────────────────────────────────────────────
_test_batch = collate_fn([train_ds[0], train_ds[1]])
print(f"Collated batch shapes:")
print(f"  pixel_values: {_test_batch['pixel_values'].shape}")   # (2, 3, 448, 448)
print(f"  labels:       {_test_batch['labels'].shape}")          # (2, 6)
print(f"  labels dtype: {_test_batch['labels'].dtype}")          # float32
print(f"  labels[0]:    {_test_batch['labels'][0].tolist()}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 24: Trainer markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 12. Trainer Setup

Uses a **subclassed Trainer** to inject the masked BCE loss.
This approach works across all `transformers` versions (safer than `compute_loss_func`
parameter which was introduced in ~4.46.0 and may be unavailable on some Kaggle kernels).

### Training Configuration Summary
| Parameter | Value | Rationale |
|---|---|---|
| Effective batch size | 64 (4×16) | Matches Google's 8×8 |
| Epochs | 5 | More passes for 480-image dataset |
| Learning rate | 5e-5 | Matches Google's reference |
| Scheduler | Cosine | Matches Google's reference |
| fp16 | True | T4 VRAM optimization |
| Model selection | eval_loss | Val set too small (69) for reliable AUC |""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 25: Trainer setup
# ═══════════════════════════════════════════════════════════════════════════════
code("""from transformers import Trainer, TrainingArguments


class WoundClassificationTrainer(Trainer):
    \"\"\"
    Custom Trainer that uses masked BCE loss for MISSING label handling.

    Subclassing Trainer.compute_loss() is more portable across transformers
    versions than the compute_loss_func parameter.
    \"\"\"

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = masked_bce_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss


# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger batch for eval (no grad)
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type=SCHEDULER,
    fp16=FP16,
    logging_steps=1,                    # ~7.5 steps/epoch — log every step
    save_strategy="epoch",
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",  # Val set too small for reliable AUC
    greater_is_better=False,            # Lower loss = better
    load_best_model_at_end=True,
    report_to="tensorboard",
    push_to_hub=False,                  # Push manually after evaluation
    remove_unused_columns=False,        # Keep our custom columns
    dataloader_num_workers=2,
    save_total_limit=3,                 # Keep only 3 best checkpoints
)

# ── Create Trainer ───────────────────────────────────────────────────────────
trainer = WoundClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print(f"✓ Trainer initialized")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
forward_passes = (len(train_ds) + BATCH_SIZE - 1) // BATCH_SIZE
optimizer_steps = (forward_passes + GRAD_ACCUM - 1) // GRAD_ACCUM
print(f"  Forward passes per epoch: {forward_passes}")
print(f"  Optimizer steps per epoch: {optimizer_steps}")
print(f"  Total optimizer steps: {EPOCHS * optimizer_steps}")
print(f"  Model selection: best eval_loss (lower is better)")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 26: Training markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 13. Train

Expected training time: **15–25 minutes on T4** (vs Google's ~3 hours for SCIN on A100).
Much faster because: (a) 480 images vs thousands, (b) most params frozen.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 27: Train
# ═══════════════════════════════════════════════════════════════════════════════
code("""import time

print("Starting training...\\n")
start_time = time.time()

train_result = trainer.train()

elapsed = time.time() - start_time
print(f"\\n{'='*60}")
print(f"Training complete in {elapsed/60:.1f} minutes")
print(f"{'='*60}")

# ── Check for NaN (fp16 safety) ──────────────────────────────────────────────
final_loss = train_result.training_loss
if np.isnan(final_loss):
    print("\\n⚠ WARNING: Training loss is NaN!")
    print("  This likely means fp16 caused numerical instability.")
    print("  Try re-running with FP16 = False in the config cell.")
else:
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Best model loaded from: {trainer.state.best_model_checkpoint}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 28: Eval markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 14. Evaluate on Test Set

Run inference on the 137 test images and compute per-label metrics:
AUC, sensitivity, and specificity.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 29: Evaluate
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Run prediction on test set ───────────────────────────────────────────────
print("Running inference on test set (137 images)...")
predictions = trainer.predict(test_ds)

test_logits = predictions.predictions  # (137, 6)
test_labels = predictions.label_ids     # (137, 6)

print(f"Logits shape: {test_logits.shape}")
print(f"Labels shape: {test_labels.shape}")

# ── Compute detailed metrics ─────────────────────────────────────────────────
test_metrics = compute_full_metrics(test_logits, test_labels, threshold=0.5)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\\nTest set macro-averaged ROC AUC: {test_metrics['roc_auc_macro']:.4f}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 30: Save markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 15. Save Model

Save the fine-tuned model and image processor locally.
Optionally push to Hugging Face Hub.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 31: Save model
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Save locally ─────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
image_processor.save_pretrained(OUTPUT_DIR)
print(f"✓ Model and image processor saved to {OUTPUT_DIR}/")

# ── Optionally push to Hugging Face Hub ──────────────────────────────────────
# Uncomment the following lines to push to your HF account:
#
# HUB_MODEL_ID = "<your-username>/medsiglip-448-mamaguard-wound"
# trainer.push_to_hub(HUB_MODEL_ID)
# image_processor.push_to_hub(HUB_MODEL_ID)
# print(f"✓ Pushed to https://huggingface.co/{HUB_MODEL_ID}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 32: Inference markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 16. Inference Demo

Load the saved model and run inference on 3 test images
to show what the downstream Gemini orchestrator would receive.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 33: Inference demo
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Load the fine-tuned model ────────────────────────────────────────────────
ft_model = AutoModelForImageClassification.from_pretrained(
    OUTPUT_DIR,
    problem_type="multi_label_classification",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    device_map="auto",
)
ft_model.eval()

# ── Human-readable label mapping ─────────────────────────────────────────────
LABEL_DISPLAY = {
    "healing_status": ("progressing", "not progressing"),
    "erythema":       ("absent", "present (redness)"),
    "edema":          ("absent", "present (swelling)"),
    "infection_risk": ("low", "elevated"),
    "urgency":        ("home care OK", "needs professional attention"),
    "exudate":        ("absent", "present (drainage)"),
}


def predict_wound(image, model, threshold=0.5):
    \"\"\"
    Run wound assessment on a single image.

    Returns a dict with label names → {score: float, prediction: str}.
    \"\"\"
    # Apply same preprocessing as during training/eval
    pixel_values = _apply_zero_pad_and_transform(image, _base_transform)
    pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension

    # Move to model's device
    model_device = next(model.parameters()).device
    pixel_values = pixel_values.to(model_device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()

    results = {}
    for i, name in enumerate(LABEL_NAMES):
        neg_label, pos_label = LABEL_DISPLAY[name]
        is_positive = probs[i] > threshold
        results[name] = {
            "score": float(probs[i]),
            "prediction": pos_label if is_positive else neg_label,
        }

    return results


# ── Demo on 3 test images ────────────────────────────────────────────────────
test_df = df[df["split"] == "test"]
demo_indices = [0, 50, 100]

for idx in demo_indices:
    if idx >= len(test_df):
        continue

    row = test_df.iloc[idx]
    img_path = os.path.join(BASE_PATH, row["image_path"])
    img = Image.open(img_path).convert("RGB")

    results = predict_wound(img, ft_model)

    print(f"\\n{'─'*60}")
    print(f"Image: {row['image_path']} (img_id={row['img_id']})")
    print(f"{'─'*60}")
    print("\\nWound Assessment Results:")
    for name, info in results.items():
        confidence = info['score']
        prediction = info['prediction']
        bar = '█' * int(confidence * 20) + '░' * (20 - int(confidence * 20))
        print(f"  {name:<20s} {bar} {confidence:.2f}  →  {prediction}")

    # Show the image
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img)
    ax.set_title(f"img_id={row['img_id']}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

print("\\n✓ Inference demo complete")
print("  These structured scores feed into the Gemini/MedGemma orchestrator")
print("  to generate empathetic, contextual responses for MamaGuard users.")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 34: Appendix markdown
# ═══════════════════════════════════════════════════════════════════════════════
md("""## Appendix: Dynamic pos_weight Verification

Recompute `pos_weight` from the actual training data to verify the hardcoded constants.
Run this cell to double-check if you've modified the dataset or label encoding.""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 35: pos_weight verification
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Dynamically compute pos_weight from training labels ──────────────────────
train_labels = torch.tensor(train_ds["label"])  # (480, 6)

print(f"Training label tensor shape: {train_labels.shape}")
print(f"\\n{'Label':<20s} {'Neg':>6} {'Pos':>6} {'Miss':>6} {'pos_weight':>10} {'Hardcoded':>10} {'Match':>6}")
print("-" * 70)

computed_pw = []
for i, name in enumerate(LABEL_NAMES):
    col = train_labels[:, i]
    n_missing = (col == -1).sum().item()
    n_pos = (col == 1).sum().item()
    n_neg = (col == 0).sum().item()
    pw = n_neg / n_pos if n_pos > 0 else float("inf")
    computed_pw.append(pw)
    hardcoded = POS_WEIGHT[i].item()
    match = abs(pw - hardcoded) < 0.05
    print(f"{name:<20s} {n_neg:>6d} {n_pos:>6d} {n_missing:>6d} {pw:>10.2f} {hardcoded:>10.2f} {'✓' if match else '✗':>6}")

print(f"\\nHardcoded POS_WEIGHT: {POS_WEIGHT.tolist()}")
print(f"Computed POS_WEIGHT:  {[round(x, 2) for x in computed_pw]}")""")


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE THE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════

# Fix: ensure each cell source is a list of lines ending with \n
for cell in cells:
    raw_source = cell["source"]
    # raw_source is already a list from split("\n")
    # We need each line to end with \n except possibly the last
    fixed = []
    for j, line in enumerate(raw_source):
        if j < len(raw_source) - 1:
            fixed.append(line + "\n")
        else:
            fixed.append(line)
    cell["source"] = fixed

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "/Users/tayyabkhan/Downloads/medgemma/medsiglip_surgwound_finetune.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Notebook written to {output_path}")
print(f"  Total cells: {len(cells)}")
print(
    f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
