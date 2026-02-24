#!/usr/bin/env python3
"""Patch cells 12, 14, and 34 in the notebook JSON directly."""

import json

NB = "/Users/tayyabkhan/Downloads/medgemma/medsiglip_surgwound_finetune.ipynb"

with open(NB) as f:
    nb = json.load(f)


def src_lines(code: str) -> list[str]:
    """Convert a code string into notebook source line format."""
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)  # last line has no trailing newline
    # Remove trailing empty string if present
    if result and result[-1] == "":
        result.pop()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Build Datasets (fix PIL import + explicit Features)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_12 = '''\
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
import PIL.Image  # Ensure PIL.Image is in sys.modules for datasets internals
from tqdm.auto import tqdm


def build_dataset_from_split(split_df: pd.DataFrame, split_name: str) -> Dataset:
    """
    Build a HuggingFace Dataset from a pandas DataFrame for one split.

    Loads images from disk and encodes labels as 6-dim float vectors.
    Uses explicit Features schema to avoid datasets library PIL type-inference bugs.
    """
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

    # Explicit schema avoids PIL.Image.Image isinstance check inside datasets
    features = Features({
        "image": HFImage(),
        "label": Sequence(Value("float32"), length=6),
    })

    ds = Dataset.from_dict(
        {"image": image_paths, "label": labels},
        features=features,
    )

    print(f"  ✓ {split_name}: {len(ds)} samples")
    return ds


# ── Build all three splits ────────────────────────────────────────────────────
train_ds_raw = build_dataset_from_split(df[df["split"] == "train"], "train")
val_ds_raw   = build_dataset_from_split(df[df["split"] == "validation"], "validation")
test_ds_raw  = build_dataset_from_split(df[df["split"] == "test"], "test")

print(f"\\nDataset sizes: train={len(train_ds_raw)}, val={len(val_ds_raw)}, test={len(test_ds_raw)}")'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Pure PIL Transforms (no torchvision)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_14 = '''\
import random
import numpy as np
import torch
from PIL import Image as PILImage, ImageEnhance
from transformers import AutoImageProcessor

# ── Load processor to get canonical image size and normalization ──────────────
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
IMG_SIZE = image_processor.size["height"]   # 448
IMG_MEAN = image_processor.image_mean       # [0.5, 0.5, 0.5]
IMG_STD  = image_processor.image_std        # [0.5, 0.5, 0.5]
print(f"✓ Image size: {IMG_SIZE}, mean: {IMG_MEAN}, std: {IMG_STD}")


# ── Pure PIL/numpy transform functions (no torchvision dependency) ────────────

def _pil_to_tensor(img: PILImage.Image) -> torch.Tensor:
    """Convert PIL RGB image → float tensor (C, H, W) in [-1, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0                        # [0, 1]
    arr = (arr - np.array(IMG_MEAN, dtype=np.float32)) / \\
          np.array(IMG_STD, dtype=np.float32)                             # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)                        # (C, H, W)


def _zero_pad_to_square(img: PILImage.Image) -> PILImage.Image:
    """
    Zero-pad shorter dimension to match longer — replicates Google's
    CenterCrop(max(image.size)) trick exactly.
    """
    w, h = img.size
    max_dim = max(w, h)
    if w == h:
        return img
    padded = PILImage.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    return padded


def _augment(img: PILImage.Image) -> PILImage.Image:
    """Light training augmentation: flip, rotation, brightness/contrast jitter."""
    if random.random() < 0.5:
        img = img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=PILImage.Resampling.BILINEAR,
                     fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    return img


def _process_image(img: PILImage.Image, augment: bool) -> torch.Tensor:
    """Full pipeline: RGB → pad-to-square → [augment] → resize → normalise."""
    img = img.convert("RGB")
    img = _zero_pad_to_square(img)
    if augment:
        img = _augment(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.Resampling.BILINEAR)
    return _pil_to_tensor(img)


def preprocess_train(examples: dict) -> dict:
    """Preprocess training examples with augmentation."""
    examples["pixel_values"] = [
        _process_image(img, augment=True) for img in examples["image"]
    ]
    return examples


def preprocess_eval(examples: dict) -> dict:
    """Preprocess validation/test examples without augmentation."""
    examples["pixel_values"] = [
        _process_image(img, augment=False) for img in examples["image"]
    ]
    return examples


# ── Apply preprocessing ───────────────────────────────────────────────────────
print("Preprocessing training data (with augmentation)...")
train_ds = train_ds_raw.map(preprocess_train, batched=True, remove_columns=["image"])

print("Preprocessing validation data...")
val_ds = val_ds_raw.map(preprocess_eval, batched=True, remove_columns=["image"])

print("Preprocessing test data...")
test_ds = test_ds_raw.map(preprocess_eval, batched=True, remove_columns=["image"])

# ── Sanity check ──────────────────────────────────────────────────────────────
sample_pv = torch.tensor(train_ds[0]["pixel_values"])
assert sample_pv.shape == torch.Size([3, IMG_SIZE, IMG_SIZE]), \\
    f"Unexpected shape: {sample_pv.shape}"
assert sample_pv.min() >= -1.5 and sample_pv.max() <= 1.5, \\
    f"Pixel range unexpected: [{sample_pv.min():.2f}, {sample_pv.max():.2f}]"

print(f"\\n✓ Preprocessed: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"  pixel_values shape : {list(sample_pv.shape)}")
print(f"  pixel value range  : [{sample_pv.min():.3f}, {sample_pv.max():.3f}]")
print(f"  label              : {train_ds[0][\'label\']}")'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 34: Inference Demo (fix _process_image, PILImage.open, BASE_PATH)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_34 = '''\
# ── Load the fine-tuned model ────────────────────────────────────────────────
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
    """
    Run wound assessment on a single image.

    Returns a dict with label names → {score: float, prediction: str}.
    """
    # Apply same preprocessing as eval (pure PIL pipeline, no augmentation)
    pixel_values = _process_image(image, augment=False)
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
    img = PILImage.open(img_path).convert("RGB")

    results = predict_wound(img, ft_model)

    print(f"\\n{chr(9472)*60}")
    print(f"Image: {row[\'image_path\']} (img_id={row[\'img_id\']})")
    print(f"{chr(9472)*60}")
    print("\\nWound Assessment Results:")
    for name, info in results.items():
        confidence = info[\'score\']
        prediction = info[\'prediction\']
        bar = chr(9608) * int(confidence * 20) + chr(9617) * (20 - int(confidence * 20))
        print(f"  {name:<20s} {bar} {confidence:.2f}  →  {prediction}")

    # Show the image
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img)
    ax.set_title(f"img_id={row[\'img_id\']}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

print("\\n✓ Inference demo complete")
print("  These structured scores feed into the Gemini/MedGemma orchestrator")
print("  to generate empathetic, contextual responses for MamaGuard users.")'''

# Also fix cell 13 (markdown) for the preprocessing section
CELL_13_MD = '''\
## 6. Image Preprocessing

Following Google's MedSigLIP preprocessing exactly:
1. **Zero-pad to square** — pads shorter dimension with black pixels to preserve aspect ratio
2. **Resize** to 448×448 (bilinear)
3. **Normalize** — scale to [-1, 1] with mean=0.5, std=0.5

Training adds light augmentation (horizontal flip, rotation, color jitter) to compensate for the small dataset.

> **Note**: Uses pure PIL + numpy instead of torchvision to avoid Kaggle's torch/torchvision version conflict.'''

# ── Apply patches ─────────────────────────────────────────────────────────────
nb["cells"][11]["source"] = src_lines(CELL_12)   # Cell 12
nb["cells"][12]["source"] = src_lines(CELL_13_MD)  # Cell 13 (markdown)
nb["cells"][13]["source"] = src_lines(CELL_14)   # Cell 14
nb["cells"][33]["source"] = src_lines(CELL_34)   # Cell 34

with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Patched cells 12, 13, 14, and 34")
print(f"  Notebook saved to {NB}")
