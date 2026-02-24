# MedSigLIP Fine-Tuning Plan v2 — MamaGuard Surgical Wound Assessment

## Context & Purpose

MamaGuard is a maternal health assistant prototype. The MedSigLIP model serves as the **structured clinical signal provider** in a two-stage pipeline:

```
Woman takes wound photo → MedSigLIP (this model) → structured scores
    → Gemini/MedGemma orchestrator → empathetic, contextual response
```

MedSigLIP's job is to answer: "What's clinically happening in this image?" Gemini's job is to answer: "How do I communicate this to a postpartum woman in a caring, actionable way?"

This document is the implementation plan for the MedSigLIP fine-tuning step.

---

## Dataset: SurgWound

- **Source:** `xuxuxuxuxu/SurgWound` (HuggingFace)
- **Total images:** 686 (480 train / 69 val / 137 test)
- **Image type:** Consumer smartphone photos of post-surgical wounds — taken at home, natural lighting, varying angles and quality. Many contain Chinese text overlays suggesting crowdsourced origin (likely patient-shared recovery photos). **This matches our deployment domain perfectly** — a postpartum woman photographing her C-section wound at home will produce images in this exact style.
- **Wound locations:** Abdomen (112), Other (178), MISSING (162), Ankle (72), Facial (66), Manus/Hand (35), Patella/Knee (35), Cervical (26)

### Domain Relevance Assessment

**Verified via visual inspection of 5 representative images:**

- Abdomen wounds are primarily laparoscopic port sites and abdominal incisions — not C-section horizontal incisions specifically, but the tissue-level pathology (erythema, edema, exudate, infection signs) is identical
- Images are consumer-grade smartphone photos with natural lighting, text overlays, bandages partially removed — exactly what MamaGuard users would produce
- No sterile clinical photography — the domain gap concern is resolved
- **Known limitation:** Few horizontal C-section scars in dataset. Document this honestly. The model learns wound pathology signs, not incision-shape recognition.

### Training Strategy: Use ALL 480 images, not abdomen-only

**Rationale:** Erythema, edema, and exudate look the same regardless of body site. Training on mixed wound locations with only 78 abdomen samples would be data-starvation. All 480 images provide richer signal about what inflammation and infection look like across contexts. The body-site diversity makes the model more robust, not less.

---

## Label Architecture: 6 Binary Prediction Heads

### What Changed From v1

| v1 (4 labels) | v2 (6 labels) | Why |
|---|---|---|
| healing_status, erythema, edema, infection_risk | + urgency_level, exudate_type | We discovered 2 fields with strong independent clinical signal that we were ignoring |
| infection_risk as sole triage signal | urgency_level as PRIMARY triage signal | Zero MISSING values, maps directly to user action ("go to ER" vs "you're fine"), and is the headline output for the user |
| No exudate tracking | exudate_type (binary: none vs any) | Purulent/seropurulent exudate is a direct infection marker. Adds signal independent of erythema. |

### Label Definitions

| # | Label | Encoding | Positive Class | MISSING | Train Distribution | pos_weight |
|---|---|---|---|---|---|---|
| 0 | `healing_status` | Healed=0, Not Healed=1 | Not Healed | 0 | 282 neg / 198 pos (41.2%) | 1.42 |
| 1 | `erythema` | Non-existent=0, Existent=1 | Existent | 26 | 334 neg / 129 pos (27.9%) | 2.59 |
| 2 | `edema` | Non-existent=0, Existent=1 | Existent | 132 | 328 neg / 50 pos (13.2%) | 6.56 |
| 3 | `infection_risk` | Low=0, Medium+High=1 | Elevated | 0 | 402 neg / 78 pos (16.3%) | 5.15 |
| 4 | `urgency` | Green=0, Yellow+Red=1 | Needs Attention | 0 | 423 neg / 57 pos (11.9%) | 7.42 |
| 5 | `exudate` | Non-existent=0, Any exudate=1 | Has Exudate | 60 | 367 neg / 70 pos (16.0%) | 5.24 |

**MISSING values:** Encoded as -1 in the label tensor. Masked loss zeros out gradient for MISSING entries. Three labels have MISSING: erythema (26), edema (132), exudate (60). Three labels have zero MISSING: healing_status, infection_risk, urgency.

### Why urgency is binary, not 3-class

Red (Emergency) has only 13 training samples. A 3-class classifier would produce unreliable Red predictions. Binary (Green=home-care vs Yellow+Red=needs-attention) is robust and still captures the critical clinical decision: "Does this woman need to contact her provider?"

### Redundancy analysis

- `urgency` vs `infection_risk`: 96.1% agreement. But they differ in 27 cases where Medium infection → still Green urgency. These represent "some risk factors present, but manageable at home" — the model learning this distinction is clinically valuable.
- `exudate` vs `infection_risk`: Not redundant. 33 Low-infection wounds have exudate (normal serous drainage during healing), and 32 Medium-infection wounds have no visible exudate (erythema-driven risk). Independent signal confirmed.

### How These Map to User Communication

The 6 labels feed into the Gemini orchestrator prompt as structured data:

```
Wound Assessment Results:
- Healing: progressing / not progressing
- Erythema (redness): present / absent
- Edema (swelling): present / absent
- Exudate (drainage): present / absent
- Infection risk: low / elevated
- Urgency: home care OK / needs professional attention

→ Gemini generates: "Your wound is healing well! I notice some redness
   around the incision, which is common in the first week. Keep it clean
   and dry, and watch for any spreading redness or warmth..."

→ OR: "I'm noticing some signs that your provider should look at —
   there appears to be drainage from the wound along with redness.
   This isn't necessarily serious, but it's worth getting checked
   within the next day or two. Would you like help preparing
   notes for your appointment?"
```

---

## Model Architecture

### Approach: `AutoModelForImageClassification` with Selective Layer Unfreezing

Following Google's `fine_tune_for_image_classification.ipynb` notebook, adapted for T4 GPU.

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

model = AutoModelForImageClassification.from_pretrained(
    "google/medsiglip-448",
    problem_type="multi_label_classification",
    num_labels=6,
    id2label={0: "healing_status", 1: "erythema", 2: "edema",
              3: "infection_risk", 4: "urgency", 5: "exudate"},
    label2id={"healing_status": 0, "erythema": 1, "edema": 2,
              "infection_risk": 3, "urgency": 4, "exudate": 5},
    ignore_mismatched_sizes=True,
)
```

### T4 Adaptation: Selective Freezing

Google's notebook fine-tunes all 400M params on A100 (40GB). T4 has 16GB. We freeze most of the encoder and unfreeze only the last N blocks + the classification head.

```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier head (randomly initialized — MUST be trainable)
for param in model.classifier.parameters():
    param.requires_grad = True

# Unfreeze last N encoder blocks
N_UNFREEZE = 4  # Start with 2 if VRAM tight, scale to 4
for layer in model.vision_model.encoder.layers[-N_UNFREEZE:]:
    for param in layer.parameters():
        param.requires_grad = True

# Verification
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Expected: ~50-60M trainable / ~400M total (~13-15%)
```

**VRAM estimate (N_UNFREEZE=4):**
- Frozen weights in fp32: ~1.6GB
- Trainable params (~60M) + Adam states: ~0.7GB
- Gradients: ~0.24GB
- Activations (batch_size=4, 448×448): ~2-4GB
- **Total: ~5-7GB → fits T4 comfortably**

---

## Loss Function: Masked BCE with pos_weight

### Why Masked Loss

Three labels (erythema, edema, exudate) have MISSING values. Without masking, we'd either:
- Drop rows → lose training signal for the labels they DO have
- Encode MISSING as 0 → introduce label noise

Masked loss lets every image contribute to every label it has data for.

```python
import torch
from torch.nn import BCEWithLogitsLoss

# Precomputed from training set (non-MISSING samples only)
POS_WEIGHT = torch.tensor([
    1.42,   # healing_status
    2.59,   # erythema
    6.56,   # edema
    5.15,   # infection_risk
    7.42,   # urgency
    5.24,   # exudate
])

def masked_bce_loss(outputs, labels, num_items_in_batch):
    logits = outputs.get("logits")  # shape: (batch, 6)

    # labels: shape (batch, 6) — valid entries are 0.0 or 1.0, MISSING encoded as -1.0
    mask = (labels >= 0).float()           # 1.0 for valid, 0.0 for MISSING
    safe_labels = labels.clamp(min=0)      # Replace -1 with 0 for BCE math

    pos_weight = POS_WEIGHT.to(logits.device)
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    per_element_loss = loss_fn(logits, safe_labels)  # shape: (batch, 6)

    # Zero out loss for MISSING entries
    masked_loss = per_element_loss * mask

    # Average over VALID entries only (not batch size × 6)
    return masked_loss.sum() / mask.sum()
```

---

## Preprocessing Pipeline

Exact reproduction of Google's notebook preprocessing:

```python
from torchvision.transforms import (
    Compose, CenterCrop, Resize, ToTensor, Normalize, InterpolationMode
)
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/medsiglip-448")
IMG_SIZE = image_processor.size["height"]  # 448
IMG_MEAN = image_processor.image_mean      # [0.5, 0.5, 0.5]
IMG_STD = image_processor.image_std        # [0.5, 0.5, 0.5]

_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    ToTensor(),
    Normalize(mean=IMG_MEAN, std=IMG_STD),  # pixel range [-1, 1]
])

def preprocess_image(image):
    """Apply Google's exact preprocessing: zero-pad to square, then resize."""
    image = image.convert("RGB")
    # CenterCrop(max(image.size)) zero-pads to square — matches Google's training
    max_dim = max(image.size)
    image = CenterCrop(max_dim)(image)
    return _transform(image)
```

### Why CenterCrop Matters

Google's training data was preprocessed with `CenterCrop(max(image.size))` before resizing. This zero-pads non-square images to a square canvas, preserving aspect ratio. If we skip this, our pixel statistics differ from what MedSigLIP was trained on, degrading feature extraction quality. SurgWound images are likely rectangular (smartphone photos), so this step is critical.

---

## Training Configuration

Adapted from Google's notebook for T4 constraints:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="medsiglip-448-surgwound-6label",
    
    # Epochs & batching
    num_train_epochs=5,                  # Google uses 3; we use 5 for smaller dataset
    per_device_train_batch_size=4,       # Google uses 8; halved for T4
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,      # Effective batch = 4×16 = 64 (same as Google's 8×8)
    
    # Optimizer (same as Google)
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10,                     # Google uses 5; slightly more for smaller dataset
    lr_scheduler_type="cosine",
    
    # T4-specific
    fp16=True,                           # T4 doesn't support bf16; Google doesn't need this on A100
    dataloader_num_workers=2,
    
    # Evaluation & saving
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc_macro",
    greater_is_better=True,
    
    # Logging & output
    logging_steps=10,
    report_to="tensorboard",
    push_to_hub=True,
    hub_model_id="<username>/medsiglip-448-mamaguard-wound",
)
```

### Key Differences from Google's Notebook

| Parameter | Google (A100) | Ours (T4) | Reason |
|---|---|---|---|
| batch_size | 8 | 4 | VRAM constraint |
| grad_accum | 8 | 16 | Maintain effective batch=64 |
| epochs | 3 | 5 | Smaller dataset needs more passes |
| fp16 | not needed | True | T4 has no bf16; fp16 saves VRAM |
| warmup_steps | 5 | 10 | Proportionally more warmup |
| num_labels | 10 (SCIN) | 6 | Our label count |
| All params trainable | Yes | Last 4 blocks + head | T4 VRAM |

---

## Evaluation Metrics

Following Google's notebook: macro-averaged One-vs-Rest ROC AUC + per-label sensitivity/specificity.

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    results = {}
    aucs = []

    label_names = ["healing_status", "erythema", "edema",
                   "infection_risk", "urgency", "exudate"]

    for i, name in enumerate(label_names):
        # Mask out MISSING (encoded as -1)
        valid = labels[:, i] >= 0
        if valid.sum() < 10:
            continue

        y_true = labels[valid, i]
        y_prob = probs[valid, i]

        # Skip if only one class present in this eval batch
        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_prob)
        aucs.append(auc)
        results[f"auc_{name}"] = auc

        # Sensitivity & Specificity at threshold 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        results[f"sensitivity_{name}"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[f"specificity_{name}"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    results["roc_auc_macro"] = np.mean(aucs) if aucs else 0.0
    return results
```

---

## Data Collation

```python
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])  # shape: (batch, 6), float32
    return {"pixel_values": pixel_values, "labels": labels}
```

### Label Encoding from CSV

```python
def encode_labels(row):
    """Convert a CSV row to a 6-dim label tensor. MISSING → -1."""
    labels = []

    # 0: healing_status
    labels.append(1.0 if row["healing_status"] == "Not Healed" else 0.0)

    # 1: erythema
    if row["erythema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["erythema"] == "Existent" else 0.0)

    # 2: edema
    if row["edema"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(1.0 if row["edema"] == "Existent" else 0.0)

    # 3: infection_risk (binary: Low=0, Medium+High=1)
    labels.append(1.0 if row["infection_risk"] in ["Medium", "High"] else 0.0)

    # 4: urgency (binary: Green=0, Yellow+Red=1)
    labels.append(0.0 if "Green" in row["urgency_level"] else 1.0)

    # 5: exudate (binary: Non-existent=0, any exudate=1)
    if row["exudate_type"] == "MISSING":
        labels.append(-1.0)
    else:
        labels.append(0.0 if row["exudate_type"] == "Non-existent" else 1.0)

    return torch.tensor(labels, dtype=torch.float32)
```

---

## Notebook Cell Structure

```
Cell 1:  Install dependencies
         pip install transformers accelerate datasets evaluate
         pip install tensorboard scikit-learn tqdm pillow

Cell 2:  HuggingFace login
         from huggingface_hub import login
         login(token=HF_TOKEN)

Cell 3:  Configuration constants
         BASE_PATH, device, N_UNFREEZE=4, label names, pos_weights,
         id2label/label2id dicts

Cell 4:  Load CSV + show dataset statistics
         Read labels.csv, print split sizes, label distributions,
         MISSING counts, pos_weight calculations
         Sanity check: assert 480 train, 69 val, 137 test

Cell 5:  Create HuggingFace Dataset
         Load images from disk, apply encode_labels(),
         store as Dataset with columns: image, pixel_values, labels
         Apply preprocess_image as map function

Cell 6:  Visual sanity check
         Display 4 sample images with their encoded labels
         Verify a known-MISSING label shows -1.0
         Verify a known-positive label shows 1.0

Cell 7:  Load model with selective freezing
         AutoModelForImageClassification.from_pretrained(...)
         Freeze all → unfreeze classifier + last N blocks
         Print trainable param count
         VRAM check: torch.cuda.memory_allocated()

Cell 8:  Define masked loss function
         masked_bce_loss as defined above
         Quick test: create dummy logits+labels with a -1 entry,
         verify loss ignores it

Cell 9:  Define compute_metrics function
         roc_auc per label + macro average + sensitivity/specificity

Cell 10: Define collate_fn

Cell 11: Create Trainer
         Pass: model, training_args, train/eval datasets,
         collate_fn, compute_metrics, loss function
         
         trainer = Trainer(
             model=model,
             args=training_args,
             train_dataset=train_ds,
             eval_dataset=val_ds,
             data_collator=collate_fn,
             compute_metrics=compute_metrics,
             compute_loss_func=masked_bce_loss,
         )

Cell 12: Train
         trainer.train()
         Print elapsed time, final metrics

Cell 13: Evaluate on test set
         test_results = trainer.evaluate(test_ds)
         Print per-label AUC table
         Print per-label sensitivity/specificity
         Print macro AUC

Cell 14: Push to HuggingFace Hub
         trainer.push_to_hub()
         Also push image_processor for easy inference loading

Cell 15: Inference demo
         Load saved model from Hub
         Load a single test image
         Run through model, print predictions with confidence
         Show: "healing: progressing (0.73), erythema: present (0.81), ..."
```

---

## Expected Results

| Label | Expected AUC | Confidence | Notes |
|---|---|---|---|
| healing_status | 0.82–0.90 | HIGH | Balanced (41% pos), zero MISSING, most distinct visual signal |
| erythema | 0.78–0.85 | MEDIUM | Only 26 MISSING, 28% positive rate, clear visual feature |
| edema | 0.65–0.78 | LOW | 132 MISSING (27% of train data gone), only 50 positive train samples |
| infection_risk | 0.75–0.85 | MEDIUM | Binary aggregation helps; 78 positive train samples |
| urgency | 0.72–0.82 | MEDIUM | Only 57 positive train samples; but zero MISSING helps |
| exudate | 0.70–0.80 | MEDIUM | 70 positive train samples, 60 MISSING; moderate signal |

**Macro AUC target: 0.74–0.83**

These should exceed linear probe baselines because:
1. End-to-end gradient flow adapts visual features to wound domain
2. Multi-label structure captures label co-occurrence (infected wounds show erythema + exudate together)
3. Masked loss preserves all training signal (no row dropping)
4. pos_weight handles class imbalance (Google's recommended approach)

---

## Fallback Plan

If `AutoModelForImageClassification` fails on T4 (OOM, NaN, dtype issues):

### Fallback: MLP on Frozen Embeddings

From Google's `train_data_efficient_classifier.ipynb`:

```python
# Step 1: Extract embeddings (T4 or CPU, ~2 min)
# MedSigLIP → 1152-dim embedding per image

# Step 2: Train MLP (CPU, ~5 min)
# Architecture (Google's exact design):
# 1152 → Dense(512, ReLU, L2=1e-5) → Dropout(0.05)
#      → Dense(256, ReLU, L2=1e-5) → Dropout(0.10)
#      → Dense(6, sigmoid)
# Loss: masked BCE with pos_weight (same as above)
```

This is explicitly better than LogisticRegression per Google's own benchmarks, runs on CPU in 5 minutes, and still produces a 6-label multi-output classifier. Less impressive than real fine-tuning but a solid backup.

---

## Integration with MamaGuard Pipeline

### At Inference Time

```
1. User uploads wound photo via MamaGuard app
2. Image sent to Modal serverless endpoint
3. Modal loads fine-tuned MedSigLIP model
4. Model outputs 6 sigmoid scores:
   {
     "healing_status": 0.73,   // Not Healed probability
     "erythema": 0.81,         // Erythema present probability
     "edema": 0.22,            // Edema present probability
     "infection_risk": 0.67,   // Elevated infection probability
     "urgency": 0.54,          // Needs-attention probability
     "exudate": 0.45           // Exudate present probability
   }
5. Scores injected into Gemini orchestrator prompt
6. Gemini generates caring, contextual response for the user
```

### Threshold Strategy

For the prototype, use 0.5 as default threshold for all labels. In production, you'd optimize thresholds per-label on the validation set to maximize sensitivity for the safety-critical labels (urgency, infection_risk) — better to over-alert than under-alert for a postpartum woman.

---

## Known Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| T4 fp16 + SigLIP may produce NaN | MEDIUM | Test inference first before training. If NaN: try fp16=False and reduce batch_size to 2 |
| Classification head randomly initialized → epoch 1 metrics terrible | LOW | Expected behavior. Head needs 1-2 epochs to learn. Don't panic at first eval. |
| Edema probe weak (only 50 positive train samples) | HIGH | Document honestly. Consider zero-shot MedSigLIP for edema as supplementary signal |
| Red urgency has 13 train samples | MEDIUM | Binary urgency (Green vs not-Green) mitigates this. Don't present 3-class urgency predictions. |
| Dataset is mostly laparoscopic ports, not C-section scars | LOW | Tissue-level pathology transfers. Document as known limitation. The signs of infection look identical regardless of incision shape. |
| Chinese text overlays in some images | LOW | MedSigLIP should be robust to text overlays — it's trained on medical images with annotations. May actually help generalization. |

---

## Judge Narrative

> "We fine-tuned MedSigLIP's vision encoder for multi-label surgical wound assessment using Google's recommended `AutoModelForImageClassification` approach with selective layer unfreezing adapted for T4 GPU. The model simultaneously predicts 6 clinical attributes — healing status, erythema, edema, infection risk, urgency level, and exudate presence — from a single forward pass. We implemented masked loss to handle partially-labeled training data across 686 real-world smartphone wound photos, preserving all training signal. The structured clinical scores feed directly into our Gemini orchestrator, which translates them into empathetic, actionable guidance for postpartum women monitoring their C-section wounds at home."

---

## Files & Artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Training labels | `labels.csv` | 686 rows, 8 fields + split column |
| Training images | `images/*.jpg` | 686 JPEG files |
| Fine-tuned model | HuggingFace Hub: `<user>/medsiglip-448-mamaguard-wound` | Load with one line for inference |
| This plan | `medsiglip_finetune_plan_v2.md` | Implementation reference |
| Previous plan (v1) | `medsiglip_final_plan.md` | Superseded — had 4 labels, not 6 |
