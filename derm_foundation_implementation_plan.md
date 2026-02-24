# Derm Foundation — Implementation Plan for Momnetrix / MamaGuard

**Date:** Feb 22, 2026  
**Hackathon:** HAI-DEF MedGemma Impact Challenge (Deadline: Feb 24, 2026)  
**Model:** Google Derm Foundation (HAI-DEF family)

---

## 1. What Derm Foundation Is

- **Architecture:** BiT-M ResNet101x3 CNN (Convolutional Neural Network)
- **Framework:** Keras / TensorFlow (NOT PyTorch — separate container needed)
- **Input:** Skin image (any resolution, consumer smartphone photos supported)
- **Output:** 6,144-dimensional embedding vector
- **What it does NOT do:** It does NOT classify, diagnose, or label anything on its own. It is purely an embedding extractor. A downstream classifier must be trained on top of the embeddings.
- **HuggingFace ID:** `google/derm-foundation`
- **Loading:** `from_pretrained_keras("google/derm-foundation")`

---

## 2. Why We're Using It

### Hackathon Strategy (20% of scoring = "Effective Use of HAI-DEF")
- Derm Foundation is our **5th HAI-DEF model** (alongside MedGemma, MedSigLIP, MedASR, Gemini 2.5 Pro)
- It serves a **completely distinct clinical role** from MedSigLIP:
  - **MedSigLIP** = surgical wound assessment (C-section incision monitoring, 6 wound labels)
  - **Derm Foundation** = general skin condition screening (rashes, infections, inflammations)
- Same input modality (phone camera), different clinical domain — this is elegant, not redundant

### Clinical Rationale
- Pregnant and postpartum women experience elevated rates of skin conditions due to immune suppression and hormonal shifts
- Common conditions that flare during pregnancy: eczema, contact dermatitis, urticaria, fungal infections (tinea), folliculitis, psoriasis, drug rashes
- These are ALL conditions that our classifier can identify
- When the classifier cannot confidently match a condition, MedGemma adds pregnancy-specific context and escalation guidance

---

## 3. Fundamental Starting Points / Assumptions

### What We Have
1. **Google's SCIN dataset** — 10,000+ consumer smartphone photos of skin conditions, labeled by 1–3 dermatologists each, with confidence scores. Hosted on Google Cloud Storage, free and open access.
2. **Precomputed embeddings** — Google already ran every SCIN image through Derm Foundation and published the resulting embeddings as a 514MB `.npz` file on HuggingFace. We do NOT need to run Derm Foundation inference on the training set ourselves.
3. **Google's official classifier notebook** — `train_data_efficient_classifier.ipynb` in the `Google-Health/derm-foundation` GitHub repo. Trains a multi-label sklearn classifier on SCIN embeddings. Runs on CPU in minutes.

### What We Are NOT Doing
- **No fine-tuning of Derm Foundation itself** — the model weights are frozen. We use it as-is for embedding extraction.
- **No fine-tuning of MedGemma for derm** — MedGemma stays focused on vitals reasoning (its fine-tuned strength). Pregnancy-context interpretation of derm results goes through Gemini 2.5 Pro via prompt engineering.
- **No pregnancy-specific skin condition dataset** — PUPPP, pemphigoid gestationis, cholestasis skin changes have ZERO open-source labeled image datasets. We are not pretending to classify these.
- **No scraping images from the internet** — ethically and legally problematic, and yields unreliable labels.

### Key Constraint
- Derm Foundation is TensorFlow/Keras. Our other models are PyTorch. This means a **separate Modal container** with TensorFlow installed for the Derm Foundation endpoint. Same pattern as MedASR.

---

## 4. The 10 Conditions We Classify

Trained using SCIN dataset labels:

| # | Condition | Training Samples | Pregnancy Relevance |
|---|-----------|-----------------|---------------------|
| 1 | Eczema | 1,211 | Flares due to immune changes; most common pregnancy skin issue |
| 2 | Allergic Contact Dermatitis | 952 | New sensitivities develop during pregnancy |
| 3 | Insect Bite | 449 | Baseline condition, not pregnancy-specific |
| 4 | Urticaria (hives) | 377 | Can indicate PUPPP in late pregnancy if on abdomen/striae |
| 5 | Psoriasis | 348 | May improve or worsen during pregnancy unpredictably |
| 6 | Folliculitis | 297 | Pruritic folliculitis of pregnancy is a recognized dermatosis |
| 7 | Irritant Contact Dermatitis | 254 | Increased skin sensitivity during pregnancy |
| 8 | Tinea (fungal) | 232 | Immune suppression increases susceptibility |
| 9 | Herpes Zoster (shingles) | 157 | Rare but serious in pregnancy |
| 10 | Drug Rash | 156 | Critical — new medications (labetalol, iron, etc.) in pregnancy |

**Classification type:** Multi-label (a photo can match multiple conditions simultaneously)

---

## 5. The Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PHONE CAMERA                                           │
│  Woman photographs rash/skin concern                    │
└──────────────────────┬──────────────────────────────────┘
                       │ image
                       ▼
┌─────────────────────────────────────────────────────────┐
│  DERM FOUNDATION (Modal endpoint — TensorFlow)          │
│  Image → 6,144-dim embedding vector                     │
└──────────────────────┬──────────────────────────────────┘
                       │ embedding
                       ▼
┌─────────────────────────────────────────────────────────┐
│  SKLEARN CLASSIFIER (runs inside same container)        │
│  Embedding → probability scores for 10 conditions       │
│  Output: {eczema: 0.72, urticaria: 0.18, ...}           │
└──────────────────────┬──────────────────────────────────┘
                       │ classification + confidence
                       ▼
┌─────────────────────────────────────────────────────────┐
│  GEMINI 2.5 PRO (orchestrator)                          │
│  Receives:                                              │
│    - Classification result + confidence scores          │
│    - Patient gestational age (from app profile)         │
│    - Current vitals from watch (HR, temp, etc.)         │
│    - The original photo (Gemini is multimodal)          │
│  Produces:                                              │
│    - Pregnancy-contextualized guidance                  │
│    - Escalation recommendation if warranted             │
│    - Saves to Visit Prep Summary                        │
└─────────────────────────────────────────────────────────┘
```

### Why Gemini, Not MedGemma, for Context Layer
- MedGemma 4B is fine-tuned for vitals → risk reasoning. That's its job. Don't dilute it.
- Gemini 2.5 Pro is massive, general-purpose, multimodal, and handles contextual reasoning perfectly with a good system prompt.
- No training data exists for "derm classification + pregnancy context" input-output pairs, so fine-tuning MedGemma for this is not feasible.

---

## 6. How Pregnancy Context Actually Works

The intelligence is NOT in a confidence threshold. It's in Gemini knowing the patient is pregnant and interpreting any result through that lens.

**Example outputs by scenario:**

**High confidence eczema + 34 weeks pregnant:**
> "This appears to be eczema, which commonly flares during pregnancy due to immune changes. Topical emollients and mild hydrocortisone are generally considered safe. Avoid oral corticosteroids without provider approval."

**High confidence urticaria + 36 weeks + rash on abdomen:**
> "This resembles urticaria (hives). However, hive-like rashes appearing on the abdomen in the third trimester — especially if they started in your stretch marks — can also indicate PUPPP, a pregnancy-specific condition affecting 1 in 160 pregnancies. PUPPP is benign but very uncomfortable. Mention the location and timing to your provider."

**High confidence drug rash + recently started labetalol:**
> "This may be a drug rash. You started labetalol 10 days ago for blood pressure management. Drug rashes typically appear 1–2 weeks after starting a new medication. Contact your provider before discontinuing any medication."

**Low confidence across all categories:**
> "This rash doesn't clearly match common skin conditions. During pregnancy, there are several pregnancy-specific conditions that require professional evaluation. We recommend showing this photo to your OB-GYN or dermatologist, especially if you're experiencing intense itching."

---

## 7. Implementation Steps

| Step | Task | Time Est. | GPU? |
|------|------|-----------|------|
| 1 | Download SCIN precomputed embeddings from HuggingFace (`scin_dataset_precomputed_embeddings.npz`, 514MB) | 5 min | No |
| 2 | Run Google's `train_data_efficient_classifier.ipynb` notebook on SCIN 10-label data — copy-paste, minimal changes | 20 min | No (CPU sklearn) |
| 3 | Export trained classifier as pickle file (~1MB) | 1 min | No |
| 4 | Test classifier locally: feed sample SCIN images, verify probability outputs make sense | 15 min | No |
| 5 | Create Modal container for Derm Foundation: TensorFlow + Keras + sklearn + pickle file | 1–2 hrs | T4 for inference |
| 6 | Build API endpoint: receives image → returns classification JSON `{condition: score, ...}` | 30 min | — |
| 7 | Wire into app: photo capture UI → endpoint call → Gemini contextualization → display result | 1–2 hrs | — |
| 8 | Add to Visit Prep Summary: skin screening results + photos feed into the agentic summary generator | 30 min | — |
| **Total** | | **~4–5 hours** | Minimal |

---

## 8. What We Tell the Judges

> "We use Derm Foundation with Google's SCIN dataset to screen for 10 common dermatological conditions from smartphone photos. For pregnant and postpartum women, we add a pregnancy-aware context layer powered by Gemini 2.5 Pro: every classification is interpreted through the patient's gestational age, current vitals, and medication history. Conditions like urticaria on the abdomen in the third trimester trigger additional differential considerations including PUPPP. This complements our MedSigLIP wound assessment — MedSigLIP monitors the C-section incision, while Derm Foundation screens for everything else on the skin."

**Model count in this single pipeline: 3 HAI-DEF models** (Derm Foundation → MedGemma vitals context → Gemini synthesis)

---

## 9. Files and Resources

| Resource | Location |
|----------|----------|
| Derm Foundation model | `google/derm-foundation` on HuggingFace |
| SCIN dataset | `gs://dx-scin-public-data` on Google Cloud Storage |
| Precomputed embeddings | HuggingFace `google/derm-foundation` repo files |
| Classifier notebook | `github.com/Google-Health/derm-foundation/notebooks/train_data_efficient_classifier.ipynb` |
| SCIN demo notebook | `github.com/google-research-datasets/scin/blob/main/scin_demo.ipynb` |
| SCIN paper | `arxiv.org/abs/2402.18545` |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TensorFlow/Keras container bloat on Modal | Slow cold starts in demo | `keep_warm=1` on Modal function |
| SCIN classifier accuracy on pregnancy-affected skin | Slight domain shift (SCIN is general population) | Honest framing — we're screening common conditions, not diagnosing pregnancy-specific ones |
| Judge asks "why not classify PUPPP directly?" | Could seem like a gap | Answer: "No labeled dataset exists. Our triage approach is clinically responsible — even dermatologists need biopsy to distinguish PUPPP from pemphigoid gestationis." |
| Photo quality from consumer phones | Lower accuracy | Derm Foundation was trained on mixed device types including smartphones; SCIN images are consumer-quality by design |
