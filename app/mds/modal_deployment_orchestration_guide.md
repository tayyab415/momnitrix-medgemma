# MamaGuard Modal Deployment and Orchestration Guide

## The Complete Model Inventory

Before getting into containers and GPUs, here's our full model lineup with verified specs from all our fine-tuning and research sessions:

| Model | Architecture | Parameters | Framework | Fine-Tuned? | What It Does |
|---|---|---|---|---|---|
| MedGemma 1.5 4B-IT | Decoder-only Transformer (Gemma 3 base) | 4 billion | PyTorch (transformers 4.x, PEFT, Unsloth) | YES — LoRA SFT on Maternal Health Risk Dataset (1,218 instruction-response pairs) | Central orchestrator. Receives ALL signals, produces unified clinical reasoning and patient-facing responses |
| MedSigLIP-448 | SigLIP SO400M dual-tower vision encoder + classification head | ~400 million | PyTorch (transformers 4.x, HuggingFace Trainer) | YES — Selective fine-tune (last 4 encoder blocks + classifier head) on SurgWound dataset (686 images, 6 labels) | Wound photo assessment: healing_status, erythema, edema, infection_risk, urgency, exudate |
| Derm Foundation | BiT-M ResNet101x3 CNN | ~380 million (estimated from BiT-M x3 width multiplier) | TensorFlow/Keras | NO — Embedding extractor only. Sklearn classifier trained on SCIN precomputed embeddings | Skin condition screening: 10-class classification (eczema, urticaria, psoriasis, contact dermatitis, etc.) |
| MedASR | Conformer (CNN + Transformer hybrid, CTC decoder) | 105 million | PyTorch (transformers 5.0+) | NO — Used as-is with greedy decoding | Medical speech-to-text for voice symptom check-ins |
| Gemini 2.5 Pro | Google's proprietary multimodal LLM | Unknown (API-only) | API call — no local deployment | NO — Prompt engineering only | Final patient-facing response synthesis, tone calibration, Visit Prep Summary generation |

---

## Modal Container Architecture

We need four separate containers on Modal. Each container is independent because of framework conflicts (TensorFlow vs PyTorch), transformers version conflicts (4.x vs 5.0+), and different GPU requirements. Plus, separate containers mean they scale independently — wound assessment doesn't need to be running when the user is doing a voice check-in.

### Container 1: MedGemma 4B — The Brain

**What it runs:** MedGemma 1.5 4B-IT base model + your LoRA adapter (mamaguard-vitals-lora) loaded via PEFT

**Role in the pipeline:** This is the central intelligence. It receives structured data from every other model and synthesizes it into clinical reasoning. It is NOT called directly by the phone app — it's called by the orchestrator endpoint after all other model results are collected.

**GPU:** T4 (16GB VRAM) — sufficient because we load in 4-bit quantization via bitsandbytes

**VRAM budget:**
- Base model in 4-bit: ~2.6 GB
- LoRA adapter weights: ~50-100 MB (negligible)
- KV cache for inference (2048 token context): ~0.5 GB
- Activations and overhead: ~1 GB
- Total estimated: ~4-5 GB — well within T4's 16 GB

**Key dependencies:**
- transformers 4.x (compatible version with MedSigLIP)
- peft (for LoRA adapter loading)
- bitsandbytes (for 4-bit quantization)
- accelerate

**LoRA loading approach:** Load base model in 4-bit, then apply LoRA adapter via PeftModel.from_pretrained(). Alternatively, merge and unload the adapter for simpler deployment (no PEFT dependency at inference, but slightly larger model file).

**Container configuration notes:**
- keep_warm=1 (critical for demo — avoids cold start delays)
- timeout=120 seconds (MedGemma 4B in 4-bit generates at reasonable speed, but long clinical reasoning responses can take 30-60 seconds)
- Concurrency limit: 1 (single GPU, single request at a time)

**What MedGemma receives as input:**

A structured prompt containing ALL model outputs plus patient context. The prompt template includes:

- Watch vitals block: HR, SpO2, skin temp, BP, HRV (from Samsung Watch via phone app)
- Wound assessment block: 6 sigmoid scores from MedSigLIP (if wound photo was submitted)
- Skin screening block: top-3 condition matches with confidence scores from Derm Foundation (if skin photo was submitted)
- Voice transcript block: MedASR transcription text (if voice check-in was recorded)
- Patient context block: gestational age, medication history, known conditions
- Safety rules block: escalation thresholds and red-flag symptoms

MedGemma produces a JSON-structured response containing: risk_level (green/yellow/red), clinical_reasoning (internal), patient_message (empathetic, accessible language), and action_items (specific next steps).

---

### Container 2: MedSigLIP — The Wound Eye

**What it runs:** Your fine-tuned MedSigLIP-448 model (AutoModelForImageClassification, pushed to HuggingFace Hub)

**Role in the pipeline:** Accepts a wound photo, outputs 6 sigmoid scores for healing_status, erythema, edema, infection_risk, urgency, and exudate. These scores are passed to MedGemma as structured input.

**GPU:** T4 (16GB VRAM) — this is the same GPU we used for training, inference is much lighter

**VRAM budget:**
- Full model weights in fp32: ~1.6 GB
- Single image inference activations at 448x448: ~0.5 GB
- Total estimated: ~2-3 GB — extremely comfortable on T4

**Key dependencies:**
- transformers 4.x (same version as Container 1)
- torchvision (for image preprocessing)
- Pillow

**Inference flow:**
1. Receive base64-encoded JPEG from phone app
2. Decode, convert to RGB, apply Google's exact preprocessing (CenterCrop to square, resize to 448x448, normalize to [-1, 1])
3. Forward pass through fine-tuned model
4. Apply sigmoid to logits
5. Return JSON with 6 scores

**Container configuration notes:**
- keep_warm=1 (same demo reasoning)
- timeout=30 seconds (inference is fast, < 1 second per image)
- Could potentially share a container with MedGemma since they're both PyTorch + transformers 4.x, but keeping them separate is cleaner and avoids GPU memory contention

**Why not CPU?** You could run MedSigLIP on CPU for inference (it's only 400M params). But the T4 makes inference sub-second vs 3-5 seconds on CPU. For a demo, sub-second matters.

---

### Container 3: Derm Foundation + Sklearn — The Skin Screener

**What it runs:** Derm Foundation (BiT-M ResNet101x3, TensorFlow/Keras) for embedding extraction + a trained sklearn classifier (loaded from pickle) for 10-class skin condition scoring

**Role in the pipeline:** Accepts a skin photo, extracts a 6,144-dimensional embedding via Derm Foundation, then runs the sklearn classifier to produce probabilities for 10 skin conditions. Results are passed to MedGemma for pregnancy-aware contextualization.

**GPU:** T4 (16GB VRAM) — Derm Foundation is a large CNN, benefits from GPU acceleration

**VRAM budget:**
- BiT-M ResNet101x3 in fp32: ~1.5 GB (estimated from ~380M params)
- TensorFlow runtime overhead: ~1-2 GB
- Image processing activations: ~0.5 GB
- Sklearn classifier: negligible (loaded in RAM, not VRAM)
- Total estimated: ~3-4 GB

**Key dependencies:**
- tensorflow (2.x)
- keras
- huggingface_hub (from_pretrained_keras for loading)
- scikit-learn (for the trained classifier)
- numpy

**CRITICAL: This container CANNOT share with Containers 1 or 2.** Derm Foundation runs on TensorFlow. MedGemma and MedSigLIP run on PyTorch. Mixing TF and PyTorch in one container causes CUDA context conflicts and VRAM fragmentation. Separate containers are mandatory.

**Inference flow:**
1. Receive base64-encoded image from phone app
2. Decode, convert to PNG bytes
3. Serialize as tf.train.Example (Google's required input format)
4. Extract 6,144-dim embedding via loaded_model.signatures["serving_default"]
5. Run embedding through sklearn classifier
6. Return JSON with 10 condition probabilities + top-3 matches

**Container configuration notes:**
- keep_warm=1
- timeout=30 seconds
- The TensorFlow container image will be larger than PyTorch containers (~3-4 GB image size vs ~2 GB). Account for this in build time.
- The sklearn classifier pickle file (~1 MB) should be baked into the container image, not downloaded at startup

**Why we need a separate sklearn classifier:** Derm Foundation is an embedding model only. It outputs a 6,144-dimensional vector, not classifications. Google provides precomputed SCIN embeddings (514 MB .npz file on HuggingFace) and a notebook that trains an sklearn classifier on these embeddings in about 20 minutes on CPU. The trained classifier is what actually maps embeddings to the 10 condition labels.

---

### Container 4: MedASR — The Ears

**What it runs:** MedASR (105M param Conformer, AutoModelForCTC)

**Role in the pipeline:** Accepts audio, produces medical text transcription. Transcript is passed to MedGemma.

**GPU:** CPU-only is viable (105M params is tiny), but T4 is faster for real-time experience

**VRAM budget (if using GPU):**
- Model weights: ~0.4 GB
- Audio processing buffers: ~0.1 GB
- Total: < 1 GB — ridiculously light

**Key dependencies:**
- transformers 5.0+ (THIS IS THE VERSION CONFLICT — cannot share container with anything else)
- librosa (audio resampling)
- torch

**CRITICAL: transformers 5.0+ requirement.** MedASR uses a custom model class (AutoModelForCTC with Conformer architecture) that was only added in transformers 5.0. MedGemma and MedSigLIP use transformers 4.x. These are incompatible. This container MUST be separate. This is not a preference — it will crash if you try to combine them.

**Inference flow:**
1. Receive base64-encoded WAV audio from phone app
2. Decode, load audio with librosa, resample to 16kHz
3. Process through MedASR AutoProcessor
4. Run AutoModelForCTC inference
5. Decode output tokens via processor.batch_decode()
6. Return plain text transcript

**Container configuration notes:**
- keep_warm=0 is acceptable here (MedASR is tiny, cold start is fast, and voice check-ins happen less frequently than other interactions)
- timeout=60 seconds (for longer audio clips)
- CPU mode is a viable cost-saving option. Modal charges less for CPU-only containers. Since voice check-ins aren't latency-critical (user submits audio and waits for full pipeline response), the 2-3 second CPU inference time is acceptable.

---

## The Orchestrator: How Everything Connects

The orchestrator is NOT a separate container with a model. It's a lightweight Python function on Modal (CPU-only, no GPU) that coordinates the entire pipeline. It's the endpoint that the phone app actually calls.

### Orchestration Flow

```
PHONE APP
   │
   │  HTTP POST with:
   │  - vitals JSON (from watch)
   │  - wound_image base64 (optional)
   │  - skin_image base64 (optional)
   │  - audio base64 (optional)
   │  - patient_context JSON (gestational age, meds, etc.)
   │
   ▼
ORCHESTRATOR ENDPOINT (CPU, no GPU)
   │
   ├── IF wound_image present:
   │   └── Call Container 2 (MedSigLIP) → wound_scores
   │
   ├── IF skin_image present:
   │   └── Call Container 3 (Derm Foundation) → skin_scores
   │
   ├── IF audio present:
   │   └── Call Container 4 (MedASR) → transcript
   │
   │  (These calls happen in PARALLEL using Modal's async capabilities)
   │
   ├── ASSEMBLE all results into MedGemma prompt:
   │   ├── vitals (always present)
   │   ├── wound_scores (if available)
   │   ├── skin_scores (if available)
   │   ├── transcript (if available)
   │   ├── patient_context (always present)
   │   └── safety_rules (always present)
   │
   ├── Call Container 1 (MedGemma) → clinical_response
   │
   ├── OPTIONALLY call Gemini 2.5 Pro API for:
   │   └── Patient-facing tone refinement
   │   └── Visit Prep Summary generation
   │
   └── Return final JSON response to phone app
```

### Key Orchestration Design Decisions

**Parallel execution of specialist models:** The orchestrator fires off MedSigLIP, Derm Foundation, and MedASR simultaneously (not sequentially). They're independent — wound analysis doesn't need to wait for voice transcription. This reduces total latency from (sum of all model times) to (max of all model times + MedGemma time). Estimated: 1-2 seconds for specialists + 10-15 seconds for MedGemma reasoning = 12-17 seconds total round-trip.

**MedGemma is always the final synthesizer:** No matter which combination of inputs is available, MedGemma always receives them and produces the unified response. If only vitals are available (no photo, no audio), MedGemma still runs with just vitals. If all four input types are present, MedGemma gets the richest context possible. This graceful degradation means the system works even if the user skips the photo or voice check-in.

**Gemini 2.5 Pro is optional polish, not mandatory:** The core clinical reasoning happens in MedGemma (your fine-tuned model). Gemini's role is optional: refining the tone for patient-facing communication, generating structured Visit Prep Summaries, or handling edge cases where MedGemma's 4B reasoning isn't sufficient. For the hackathon demo, you can hardcode whether Gemini is called or not. In production, Gemini would be called for Visit Prep Summary generation and for any response where MedGemma's confidence is low.

**The orchestrator itself is stateless:** It receives everything it needs in the HTTP POST from the phone app. Patient context (gestational age, medication history) is stored in the phone app's local database and sent with each request. The orchestrator doesn't maintain sessions or memory across calls.

---

## GPU and Cost Budget

### Modal GPU Pricing (approximate, as of early 2026)

| GPU | VRAM | Approx $/hour | Our Usage |
|---|---|---|---|
| T4 | 16 GB | ~$0.60 | Containers 1, 2, 3, 4 |
| A10G | 24 GB | ~$1.10 | Not needed (everything fits on T4) |
| A100 (40GB) | 40 GB | ~$3.00+ | Not needed (dropped MedGemma 27B) |

**Why we don't need A10G or A100:** Every single model fits comfortably on T4. MedGemma 4B in 4-bit uses ~4-5 GB. MedSigLIP uses ~2-3 GB. Derm Foundation uses ~3-4 GB. MedASR uses < 1 GB. No model exceeds T4's 16 GB even with generous overhead estimates. Using A10G or A100 would burn Modal credits for zero benefit.

### Per-Container Cost Estimate

With keep_warm=1, each container is running (and billing) even when idle:

| Container | GPU | keep_warm | Hourly cost | Per demo session (~5 min active) |
|---|---|---|---|---|
| MedGemma 4B | T4 | 1 | ~$0.60 | ~$0.05 |
| MedSigLIP | T4 | 1 | ~$0.60 | ~$0.05 |
| Derm Foundation | T4 | 1 | ~$0.60 | ~$0.05 |
| MedASR | CPU-only | 0 | ~$0.03 (on-demand) | ~$0.01 |
| Orchestrator | CPU-only | 1 | ~$0.03 | ~$0.01 |
| **Total active (keep_warm)** | | | **~$1.83/hour** | |

**With $200 in Modal credits:** You can keep warm for ~109 hours. That's about 4.5 full days of continuous uptime. For the hackathon, you probably need 2-3 days of development + demo readiness. Budget is sufficient but not lavish — turn off keep_warm when you're not actively testing.

**Cost optimization for development:** During development, set keep_warm=0 on all containers. Only switch to keep_warm=1 for demo recording day and the judging period. Cold starts on T4 are typically 30-60 seconds, which is fine for development iteration.

### Alternative: Consolidate to Fewer Containers

If you want to save credits, you could potentially merge Containers 1 and 2 (MedGemma + MedSigLIP) since they share the same PyTorch + transformers 4.x stack. Both together would use ~7-8 GB VRAM, still within T4's 16 GB. This saves one keep_warm container (~$0.60/hour).

You CANNOT merge Container 3 (TensorFlow) or Container 4 (transformers 5.0+) with anything else. Those framework conflicts are hard blockers.

---

## The MedGemma Orchestration Prompt Design

This is where the magic happens. MedGemma's system prompt needs to be carefully designed to handle variable inputs (some check-ins have photos, some don't) and produce consistently structured outputs.

### System Prompt Structure

The system prompt establishes MedGemma's role: a maternal health clinical reasoning engine that receives structured data from multiple sensors and models, and produces pregnancy-aware risk assessments with empathetic, actionable guidance.

Key elements of the system prompt:

1. **Role definition**: "You are a maternal health clinical reasoning assistant. You receive multi-modal health data from a pregnant woman's monitoring system and synthesize it into actionable guidance."

2. **Input schema**: Clearly defines each data block MedGemma can expect (vitals, wound scores, skin scores, transcript, patient context). States that any block may be absent.

3. **Clinical safety rules**: Hard-coded red flags that ALWAYS trigger urgent escalation regardless of other signals. Examples: systolic BP above 160, mentions of visual changes + headaches, decreased fetal movement, vaginal bleeding after 20 weeks.

4. **Output schema**: Defines the exact JSON structure MedGemma must produce: risk_level, reasoning, patient_message, action_items, flags_for_provider.

5. **Cross-signal correlation rules**: The secret sauce. Explicit instructions like: "When wound shows elevated urgency AND voice transcript mentions fever or chills, escalate infection risk." Or: "When BP is borderline AND transcript mentions headaches AND skin shows urticaria on abdomen in third trimester, consider both preeclampsia and PUPPP differential."

6. **Tone guidelines**: Patient-facing messages must be warm, non-alarmist but clear, avoid medical jargon (translate clinical terms), and always end with a specific action the user can take.

### How MedGemma Handles Different Interaction Types

**Vitals-only check-in (most common — watch sync):**
MedGemma receives watch data, compares against personal baselines, and produces trend analysis. "Your blood pressure has been steadily rising over the past 3 days. Today's reading of 138/88 is above your baseline of 120/75. Let's keep monitoring, but if it reaches 140/90 please contact your provider."

**Vitals + wound photo:**
MedGemma receives watch data AND MedSigLIP's 6 scores. It correlates wound inflammation (erythema, infection_risk) with systemic signs (elevated HR, fever). "Your C-section wound shows some redness (erythema detected). Your heart rate is slightly elevated at 92 bpm. While some redness is normal in the first week, the combination with elevated heart rate is worth watching. If you develop warmth around the incision or fever, contact your provider."

**Vitals + skin photo:**
MedGemma receives watch data AND Derm Foundation's condition scores. It adds pregnancy context. "The rash on your abdomen was identified as most closely matching urticaria (hives). At 35 weeks, hive-like rashes on the abdomen can sometimes indicate PUPPP, a harmless but itchy pregnancy condition. If the rash started in your stretch marks and is intensely itchy, mention it to your OB-GYN at your next visit."

**Vitals + voice check-in:**
MedGemma receives watch data AND MedASR transcript. It extracts symptoms from the transcript and correlates with vitals. "You mentioned headaches and ankle swelling in your check-in. Your blood pressure today is 142/91, which is elevated. Headaches combined with swelling and high blood pressure at 34 weeks are warning signs that your provider needs to evaluate. Please call your OB-GYN's office today."

**Full multi-modal (all inputs):**
MedGemma receives everything. This is the richest assessment and the most impressive demo scenario. All signals are correlated for the most comprehensive picture possible.

---

## Deployment Checklist

### Pre-Deployment

- [ ] MedGemma LoRA adapter pushed to HuggingFace Hub
- [ ] MedSigLIP fine-tuned model pushed to HuggingFace Hub
- [ ] Derm Foundation sklearn classifier saved as pickle file
- [ ] SCIN classifier validated on holdout test set
- [ ] MedASR tested on sample maternal health audio clips
- [ ] System prompt for MedGemma finalized and tested
- [ ] Gemini 2.5 Pro API key configured

### Container Build Order

Build and test in this order (each depends on the previous being verified):

1. **Container 2 (MedSigLIP)** — simplest, pure PyTorch classification, verify image in → 6 scores out
2. **Container 3 (Derm Foundation)** — TensorFlow, verify image in → 10 condition scores out
3. **Container 4 (MedASR)** — transformers 5.0+, verify audio in → text out
4. **Container 1 (MedGemma)** — most complex, verify structured prompt in → clinical response out
5. **Orchestrator endpoint** — wire everything together, test with curl/Postman
6. **Phone app integration** — connect the app to the orchestrator endpoint

### Post-Deployment Verification

- [ ] End-to-end test: vitals JSON → MedGemma response (no images, no audio)
- [ ] End-to-end test: vitals + wound photo → MedSigLIP scores + MedGemma response
- [ ] End-to-end test: vitals + skin photo → Derm Foundation scores + MedGemma response
- [ ] End-to-end test: vitals + audio → MedASR transcript + MedGemma response
- [ ] End-to-end test: all four inputs simultaneously → full multi-modal response
- [ ] Latency test: total round-trip time under 20 seconds
- [ ] Cold start test: all containers start from cold within 60 seconds

---

## Summary: The Five HAI-DEF Models Story

For judges, the narrative is clear:

"MamaGuard uses five Google Health AI Developer Foundation models in concert, each with a distinct clinical role. MedGemma 1.5 4B, fine-tuned with LoRA on maternal health vitals data, serves as the central reasoning engine that synthesizes all signals. MedSigLIP-448, fine-tuned for multi-label surgical wound assessment, monitors C-section incision healing. Derm Foundation, paired with an sklearn classifier trained on the SCIN dataset, screens for common skin conditions with pregnancy-aware triage logic. MedASR captures voice symptom check-ins with medical-grade transcription accuracy. And Gemini 2.5 Pro handles final patient-facing communication and Visit Prep Summary generation.

Each model does what it's best at. No single model tries to do everything. The orchestration layer — powered by our fine-tuned MedGemma — correlates signals across modalities to catch patterns that any single model would miss. A woman's elevated blood pressure means one thing alone. Combined with her voice report of headaches and blurry vision, it means something urgent. That cross-signal intelligence is what makes MamaGuard more than the sum of its parts."
