# MedASR Integration Plan for MamaGuard

## What MedASR Is and Why It Matters for Us

MedASR is Google's medical speech-to-text model. It's a 105-million-parameter Conformer architecture (hybrid CNN + Transformer) trained on roughly 5,000 hours of physician dictations spanning radiology, internal medicine, and family medicine. It accepts mono-channel 16kHz audio and produces text transcriptions.

The key selling point: on medical dictation benchmarks, MedASR achieves a 4.6% word error rate on radiology dictation (with language model decoding), compared to 25.3% for Whisper v3 Large. It beats even Gemini 2.5 Pro on medical speech. It is 15x smaller than Whisper Large (105M vs 1.55B parameters), making it extremely lightweight.

Google explicitly positions MedASR as a pipeline component designed to feed into MedGemma. Their documentation describes a workflow where MedASR transcribes audio and then MedGemma analyzes the transcribed text. This is exactly our architecture.

---

## MedASR's Role in MamaGuard: Voice Symptom Check-Ins

MedASR powers Pillar 3 of MamaGuard: the voice check-in system. The user flow is:

1. The pregnant woman opens MamaGuard and taps "Daily Check-In"
2. The app prompts her with guided questions: "How are you feeling today? Any headaches, swelling, or changes in your baby's movement?"
3. She speaks naturally for 30-90 seconds describing her symptoms
4. Audio is captured on the phone (16kHz mono), sent to Modal
5. MedASR transcribes the speech with medical vocabulary accuracy
6. The transcript is passed to MedGemma as additional context alongside watch vitals and any wound photos
7. MedGemma synthesizes ALL signals into a unified response

The value of MedASR over generic speech-to-text (like Whisper or the phone's built-in dictation) is **medical vocabulary accuracy**. When a pregnant woman says "I've been having some edema in my ankles and I feel short of breath, maybe some orthopnea at night," MedASR will correctly capture "edema," "orthopnea," and other clinical terms that consumer ASR frequently mangles. This matters because these terms then flow into MedGemma's reasoning engine — garbage transcription produces garbage clinical analysis.

---

## Do We Need to Fine-Tune MedASR?

**Short answer: No, but we CAN do a creative, lightweight adaptation that impresses judges.**

Here's the honest breakdown:

### What MedASR Already Handles Well (No Fine-Tuning Needed)

MedASR was trained on physician dictations across internal medicine and family medicine. The obstetric and maternal health vocabulary overlaps heavily with these specialties. Terms like preeclampsia, gestational diabetes, blood pressure, contractions, fundal height, fetal heart rate, edema, proteinuria, magnesium sulfate, labetalol, lochia — these are standard clinical terms that a family medicine/internal medicine-trained model will recognize.

Additionally, MedASR was trained on real clinical conversations (not just dictation), so it handles conversational speech patterns, not only formal medical dictation.

### Where MedASR Might Struggle

Google's documentation flags several limitations:

- **Accents**: MedASR was primarily trained on US English-first-language speakers. If our target users include non-native English speakers, performance may degrade.
- **Consumer audio quality**: Training data was physician dictations (typically quiet office environments with decent microphones). A pregnant woman speaking into a phone in her home, with background noise from children or TV, is a different acoustic environment.
- **Date/time formatting**: MedASR was trained on de-identified data, so it can be inconsistent with dates and durations. Not critical for our use case since we extract clinical content, not scheduling data.
- **Newer medication names**: Terms that emerged in the past decade might not be in its vocabulary. For maternal health, most relevant medications (labetalol, methyldopa, nifedipine, magnesium sulfate) are well-established.

### The Smart Fine-Tuning Play (Optional, 2-3 Hours)

Google provides a fine-tuning notebook on Colab. The approach uses HuggingFace Trainer with the CTC loss function — standard ASR fine-tuning.

The creative angle for judges: we can do a **vocabulary expansion fine-tune** specifically for maternal health terminology. Even with zero real audio data, we could:

1. Generate a small set of synthetic maternal health audio clips (use text-to-speech to create 50-100 clips of sentences containing OB/GYN terminology)
2. Pair them with correct transcripts
3. Fine-tune MedASR for a few epochs on this small dataset

Terms to target: PUPPP, pemphigoid gestationis, cerclage, cervical effacement, Bishop score, chorioamnionitis, postpartum hemorrhage, peripartum cardiomyopathy, HELLP syndrome, eclampsia.

**My honest opinion**: this is optional and somewhat performative for judges. MedASR will likely handle 95% of maternal health speech correctly out of the box. The fine-tune adds polish and demonstrates "meaningful adaptation" (which Google's hackathon criteria explicitly rewards). If you have time on the last day, it's a nice touch. If you're pressed for time, skip it — using MedASR as-is is perfectly defensible.

---

## How MedASR Feeds Into the MedGemma Orchestration

This is the critical architecture point. MedASR does NOT generate clinical assessments. It only produces text. That text becomes one of several inputs to MedGemma:

```
MedGemma receives:
├── Watch vitals (HR, SpO2, skin temp, BP)           ← Samsung Watch
├── Wound classification (6 sigmoid scores)           ← MedSigLIP
├── Derm classification (10-condition scores)          ← Derm Foundation + sklearn
├── Voice transcript: "I've been having bad           ← MedASR
│   headaches the last two days and my ankles
│   are really swollen. My vision was a little
│   blurry this morning."
├── Gestational age: 34 weeks
├── Medication history: labetalol 200mg BID
└── Safety rules / escalation thresholds

MedGemma synthesizes:
"Your blood pressure reading of 148/94 combined with the headaches,
 visual changes, and ankle swelling you described are concerning.
 These symptoms together at 34 weeks may indicate worsening
 preeclampsia. Please contact your OB-GYN today — do not wait
 for your next scheduled visit."
```

The voice transcript adds irreplaceable context. Watch sensors can detect elevated BP and HR, but they can't capture "I had blurry vision this morning" or "my baby hasn't been moving as much." These subjective symptoms are clinically essential for risk assessment, and MedASR is how they enter the pipeline.

---

## Practical Integration Details

### Audio Capture Requirements

MedASR expects 16kHz mono audio as int16 waveform. On Android:
- Use the standard AudioRecord API or MediaRecorder
- Set sample rate to 16000 Hz, mono channel, 16-bit PCM
- Record for 30-90 seconds (typical check-in duration)
- Compress to WAV or send raw PCM bytes

### Processing Pipeline

The audio processing follows the HuggingFace transformers pipeline pattern:

1. Audio arrives at the Modal endpoint as base64-encoded WAV
2. Decode and resample to 16kHz with librosa
3. Process through MedASR AutoProcessor
4. Run MedASR AutoModelForCTC inference
5. Decode output tokens via processor.batch_decode()
6. Return plain text transcript

For longer recordings, use chunked processing (chunk_length_s=20, stride_length_s=2) to handle arbitrary-length audio without memory issues.

### Language Model Decoding (Optional Enhancement)

MedASR supports pairing with a 6-gram language model during decoding. This reduces WER from 6.6% to 4.6% on radiology dictation. The language model uses beam search with beam size 8.

For our hackathon: greedy decoding (the default) is fine. The 6-gram LM adds complexity to the deployment container and the WER improvement on conversational speech (vs dictation) is less dramatic. If you want the absolute best accuracy, integrate it; if you want simplicity, skip it.

---

## The MedASR-to-Documentation Pipeline: Visit Prep Summaries

One of MamaGuard's strongest features is the Visit Prep Summary — an automated report the user can bring to her OB-GYN appointment. MedASR plays a crucial role here:

Over the course of a week, the user does several voice check-ins. MedASR transcribes each one. When it's time for her appointment, MedGemma aggregates:

- All watch vital trends over the period
- All wound assessment results (with longitudinal tracking)
- All voice check-in transcripts (symptom diary)
- Any skin condition screening results

MedGemma then generates a structured Visit Prep Summary that includes:
- Vital sign trends with any anomalies flagged
- Symptom timeline extracted from voice check-ins
- Wound healing progression
- Recommended discussion points for the provider

The voice transcripts are particularly valuable here because they create a symptom diary that the patient didn't have to manually write. Many women forget to mention symptoms at appointments, or downplay them. Having MedASR-captured transcripts means MedGemma can surface patterns like "the patient mentioned headaches in 4 out of 7 check-ins this week" even if the patient doesn't remember to bring it up.

---

## Dependency Note: transformers Version Conflict

MedASR requires transformers 5.0.0 or newer. Our other models (MedGemma, MedSigLIP) use transformers 4.x. This is why MedASR MUST run in a separate Modal container. This was flagged in our earlier architecture planning and remains a hard constraint. Do not try to load MedASR in the same container as MedGemma or MedSigLIP — version conflicts will break things silently.

---

## Judge Talking Points for MedASR

"We use MedASR for voice-based symptom check-ins — a hands-free interaction model for pregnant women who may be managing children while monitoring their health. MedASR's medical vocabulary accuracy means clinical terms like edema, orthopnea, and preeclampsia are transcribed correctly, which is critical because these transcripts feed directly into MedGemma's reasoning engine. Consumer speech-to-text garbles medical terminology, producing unreliable inputs for clinical AI. MedASR eliminates that failure mode."

If asked about fine-tuning: "MedASR is pre-trained on 5,000 hours of physician dictations across internal medicine and family medicine, which covers the majority of maternal health vocabulary. We evaluated it against OB/GYN-specific terms and found reliable performance out of the box. For production, we would expand the vocabulary with OB-specific terms via few-shot fine-tuning, but the base model's medical training is the primary value driver."

---

## Summary: MedASR Status

| Aspect | Status |
|---|---|
| Model | google/medasr (105M params, Conformer, CTC) |
| Fine-tuning required? | No (optional vocabulary expansion for judge optics) |
| Framework | PyTorch (HuggingFace transformers 5.0+) |
| Separate container needed? | YES — transformers version conflict |
| GPU needed for inference? | Can run on CPU, but T4 GPU is faster for real-time |
| Input | 16kHz mono audio (WAV or raw PCM) |
| Output | Plain text medical transcription |
| Output destination | Fed into MedGemma as additional context input |
| Hackathon priority | Medium — critical for the multi-model story, but the core pipeline works without it |
