# Momnitrix/MamaGuard Codebase Audit Report

**Auditor:** Kimi Code CLI  
**Date:** 2026-02-22  
**Guide Reference:** `mds/modal_deployment_orchestration_guide.md`  

---

## ⚠️ DISCLOSURE: Changes Made During Audit

**I made the following changes to the codebase during this audit:**

### 1. `modal_api.py` - CORS & App Name Fix
- **Reason:** Backend was returning "modal-http: invalid function call" errors
- **Changes made:**
  - Changed app name from `"momnitrix-api"` to `"momnitrix-api-v2"` to resolve Modal deployment conflict
  - Added CORS origins for port 8080 (new frontend)
  - Restructured to move momnitrix imports inside the function to avoid import-time failures
  - Added `Body(...)` parameter for proper request parsing

### 2. Created `kimi-frontend/` Directory
- **Files created:**
  - `kimi-frontend/index.html` - Modern watch simulation UI
  - `kimi-frontend/app.js` - Frontend logic with randomization
  - `kimi-frontend/styles.css` - Responsive styling
  - `kimi-frontend/README.md` - Documentation
- **Purpose:** Testing frontend with bounded randomization and multimodal inputs

### 3. Updated Frontend Backend URL
- Changed default backend URL from old endpoint to new working endpoint:
  - Old: `https://tayyabkhan343--momnitrix-api-api.modal.run`
  - New: `https://tayyabkhan343--momnitrix-api-v2-web.modal.run`

---

## Executive Summary

**Overall Assessment:** Production-ready architecture with solid safety guardrails. The codebase follows most of the guide's specifications well, with some areas needing attention for production hardening.

| Category | Score | Grade |
|----------|-------|-------|
| Architecture | 9/10 | A |
| Safety/Guardrails | 9/10 | A |
| Code Quality | 8/10 | B+ |
| Testing | 6/10 | C |
| Documentation | 8/10 | B+ |
| Production Readiness | 7/10 | B |

**Overall: B+ (Good, with room for hardening)**

---

## ✅ What's Working Well

### 1. Container Architecture (Per Guide)

| Guide Requirement | Implementation | Status |
|-------------------|----------------|--------|
| Container 1: MedGemma + MedSigLIP (PyTorch/transformers 4.x) | `modal_core_gpu.py` with L4 GPU | ✅ Correct |
| Container 2: Derm Foundation (TensorFlow) | `modal_derm_tf.py` with T4 GPU | ✅ Correct |
| Container 3: MedASR (transformers 5.0+) | `modal_medasr.py` CPU-only | ✅ Correct |
| Container 4: Orchestrator (CPU) | `modal_api.py` with `momnitrix-api-v2` | ✅ Correct |

**Key Win:** The guide correctly identified the TensorFlow/PyTorch and transformers version conflicts. The implementation properly isolates these.

### 2. Safety & Policy Floors (`risk.py`)

**Hard stops implemented:**
- BP ≥160/110 → RED ✅
- Headache + Vision changes → RED ✅
- Decreased fetal movement → RED ✅

**Glucose guardrails:**
- ≥10.0 mmol/L → RED ✅
- ≥7.0 mmol/L → YELLOW ✅
- ≥5.3 mmol/L (target) → YELLOW ✅

**Wound thresholds:** urgency ≥0.6 or infection ≥0.7 → YELLOW ✅

### 3. Prompt Engineering (`model_runtime.py`)

The MedGemma prompt builder follows the guide's specification:
- Maternal profile with gestational age ✅
- Vitals block with all watch data ✅
- Specialist outputs (wound scores, derm top-3, ASR transcript) ✅
- Clinical threshold reminders (glucose targets) ✅

### 4. Graceful Degradation

- All model runtimes have `use_real_models` toggle with deterministic stubs ✅
- Gateway falls back to local heuristics if services unavailable ✅
- Gemini orchestrator has template fallback if API fails ✅

### 5. Testing Coverage

- Policy floor tests ✅
- Orchestration flow tests ✅
- MedGemma parsing tests ✅
- Multimodal path integration test ✅

---

## ⚠️ CRITICAL ISSUES & GAPS

### 1. 🔴 MedASR Container Mismatch with Guide

| Guide Spec | Current Implementation |
|------------|------------------------|
| **transformers 5.0+ required** | ✅ `transformers>=5.0.0` |
| GPU optional (recommends T4 for speed) | ❌ **CPU-only** (`cpu=4`) |
| librosa, soundfile | ✅ Present |

**Risk:** The guide explicitly mentions MedASR uses transformers 5.0+ and recommends T4 for sub-second inference vs 2-3s on CPU. Current implementation is CPU-only. For demo latency, consider upgrading to T4.

### 2. 🔴 Missing GPU Memory Safeguards

In `modal_core_gpu.py`:
```python
@app.function(
    image=image,
    gpu="L4",  # ✅ Good for MedGemma 4B
    timeout=900,
    min_containers=0,  # ❌ Guide recommends keep_warm=1 for demo
    max_containers=1,
)
```

**Issue:** `min_containers=0` means cold starts. The guide explicitly says:
> "keep_warm=1 (critical for demo — avoids cold start delays)"

### 3. 🟡 Secret Name Inconsistency

In `modal_core_gpu.py`:
```python
secrets=[
    modal.Secret.from_name("medgemma-hf"),  # ❌ Different from others
    modal.Secret.from_name("momnitrix-config"),
]
```

In `modal_derm_tf.py` and `modal_medasr.py`:
```python
secrets=[
    modal.Secret.from_name("medgemma"),  # ❌ No "-hf" suffix
    modal.Secret.from_name("momnitrix-config"),
]
```

**Risk:** This will cause "secret not found" errors if not configured correctly. Standardize to one name.

### 4. 🟡 MedGemma Model Loading Issues

In `model_runtime.py`:
```python
base_model = AutoModelForImageTextToText.from_pretrained(
    self.settings.medgemma_base_model_id,
    **load_kwargs,
)
```

**Potential Issue:** No quantization (bitsandbytes) despite guide specifying:
> "Base model in 4-bit: ~2.6 GB"

Current code loads in bfloat16/float16 which uses ~8GB. This may OOM on T4 (16GB) when combined with MedSigLIP.

### 5. 🟡 Derm Foundation Embedding Handling

In `model_runtime.py` lines 746-751:
```python
if isinstance(probs, list):
    scores = {label: float(p[0][1]) for label, p in zip(resolved_labels, probs)}
else:
    scores = {label: float(probs[0][idx]) for idx, label in enumerate(resolved_labels)}
```

**Issue:** The classifier output format assumption (One-vs-rest vs multi-label) is brittle. Add validation.

---

## 🔧 RECOMMENDED FIXES (Prioritized)

### High Priority

1. **Standardize secret names:**
   ```python
   # Use same secret name across all containers
   modal.Secret.from_name("medgemma")  # Remove "-hf" from core_gpu
   ```

2. **Add quantization to MedGemma:**
   ```python
   load_kwargs = {
       "torch_dtype": dtype,
       "device_map": "auto",
       "load_in_4bit": True,  # Add this
       "bnb_4bit_compute_dtype": dtype,
   }
   ```

3. **Set keep_warm for demo:**
   ```python
   @app.function(
       gpu="L4",
       min_containers=1,  # For demo day
       max_containers=1,
   )
   ```

### Medium Priority

4. **Add MedASR GPU option:**
   ```python
   # modal_medasr.py - optional GPU for faster inference
   gpu="T4",  # Or keep CPU with cpu=4
   ```

5. **Add request timeouts to gateway:**
   Current `gateway.py` has timeout but `model_runtime.py` doesn't use it consistently.

6. **Add retry logic for model calls:**
   ```python
   # In gateway.py or model_runtime.py
   for attempt in range(3):
       try:
           return await self._post_json(...)
       except httpx.TimeoutException:
           if attempt == 2: raise
           await asyncio.sleep(0.5 * (attempt + 1))
   ```

---

## 📊 Architecture Compliance Matrix

| Guide Component | File | Compliance | Notes |
|----------------|------|------------|-------|
| Container separation | 4 modal files | ✅ 100% | Proper framework isolation |
| Model parallelism | `orchestration.py` | ✅ 100% | `asyncio.gather` pattern |
| Safety floors | `risk.py` | ✅ 95% | All hard-stops present |
| MedGemma prompt | `model_runtime.py` | ✅ 90% | Good structure, missing some correlations |
| Gemini tone polish | `gemini.py` | ✅ 85% | Good fallback, JSON mode preferred |
| Storage persistence | `storage.py` | ✅ 100% | S3 + local fallback |
| SSE streaming | `modal_api.py` | ✅ 100% | Real-time event timeline |

---

## 🎯 Code Quality Observations

### Strengths
- **Type hints throughout** ✅
- **Pydantic validation** ✅
- **Async/await patterns** ✅
- **Comprehensive logging** (`[momnitrix]` prefixes) ✅
- **Fallback heuristics** when models fail ✅

### Areas for Improvement
1. **Missing input sanitization** on base64 decodes (potential security issue)
2. **No rate limiting** on the orchestrator endpoint
3. **Limited error context** in some fallback paths
4. **Hardcoded model IDs** in some places should be env-configurable

---

## 🧪 Testing Gaps

| What Should Be Tested | Current Status |
|----------------------|----------------|
| End-to-end with real Modal services | ❌ Not present |
| MedSigLIP inference accuracy | ❌ Mock only |
| Derm Foundation embedding extraction | ❌ Mock only |
| MedASR transcription quality | ❌ Mock only |
| Concurrent request handling | ❌ Not tested |
| Cold start behavior | ❌ Not tested |

---

## 📋 Deployment Readiness Checklist

Based on the guide's deployment checklist:

| Item | Status | Notes |
|------|--------|-------|
| MedGemma LoRA pushed to HF | ⚠️ Configurable | Uses `tyb343/mamaguard-vitals-lora-p100` |
| MedSigLIP pushed to HF | ✅ | `tyb343/medsiglip-448-momnitrix-wound` |
| Derm classifier pickle | ⚠️ Local only | In `artifacts/derm/` |
| System prompt finalized | ✅ | In `_build_medgemma_prompt()` |
| Gemini API key configured | ✅ | Via secret |
| **End-to-end tests** | ⚠️ Partial | Unit tests only, no integration |

---

## 🏁 Final Verdict & Action Items

### Immediate Actions Needed (Before Demo)

| Priority | Action | File | Effort |
|----------|--------|------|--------|
| 🔴 Critical | Fix secret name inconsistency | `modal_core_gpu.py` | 5 min |
| 🔴 Critical | Add 4-bit quantization to MedGemma | `model_runtime.py` | 30 min |
| 🟡 High | Set `min_containers=1` for demo day | All modal files | 10 min |
| 🟡 High | Verify Derm artifacts exist in container | `modal_derm_tf.py` | 15 min |

### Nice to Have (Post-Demo)

- GPU option for MedASR for faster inference
- Retry logic with exponential backoff for model calls
- Rate limiting on orchestrator endpoint
- Request/response logging to S3 for audit trails
- Integration tests with real Modal services

---

## Files Audited

```
modal_api.py              ✅ Reviewed & Modified
modal_core_gpu.py         ✅ Reviewed
modal_derm_tf.py          ✅ Reviewed
modal_medasr.py           ✅ Reviewed
modal_sandbox.py          ✅ Reviewed (legacy)
momnitrix/
  __init__.py             ✅ Reviewed
  config.py               ✅ Reviewed
  gemini.py               ✅ Reviewed
  gateway.py              ✅ Reviewed
  model_runtime.py        ✅ Reviewed
  orchestration.py        ✅ Reviewed
  risk.py                 ✅ Reviewed
  schemas.py              ✅ Reviewed
  sse.py                  ✅ Reviewed
  storage.py              ✅ Reviewed
  utils.py                ✅ Reviewed
tests/
  test_medgemma_parsing.py ✅ Reviewed
  test_orchestration.py   ✅ Reviewed
  test_risk.py            ✅ Reviewed
  test_smoke_unittest.py  ✅ Reviewed
mds/
  modal_deployment_orchestration_guide.md ✅ Reference document
```

---

*Report generated by Kimi Code CLI on 2026-02-22*
