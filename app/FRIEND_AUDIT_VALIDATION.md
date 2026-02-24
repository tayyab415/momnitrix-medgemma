# Friend's Audit Validation & Related Issues Analysis

**Analysis Date:** 2026-02-22  
**Validator:** Kimi Code CLI  
**Scope:** Validate friend's findings + discover related issues

---

## Friend's Finding #1: No Startup Health Verification

### Validation: ✅ **CONFIRMED - CRITICAL ISSUE**

**Location:** `modal_api.py` lines 78-90

```python
@api.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "core_gpu_configured": bool(settings.core_gpu_base_url),  # Just checks string
        ...
    }
```

**The Problem:** Only checks if environment variables are non-empty strings. A service could be:
- Crashed
- In cold-start (not ready yet)
- Have wrong secrets
- Network unreachable

And the health check would still return ✓ configured.

### Related Issues Discovered

#### 1a. **No Timeout Validation on Gateway Calls**
**Location:** `momnitrix/gateway.py` lines 19-34

The `_post_json` method has timeout parameters, but the actual health verification doesn't test connectivity:

```python
# Current: No validation that URL is reachable
async def _post_json(self, base_url: str, path: str, ...):
    url = f"{base_url.rstrip('/')}{path}"
    # If base_url is wrong, this fails on FIRST REAL REQUEST, not at startup
```

#### 1b. **Silent Fallback Masks Service Failures**
**Location:** `momnitrix/gateway.py` lines 41-57, 59-84, 86-96

All three specialist model calls silently fall back to deterministic stubs if `base_url` is falsy:

```python
async def medsiglip_infer(self, image_b64: str) -> dict[str, float]:
    if self._settings.core_gpu_base_url:  # Only checks if string exists
        # ... call service
    # SILENT FALLBACK - user doesn't know they're getting fake data
    return {
        "healing_status": self._hash_unit_interval(image_b64, "heal"),
        ...  # Fake deterministic scores!
    }
```

**Risk:** During demo, if a backend service is down, the system continues with fake data and the user never knows.

#### 1c. **No Circuit Breaker Pattern**
If a service fails once, every subsequent request still attempts to call it (with full timeout) before falling back. No "fast fail" for known-down services.

---

## Friend's Finding #2: MedGemma Response Parsing Fragility

### Validation: ✅ **CONFIRMED - HIGH SEVERITY**

**Location:** `momnitrix/model_runtime.py` lines 604-661

**Evidence of Model Output Instability:**

The sanitization code reveals multiple known failure modes:

| Sanitization Rule | What It Reveals | Line |
|-------------------|-----------------|------|
| `re.sub(r"<unused\d+>", "", ...)` | Model leaks special tokens | 150, 308 |
| Strip duplicate `model\nRISK LEVEL:` | Model generates duplicate responses | 310-316 |
| Filter `12.0/8.0 mmHg` | Hallucinates impossible BP as decimals | 169-170 |
| Filter `7.0 mg/dL glucose` | Wrong units for pregnancy context | 173-175 |
| Strip `<start_of_turn>` / `<end_of_turn>` | Chat template tokens leak | 149, 304 |
| `no_repeat_ngram_size=4` | Model prone to repetition loops | 634 |

### Related Issues Discovered

#### 2a. **max_new_tokens=192 is Critically Tight**
**Location:** `model_runtime.py` line 631

A complete JSON response with:
- `risk_level`
- `clinical_reasoning` (3-4 sentences)
- `patient_message` (empathetic, ~100 words)
- `action_items` (3-4 items)
- `flags_for_provider`

**Easily exceeds 192 tokens.** Typical complete response: **300-400 tokens**.

**Evidence:** The parsing has 3 fallback layers because truncated JSON is common.

#### 2b. **No Parse Layer Metrics**
**Location:** `model_runtime.py` lines 647, 652, 656, 660

The code prints which layer succeeded:
- `[momnitrix] medgemma_parse_mode=json` ✅
- `[momnitrix] medgemma_parse_mode=sectioned_text` ⚠️
- `[momnitrix] medgemma_parse_failed` ❌
- `[momnitrix] medgemma_fallback=heuristic` ❌

**But there's no aggregation.** You can't tell during testing "Layer 1 succeeded 40% of the time."

#### 2c. **Truncated JSON is Not Gracefully Handled**
If the model generates:
```json
{
  "risk_level": "yellow",
  "clinical_reasoning": "Blood pressure is elevated at 140/90, which is above target for pregnancy. Additionally, the patient reports headache and vision changes which are
```

(Hits token limit mid-sentence)

The `_extract_json_candidate` will find the opening `{` but the JSON will be invalid. The system falls back to heuristic instead of trying to recover partial reasoning.

#### 2d. **No Temperature Sampling Control**
`do_sample=False` (line 632) is good for determinism, but with tight token limits, the model may be forced to cut off mid-thought.

---

## Friend's Finding #3: Derm Artifacts Path Works by Coincidence

### Validation: ✅ **CONFIRMED - MAINTENANCE RISK**

**Location:** 
- `modal_derm_tf.py` line 28: `.add_local_dir("artifacts/derm", remote_path="/root/artifacts/derm")`
- `config.py` line 92: `"artifacts/derm/derm_classifier.pkl"` (relative)

**Why it works:**
- Modal's container CWD is `/root`
- `open("artifacts/derm/...")` → `/root/artifacts/derm/...` ✅

**Why it's fragile:**
- Depends on Modal's undocumented CWD default
- Local testing may fail (different CWD)
- If Modal changes CWD, production breaks

### Related Issues Discovered

#### 3a. **Same Issue in Storage Layer**
**Location:** `momnitrix/storage.py` lines 21-25

```python
self._root = Path(settings.local_storage_dir)  # Default: ".momnitrix_local_store"
(self._root / "artifacts").mkdir(parents=True, exist_ok=True)
```

Relative path `.momnitrix_local_store` depends on CWD. In Modal container, this resolves to `/root/.momnitrix_local_store`.

#### 3b. **No Path Existence Validation**
**Location:** `model_runtime.py` lines 683-689

```python
scaler_path = Path(self.settings.derm_scaler_path)
if scaler_path.exists():  # Silent skip if missing!
    with open(scaler_path, "rb") as fp:
        self._scaler = pickle.load(fp)
```

If the scaler file is missing, the code silently continues without it. No warning that a configured component is missing.

#### 3c. **Local Storage Not Persisted Across Container Restarts**
Files stored in `/root/.momnitrix_local_store` are ephemeral. If the container restarts, previous request logs are lost (unless S3 is configured).

---

## 🔴 Additional Critical Issues (Not in Friend's Audit)

### Issue #4: **Base64 Decoding Uses `validate=False`**
**Location:** 
- `momnitrix/storage.py` line 40
- `momnitrix/model_runtime.py` line 29

```python
return base64.b64decode(value, validate=False)
```

**Risk:** 
- Accepts malformed base64
- Potential security issue if validation depends on structure
- Harder to debug client issues (accepts garbage, fails later)

### Issue #5: **No Input Size Limits on Base64 Payloads**
**Location:** `momnitrix/storage.py` lines 46-71

```python
async def store_blob_b64(self, request_id: str, name: str, data_b64: str, ...):
    content = self._clean_base64(data_b64)  # No size check!
```

**Risk:** 
- Client could upload 100MB+ image → OOM
- No protection against DoS via large payloads

### Issue #6: **Async Task Cancellation Risk**
**Location:** `momnitrix/orchestration.py` lines 120-140

```python
tasks: list[asyncio.Task[tuple[str, Any, str | None]]] = []

if request.inputs.wound_image_b64:
    tasks.append(asyncio.create_task(_run_named("medsiglip", ...)))

for task in asyncio.as_completed(tasks):
    name, result, error = await task
```

**Risk:** If the client disconnects mid-stream, tasks continue running (wasting GPU time). No cancellation propagation.

### Issue #7: **S3 Client Initialization Failure is Silent**
**Location:** `momnitrix/storage.py` lines 28-34

```python
self._s3 = None
if settings.s3_bucket:
    try:
        import boto3
        self._s3 = boto3.client("s3", ...)
    except Exception:
        self._s3 = None  # Silent! No log of why S3 failed
```

If S3 credentials are wrong, the system silently falls back to local storage. User thinks data is persisted to S3 but it's not.

### Issue #8: **Request ID Collision Risk**
**Location:** `momnitrix/orchestration.py` lines 90-91

```python
request_id = request.request_id or str(uuid4())
trace_id = str(uuid4())
```

If client provides `request_id`, there's no validation of uniqueness. A duplicate request_id could:
- Overwrite stored artifacts
- Confuse the `GET /v1/triage/{request_id}` endpoint

### Issue #9: **Timeout Mismatch Between Layers**

| Layer | Timeout | Issue |
|-------|---------|-------|
| Modal function | 900s (core GPU) | Good for cold start |
| Gateway HTTP call | 120s (default) or 300s (medgemma) | Good |
| Orchestrator SSE | 600s | Good |
| **Queue wait** | **0.75s** | ⚠️ Short for slow generation |

**Location:** `modal_api.py` line 127
```python
event_name, envelope = await asyncio.wait_for(queue.get(), timeout=0.75)
```

**Risk:** If MedGemma takes >0.75s between tokens, the SSE sends a keep-alive. This is fine functionally, but during heavy load, could cause client timeouts.

### Issue #10: **No Retry Logic for Transient Failures**
**Location:** `momnitrix/gateway.py` lines 19-34

```python
async with httpx.AsyncClient(...) as client:
    response = await client.post(url, json=payload)
    response.raise_for_status()
```

**Risk:** A transient 502/503 from a Modal service (common during scale-up) immediately fails the request. No retry with backoff.

---

## 📊 Severity Matrix

| Issue | Severity | Likelihood | Demo Impact | Fix Complexity |
|-------|----------|------------|-------------|----------------|
| #1: Health check doesn't verify reachability | 🔴 High | High | **Silent failures** | Low |
| #2: MedGemma parsing fragility | 🔴 High | High | **Template responses** | Medium |
| #3: Derm path by coincidence | 🟡 Medium | Low | Breaks if Modal changes | Low |
| #4: Base64 validate=False | 🟡 Medium | Medium | Security/maintenance | Low |
| #5: No input size limits | 🔴 High | Medium | **DoS/OOM** | Low |
| #6: Task cancellation risk | 🟡 Medium | Medium | Wasted GPU $ | Medium |
| #7: Silent S3 failures | 🔴 High | Medium | **Data loss** | Low |
| #8: Request ID collision | 🟡 Medium | Low | Data corruption | Low |
| #9: Queue timeout 0.75s | 🟢 Low | Low | Minor | Low |
| #10: No retry logic | 🟡 Medium | Medium | Transient failures | Medium |

---

## 🎯 Recommended Fixes (Prioritized)

### Before Demo Day (Critical)

1. **Fix #1:** Add actual HTTP health pings to downstream services
2. **Fix #2:** Increase `max_new_tokens` from 192 to 384
3. **Fix #5:** Add base64 payload size limits (~10MB for images, ~5MB for audio)
4. **Fix #7:** Log S3 initialization failures prominently

### After Demo Day (Important)

5. **Fix #3:** Use absolute paths for Derm artifacts
6. **Fix #4:** Use `validate=True` for base64 (with proper error handling)
7. **Fix #10:** Add retry logic with exponential backoff
8. **Fix #6:** Add task cancellation on client disconnect

### Nice to Have

9. **Fix #8:** Add request_id uniqueness validation
10. **Fix #9:** Monitor if 0.75s queue timeout causes issues

---

## Conclusion

**Friend's audit is ACCURATE and VALUABLE.** All three findings are real issues with significant demo-day impact.

**My additional findings** reveal:
- 3 more critical issues (input size, S3 silent failure, base64 validation)
- 4 medium-priority maintenance risks
- Several defensive coding gaps

**Overall Assessment:** The codebase is functionally solid but lacks defensive programming for edge cases and production hardening.
