# Momnitrix QA Console Test Checklist

## Preconditions

- Modal services deployed and healthy.
- `qa_console` served locally on `http://127.0.0.1:8000`.
- Browser has mic permissions if testing record flow.

## Core flows

1. Open console and verify default backend URL is populated.
2. Submit a vitals-only payload and confirm timeline ends with `triage.final`.
3. Upload wound image only and confirm `model.started/completed` for `medsiglip`.
4. Upload skin image only and confirm `model.started/completed` for `derm`.
5. Upload audio file only and confirm `model.started/completed` for `medasr`.
6. Record audio with mic and submit; verify audio path and final response appear.
7. Upload wound + skin + audio together and verify multimodal parallel event flow.

## Safety logic checks

1. Enter fasting glucose `15.0` at 31 weeks and verify final `risk_level=red`.
2. Enter severe BP (for example `165/112`) and verify escalation behavior.
3. Toggle headache + vision changes and verify red-hard-stop behavior.
4. Toggle decreased fetal movement and verify red-hard-stop behavior.

## Randomizer checks

1. Click randomize at least 20 times.
2. Confirm generated values remain within expected realistic bounds.
3. Confirm profile mode label updates (`normal_monitoring`, `borderline_risk`, `red_flag`).
4. Submit at least one random case from each mode.

## Logging/export checks

1. After a run, click `Download Run Log (JSON)`.
2. Confirm JSON includes:
   - `request_payload`
   - `events`
   - `final_response`
   - `started_at/completed_at/duration_ms`
3. Confirm downloaded log is readable and can be shared for debugging.
