# Frontend Agent Handoff (Momnitrix Watch Prototype)

## Purpose
This document transfers the current frontend context to a new AI agent that will improve UI/UX.

This handoff is for **frontend work only**.

## Non-Negotiable Guardrails
- Do not modify backend code.
- Do not modify any Modal deployment files, endpoints, secrets, or infra settings.
- Do not modify orchestrator/model logic files under `momnitrix/` or `modal_*.py`.
- Do not change request/response schemas expected by backend.
- Do not delete diagnostic visibility in UI (SSE timeline + final JSON must remain available).

If a desired UX change requires backend changes, stop and document the gap instead of editing backend.

## Active Frontend Surface
- `qa_console/index.html`
- `qa_console/styles.css`
- `qa_console/app.js`
- `qa_console/input_contract.json` (contract/ranges reference)
- `qa_console/INPUT_RANGE_BASIS.md` (why ranges were chosen)
- `qa_console/TEST_CHECKLIST.md` (manual QA flows)
- `qa_console/README.md` (local run basics)

Legacy/parallel frontend folders exist (for example `kimi-frontend/`) but current active watch prototype work is in `qa_console/`.

## Current UX Architecture
The UI is now watch-first and mode-first:

1. Header with `Input Values` chip (shows missing required count).
2. Modality chooser (three large cards): Text / Voice / Image.
3. Mode-specific interaction panel (one visible at a time).
4. Large full-width `Diagnose` button.
5. Final results section as single-card carousel (`Previous` / `Next`).
6. SSE pipeline event timeline for debugging/audit.

`Input Values` is a modal sheet that opens only when the user clicks `Input Values`.

## File-by-File: What Each File Does

## `qa_console/index.html`
Defines all UI structure.

Key blocks:
- Top controls and backend URL input.
- Modality card buttons:
  - `#modeTextBtn`
  - `#modeVoiceBtn`
  - `#modeImageBtn`
- Input values modal sheet:
  - `#inputValuesPanel` (hidden by default, `role="dialog"`)
  - `#inputValuesBackdrop`
  - `#closeVitalsBtn`
- Mode content panels:
  - `#textPanel`
  - `#voicePanel`
  - `#imagePanel`
- Main actions:
  - `#submitBtn` (Diagnose)
  - `#clearBtn`
  - `#downloadLogBtn`
- Final triage carousel:
  - `#prevCardBtn`, `#nextCardBtn`, `#cardPosition`
  - `.result-card` entries:
    - Overview
    - Patient Guidance
    - Visit Prep Summary
    - Specialist Signals
    - Clinical Reasons
    - Action Plan
    - Raw JSON
- Event timeline:
  - `#eventTimeline`

## `qa_console/styles.css`
Defines visual system and watch-like layout.

Important behavior/styling zones:
- Narrow center layout (`max-width: 560px`) for watch-prototype proportions.
- Mode cards made larger and icon-forward.
- Diagnose button made prominent and full-width.
- Modal sheet + dim backdrop:
  - `.sheet`
  - `.sheet-backdrop`
  - `body.sheet-open { overflow: hidden; }`
- Result carousel behavior:
  - `.result-card { display: none; }`
  - `.result-card.is-active { display: block; }`

Important past bug to avoid reintroducing:
- `main.layout` must not set a z-index that traps the sheet under backdrop.

## `qa_console/app.js`
All runtime logic.

### State
- `state` tracks:
  - active run logging
  - recording state
  - active mode
  - current result-card index

### Input Rules
- Required numeric gates in `REQUIRED_RULES`:
  - age
  - systolic BP
  - diastolic BP
  - fasting glucose
  - body temp (F input)
  - resting HR
- Optional numeric validation in `OPTIONAL_NUMERIC_RULES`.
- `updateRequiredGate()`:
  - updates missing-count badge
  - applies chip class `needs-input` or `ok`
  - disables Diagnose if required inputs invalid/missing

### Mode Control
- `setActiveMode(mode)` toggles active modality and hint copy.
- Modes:
  - `text` -> typed note
  - `voice` -> upload/record audio
  - `image` -> wound/skin uploads

### Payload Composition
- `composePayloadSkeleton()` builds final request for `/v1/triage/stream`.
- Includes:
  - `patient_context`
  - `vitals`
  - `inputs` (`free_text`, media placeholders, symptoms)
  - `metadata` (ui mode + simulator metadata)
- Temperature entered in degF, converted to `temp_c`.

### Media Handling
- `attachMedia(payload)`:
  - image files -> base64
  - audio upload or recording -> base64
- `convertToWavIfPossible(blob)` attempts WAV conversion for MedASR compatibility.
- Mic path:
  - `startRecording()`
  - `stopRecording()`

### Streaming and Diagnostics
- `submitRequest()` POSTs to `/v1/triage/stream` and consumes SSE.
- SSE parser:
  - `createSseEventParser()`
  - `parseBlock()`
- Timeline rendering:
  - `timelineEntry(eventName, payload)`
- Final response rendering:
  - `setFinalResponse(finalPayload)`
  - `renderPatientGuidance(...)`
  - `renderVisitSummary(...)`
  - `renderSpecialistSummary(...)`
- Diagnostics mapping:
  - `setDiagnosticsFromBreakdown(...)`
  - `setDiagnosticsFromFinal(...)`

### Result Carousel
- `showResultCard(index)` shows one card at a time.
- `shiftResultCard(delta)` handles next/previous.
- Wired to:
  - `#prevCardBtn`
  - `#nextCardBtn`

### Input Modal Sheet
- `openInputValuesPanel()` opens sheet + backdrop.
- `closeInputValuesPanel()` closes sheet + backdrop.
- Closes on:
  - Done button
  - Backdrop click
  - Escape key

### Randomization
- `profileModes` defines:
  - `normal_monitoring`
  - `borderline_risk`
  - `red_flag`
- `fillFromRandomProfile()` samples realistic bounded ranges.

## Data Contract Expectations (Frontend Side)
Reference: `qa_console/input_contract.json`

Core required values expected by current UX and backend flow:
- age
- gestational weeks
- BMI group
- systolic BP
- diastolic BP
- fasting glucose
- heart rate
- body temp

Do not rename outgoing payload keys unless backend is explicitly updated (not in scope).

## Current Render Strategy for Results
- Patient message text from model is parsed into sections when possible:
  - Risk level
  - Clinical reasoning
  - Potential complications
  - Recommended actions
  - Warning signs
- Specialist evidence (image/audio outputs) is surfaced in:
  - `Specialist Signals Used`
  - additional image-evidence lines in patient guidance when relevant
- Raw JSON remains available as one carousel card for debugging/audit.

## What the Next Frontend Agent Should Improve
- Improve visual polish and readability for judges while preserving current functionality.
- Keep watch-friendly interaction density and touch targets.
- Improve carousel UX (optional swipe gestures, better card indicators).
- Improve typography hierarchy and spacing across result cards.
- Improve mode transitions/animations (subtle and performant).
- Keep debugging affordances accessible but less visually dominant.

## What Must Not Break
- Required-field gate behavior for Diagnose.
- Voice upload/record flow.
- Image upload previews and payload attachment.
- SSE event timeline.
- Download run log JSON.
- Final JSON visibility.
- Modal sheet open/close interactions.

## Quick Local Run
From repo root:

```bash
python3 -m http.server 8000 -d qa_console
```

Then open:

```text
http://127.0.0.1:8000
```

## Recommended Frontend-Only Validation After Changes
1. Text-only run succeeds and displays result cards.
2. Voice-upload run shows transcript in specialist signals.
3. Image run (wound or skin) shows specialist evidence.
4. Missing required vitals disables Diagnose and updates badge count.
5. Input Values sheet opens/closes correctly and remains clickable.
6. Result cards navigate correctly with Previous/Next.
7. SSE timeline still updates during run.
8. JSON export still downloads with request/events/final response.

## Handoff Summary
The active prototype frontend is functional, watch-oriented, and tightly coupled to existing backend contract.
The next agent should focus on UI/UX refinement, not architecture or backend logic changes.
