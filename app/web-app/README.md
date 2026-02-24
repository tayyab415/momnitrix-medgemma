# Momnitrix QA Console

Local static frontend for watch-style simulation testing against the Momnitrix orchestrator.

## What it does

- Modality-first home flow (Text / Voice / Image)
- Required-input lock on Diagnose for: age, systolic BP, diastolic BP, fasting glucose, body temp, heart rate
- Manual maternal profile + vitals entry
- Clinical-profile randomizer with bounded realistic ranges
- Wound and skin image upload with previews
- Audio upload and browser mic recording
- Live SSE event timeline for `/v1/triage/stream`
- Human-readable triage output + expandable raw JSON + downloadable run log

## Tech Stack

- **Source**: TypeScript (`src/`)
- **Build**: esbuild (bundles `src/main.ts` → `app.js`)
- **Type check**: `tsc --noEmit` (strict mode)
- **Tests**: Vitest (72 tests across 5 domain module test files)
- **HTML/CSS**: Plain static files (no framework)

## Setup

```bash
cd app/qa_console
npm install
```

## Build

```bash
# One-time build
npm run build

# Watch mode (rebuilds on file change)
npm run watch
```

Both produce `app.js` in the project root, consumed by `index.html`.

## Type Check

```bash
npm run typecheck
```

## Tests

```bash
npm test
```

## Run locally

From the repo root:

```bash
cd app/qa_console
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000
```

Default backend target is the deployed Modal API URL.
You can override the backend URL in the UI.

## Project Structure

```
qa_console/
├── index.html              # Static HTML (unchanged)
├── styles.css              # Static CSS (unchanged)
├── app.js                  # esbuild output (do not edit directly)
├── app.js.bak              # Original JS backup
├── package.json            # Build/test scripts
├── tsconfig.json           # TypeScript config
├── src/
│   ├── main.ts             # Entry point
│   ├── domain/             # Pure logic (no DOM)
│   │   ├── types.ts        # All interfaces/types
│   │   ├── constants.ts    # Centralized constants
│   │   ├── validation.ts   # Input validation
│   │   ├── payload.ts      # Request composition
│   │   ├── sse.ts          # SSE parser
│   │   ├── formatters.ts   # HTML formatters
│   │   └── randomizer.ts   # Clinical profile randomizer
│   ├── adapters/           # Browser-only (isolated)
│   │   ├── base64.ts       # FileReader base64
│   │   ├── audio.ts        # WAV encoding + AudioContext
│   │   ├── media.ts        # File attachment
│   │   └── streaming.ts    # Fetch stream reader
│   ├── ui/
│   │   └── dom.ts          # DOM orchestrator
│   └── __tests__/          # Vitest unit tests
│       ├── validation.test.ts
│       ├── payload.test.ts
│       ├── sse.test.ts
│       ├── formatters.test.ts
│       └── randomizer.test.ts
├── input_contract.json
├── INPUT_RANGE_BASIS.md
└── TEST_CHECKLIST.md
```

## Notes

- Temperature input is in degF and converted client-side to `temp_c`.
- Age is sent as `patient_context.age_years` and mirrored in `metadata.simulator`.
- UI mode is sent to orchestrator as `metadata.ui_mode` (`text`, `voice`, `image`).
- For audio recording, the console attempts to convert browser-recorded media into WAV before upload to improve MedASR compatibility.
- Range rationale and source links are documented in `qa_console/INPUT_RANGE_BASIS.md`.
- Domain modules (`src/domain/`) have zero DOM access and are designed for easy Swift porting.
