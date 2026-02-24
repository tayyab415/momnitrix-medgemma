# Frontend TypeScript Migration Notes

## What Changed

### Architecture
- **Before**: Single 1370-line `app.js` with all logic (domain, DOM, adapters) interleaved.
- **After**: 12 TypeScript source files organized into 3 layers:
  - `domain/` (7 files) — pure logic, no DOM access
  - `adapters/` (4 files) — browser-only APIs isolated
  - `ui/` (1 file) — thin DOM orchestrator

### Tooling Added
- `tsconfig.json` — strict TypeScript config
- `package.json` — esbuild (bundler), TypeScript (type checker), Vitest (tests)
- esbuild produces `app.js` as an IIFE bundle consumed by existing `index.html`

### Type Coverage
- 20+ interfaces/types with Swift-mirror comments
- All domain functions are strictly typed (no `any` in domain layer)
- Adapter layer uses `any` only for `webkitAudioContext` browser compat

### Test Suite
- 72 tests across 5 test files
- Tests are pure (no browser, no backend)
- Cover: validation, payload composition, SSE parsing, formatters, randomizer

## What Stayed Stable

| Aspect | Status |
|--------|--------|
| `index.html` | Unchanged |
| `styles.css` | Unchanged |
| DOM IDs | All preserved |
| HTML structure | Unchanged |
| Request payload keys | Identical (verified by tests) |
| SSE event handling | Identical logic |
| Result carousel | Same behavior |
| Modal sheet open/close | Same behavior |
| Required vitals gating | Same logic |
| Audio WAV conversion | Same logic |
| Image preview | Same behavior |
| Run log download | Same format |

## Backend Contract Preservation

The following payload keys are verified unchanged by `payload.test.ts`:
- `request_id`, `patient_context.age_years`, `vitals.systolic_bp`, `vitals.diastolic_bp`
- `vitals.fasting_glucose`, `vitals.hr`, `vitals.temp_c`, `vitals.hrv`, `vitals.spo2`
- `inputs.wound_image_b64`, `inputs.skin_image_b64`, `inputs.audio_b64`
- `inputs.vision_changes`, `inputs.decreased_fetal_movement`
- `metadata.ui_mode`, `metadata.source`, `metadata.composer_mode`

## Known Limitations / Risks

1. **Source map**: esbuild generates `app.js.map` — browser dev tools will show TS source.
2. **Build step required**: Developers must run `npm run build` after editing TS files.
   `npm run watch` provides auto-rebuild during development.
3. **No runtime type validation**: SSE event payloads are cast with `as unknown as T`.
   This matches the original JS runtime behavior (no validation of server responses).
