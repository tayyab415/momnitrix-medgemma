# Frontend to SwiftUI Mapping Guide

Reference document for porting the Momnitrix TypeScript frontend to a native SwiftUI app.

## 1. TS Type → Swift Struct Mapping

| TypeScript Interface | Swift Equivalent | Notes |
|---|---|---|
| `TriageRequest` | `struct TriageRequest: Codable` | Top-level request body |
| `PatientContext` | `struct PatientContext: Codable` | Nested in request |
| `Vitals` | `struct Vitals: Codable` | Nested in request |
| `Inputs` | `struct Inputs: Codable` | Nested in request |
| `Metadata` | `struct Metadata: Codable` | Nested in request |
| `SimulatorMetadata` | `struct SimulatorMetadata: Codable` | Nested in metadata |
| `TriageFinalResponse` | `struct TriageFinalResponse: Codable` | SSE final event payload |
| `SpecialistOutputs` | `struct SpecialistOutputs: Codable` | Nested in final response |
| `WoundScores` | `struct WoundScores: Codable` | Nested in specialist |
| `SkinTop3Item` | `struct SkinTop3Item: Codable` | Array element |
| `InferenceBreakdownEvent` | `struct InferenceBreakdownEvent: Codable` | SSE diagnostic event |
| `ValidationResult` | `struct ValidationResult` | Local-only, not Codable |
| `RequiredRule` | `struct RequiredRule` | Validation config |
| `ProfileMode` | `struct ProfileMode` | Randomizer config |
| `RandomProfileResult` | `struct RandomProfileResult` | Randomizer output |
| `SectionedGuidance` | `struct SectionedGuidance` | Parsed model output |
| `RunLog` | `struct RunLog: Codable` | Diagnostic export |
| `FormValues` | `struct FormValues` | UI state extraction |

## 2. TS Enum / String Union → Swift Enum

| TypeScript | Swift |
|---|---|
| `type Modality = 'text' \| 'voice' \| 'image'` | `enum Modality: String, Codable, CaseIterable { case text, voice, image }` |
| `EVENT_LABELS` keys | `enum SSEEventType: String { case requestAccepted = "request.accepted", ... }` |
| BMI group strings | `enum BMIGroup: String, Codable { case underweight, normal, overweight, obese }` |

## 3. SSE Parser → Swift Async Stream

**TS implementation** (`domain/sse.ts`):
- `createSseEventParser()` returns a closure that buffers text chunks
- Splits on `\n\n`, parses `event:` and `data:` lines
- Emits via callback

**Swift equivalent**:
```swift
func parseSSEStream(_ data: URLSession.AsyncBytes) -> AsyncStream<SSEEvent> {
    AsyncStream { continuation in
        Task {
            var buffer = ""
            for try await byte in data {
                buffer.append(Character(UnicodeScalar(byte)))
                // Split on \n\n, parse blocks, yield events
            }
            continuation.finish()
        }
    }
}
```

Key mapping:
- JS `TextDecoder` → Swift `URLSession.AsyncBytes`
- JS callback → Swift `AsyncStream<SSEEvent>.Continuation`
- JS `JSON.parse()` → Swift `JSONDecoder().decode()`

## 4. Validation Rules Mapping

All rules are defined in `domain/constants.ts` and map directly:

```swift
struct RequiredRule {
    let id: String
    let label: String
    let min: Double
    let max: Double
}

let REQUIRED_RULES: [RequiredRule] = [
    .init(id: "age", label: "Age", min: 13, max: 50),
    .init(id: "systolicBp", label: "Systolic BP", min: 80, max: 180),
    // ... identical to TS constants
]
```

The validation logic in `domain/validation.ts` is pure and maps 1:1:
- `evaluateRequiredInputs()` → same function signature in Swift
- BP cross-check (systolic > diastolic) → same logic

## 5. Result Card State Machine

**TS state**: `state.resultCardIndex` (integer, 0-based)

**Transitions**:
- `showResultCard(index)` → clamp to [0, cards.length-1], set active
- `shiftResultCard(+1/-1)` → increment/decrement with bounds

**SwiftUI equivalent**:
```swift
@State private var currentCardIndex = 0
let cards: [ResultCard] = ResultCard.allCases

var body: some View {
    TabView(selection: $currentCardIndex) {
        ForEach(cards.indices, id: \.self) { i in
            cardView(cards[i])
                .tag(i)
        }
    }
    .tabViewStyle(.page)
}
```

## 6. Browser-Only Features Requiring Native Replacement

| Browser API | iOS Replacement |
|---|---|
| `MediaRecorder` + `getUserMedia` | `AVAudioRecorder` / `AVAudioSession` |
| `<input type="file">` for images | `PHPickerViewController` / `.photosPicker()` |
| `<input type="file">` for audio | `UIDocumentPickerViewController` |
| `AudioContext` + WAV encoding | `AVAudioConverter` or direct WAV writing |
| `FileReader.readAsDataURL()` | `Data(contentsOf:).base64EncodedString()` |
| `fetch()` streaming | `URLSession.bytes(from:)` |
| `URL.createObjectURL()` | `UIImage(data:)` / local file URL |
| `document.createElement('a')` download | `UIActivityViewController` share sheet |
| DOM event listeners | SwiftUI `Button`, `.onChange`, `.onSubmit` |
| CSS transitions/animations | SwiftUI `.animation()`, `.transition()` |
| `localStorage` (not currently used) | `UserDefaults` / `@AppStorage` |

## 7. Stepwise SwiftUI Implementation Sequence

### Phase 1: Data Models (1-2 days)
Copy all `domain/types.ts` interfaces → Swift structs with `Codable`.
Copy constants from `domain/constants.ts`.

### Phase 2: Domain Logic (2-3 days)
Port pure functions from:
- `validation.ts` → `ValidationService.swift`
- `payload.ts` → `PayloadComposer.swift`
- `sse.ts` → `SSEParser.swift` (use `AsyncStream`)
- `formatters.ts` → `TriageFormatter.swift` (return `AttributedString` instead of HTML)
- `randomizer.ts` → `ProfileRandomizer.swift`

Port tests from `__tests__/*.test.ts` → XCTest equivalents.

### Phase 3: Networking Layer (1-2 days)
- `URLSession` POST with streaming response
- `SSEParser` consuming `AsyncBytes`
- Error handling and retry

### Phase 4: SwiftUI Views (3-5 days)
- Main view with modality selector (Text/Voice/Image)
- Vitals input form (sheet presentation)
- Result carousel (TabView with page style)
- SSE timeline list
- Diagnose button with required-field gating

### Phase 5: Media Capture (2-3 days)
- Camera/photo picker for wound/skin images
- Audio recording with `AVAudioRecorder`
- WAV conversion if needed

### Phase 6: Polish (1-2 days)
- Animations and transitions
- Accessibility labels
- Dark mode support
- Watch complications (if targeting watchOS)

**Estimated total**: 10-17 developer days, heavily front-loaded by the pure domain logic that ports directly from TS.
