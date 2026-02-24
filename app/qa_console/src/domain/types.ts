/**
 * Momnitrix QA Console – Domain Types
 *
 * All interfaces are annotated with Swift-mirror comments for future
 * SwiftUI porting.  Keep naming stable across TS and Swift codebases.
 */

// ---------------------------------------------------------------------------
// Modality
// ---------------------------------------------------------------------------

/** The three input modalities supported by the watch prototype. */
// Swift mirror: enum Modality: String, Codable { case text, voice, image }
export type Modality = 'text' | 'voice' | 'image';

// ---------------------------------------------------------------------------
// Request payload
// ---------------------------------------------------------------------------

// Swift mirror: struct PatientContext: Codable { ... }
export interface PatientContext {
    age_years: number;
    gestational_weeks: number;
    known_conditions: string[];
    medications: string[];
}

// Swift mirror: struct Vitals: Codable { ... }
export interface Vitals {
    systolic_bp: number;
    diastolic_bp: number;
    fasting_glucose: number;
    hr: number;
    spo2: number;
    temp_c: number;
    hrv: number;
}

// Swift mirror: struct Inputs: Codable { ... }
export interface Inputs {
    headache: boolean;
    vision_changes: boolean;
    decreased_fetal_movement: boolean;
    free_text: string;
    wound_image_b64: string | null;
    skin_image_b64: string | null;
    audio_b64: string | null;
}

// Swift mirror: struct SimulatorMetadata: Codable { ... }
export interface SimulatorMetadata {
    age_years: number;
    bmi_group: string;
    gravidity: number;
    parity: number;
    temp_input_unit: string;
}

// Swift mirror: struct Metadata: Codable { ... }
export interface Metadata {
    source: string;
    composer_mode: string;
    medgemma_output_style: string;
    ui_mode: Modality;
    simulator: SimulatorMetadata;
}

// Swift mirror: struct TriageRequest: Codable { ... }
export interface TriageRequest {
    request_id: string;
    patient_context: PatientContext;
    vitals: Vitals;
    inputs: Inputs;
    metadata: Metadata;
}

// ---------------------------------------------------------------------------
// SSE events
// ---------------------------------------------------------------------------

// Swift mirror: struct SSEEventEnvelope { let event: String; let payload: Any }
export interface SSEEventEnvelope {
    event: string;
    payload: Record<string, unknown>;
}

// Swift mirror: struct InferenceBreakdownEvent: Codable { ... }
export interface InferenceBreakdownEvent {
    model?: string;
    composer_mode?: string;
    gpu_warmup_ms?: number;
    medgemma_inference_ms?: number;
    latency_ms?: number;
    intent?: string;
    prompt_strategy?: string;
    risk_level?: string;
    reason?: string;
    error?: string;
    timestamp?: string;
}

// ---------------------------------------------------------------------------
// Final triage response
// ---------------------------------------------------------------------------

// Swift mirror: struct WoundScores: Codable { ... }
export interface WoundScores {
    urgency: number;
    infection_risk: number;
    erythema: number;
}

// Swift mirror: struct SkinTop3Item: Codable { ... }
export interface SkinTop3Item {
    condition: string;
    score: number;
}

// Swift mirror: struct SpecialistOutputs: Codable { ... }
export interface SpecialistOutputs {
    transcript?: string;
    wound_scores?: WoundScores;
    skin_top3?: SkinTop3Item[];
    skin_scores?: Record<string, number>;
}

// Swift mirror: struct MedGemmaTimingBreakdown: Codable { ... }
export interface MedGemmaTimingBreakdown {
    cold_start?: boolean;
    gpu_warmup_ms?: number;
    medgemma_inference_ms?: number;
}

// Swift mirror: struct LatencySharePct: Codable { ... }
export interface LatencySharePct {
    medgemma?: number;
    gemini?: number;
}

// Swift mirror: struct InferenceDiagnostics: Codable { ... }
export interface InferenceDiagnostics {
    composer_mode?: string;
    medgemma_timing_breakdown?: MedGemmaTimingBreakdown;
    latency_share_pct?: LatencySharePct;
}

// Swift mirror: struct LatencyMs: Codable { ... }
export interface LatencyMs {
    total?: number;
}

// Swift mirror: struct TriageFinalResponse: Codable { ... }
export interface TriageFinalResponse {
    risk_level?: string;
    policy_floor?: string;
    patient_message?: string;
    visit_prep_summary?: string;
    specialist_outputs?: SpecialistOutputs;
    medgemma_reasons?: string[];
    action_items?: string[];
    inference_diagnostics?: InferenceDiagnostics;
    latency_ms?: LatencyMs;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Swift mirror: struct RequiredRule { let id: String; let label: String; let min: Double; let max: Double }
export interface RequiredRule {
    id: string;
    label: string;
    min: number;
    max: number;
}

// Swift mirror: struct OptionalNumericRule { let id: String; let label: String; let min: Double; let max: Double }
export interface OptionalNumericRule {
    id: string;
    label: string;
    min: number;
    max: number;
}

export interface ValidationResult {
    missing: string[];
    errors: string[];
}

// ---------------------------------------------------------------------------
// Randomizer / profile modes
// ---------------------------------------------------------------------------

export interface ProfileValueRanges {
    age: [number, number];
    gestWeeks: [number, number];
    gravidity: [number, number];
    parity: [number, number];
    systolicBp: [number, number];
    diastolicBp: [number, number];
    fastingGlucose: [number, number];
    tempF: [number, number];
    restingHr: [number, number];
    spo2: [number, number];
    hrv: [number, number];
}

export interface ProfileBooleanProbabilities {
    headache: number;
    visionChanges: number;
    decreasedFetalMovement: number;
}

// Swift mirror: struct ProfileMode: Codable { ... }
export interface ProfileMode {
    name: string;
    weight: number;
    valueRanges: ProfileValueRanges;
    bmiGroup: string[];
    knownConditions: string[];
    medications: string[];
    booleans: ProfileBooleanProbabilities;
    notes: string[];
}

/**
 * Result of random profile generation — all form field values in one object,
 * ready for the DOM layer to apply.
 */
export interface RandomProfileResult {
    age: number;
    gestWeeks: number;
    gravidity: number;
    parity: number;
    bmiGroup: string;
    systolicBp: number;
    diastolicBp: number;
    fastingGlucose: number;
    tempF: number;
    restingHr: number;
    spo2: number;
    hrv: number;
    headache: boolean;
    visionChanges: boolean;
    decreasedFetalMovement: boolean;
    knownConditions: string;
    medications: string;
    freeText: string;
    profileModeName: string;
}

// ---------------------------------------------------------------------------
// Sectioned guidance (parsed from model output)
// ---------------------------------------------------------------------------

// Swift mirror: struct SectionedGuidance { ... }
export interface SectionedGuidance {
    risk_level: string;
    clinical_reasoning: string[];
    potential_complications: string[];
    recommended_actions: string[];
    warning_signs: string[];
}

// ---------------------------------------------------------------------------
// Run log
// ---------------------------------------------------------------------------

export interface RunLogEvent {
    event: string;
    payload: Record<string, unknown>;
}

// Swift mirror: struct RunLog: Codable { ... }
export interface RunLog {
    backend_url: string;
    started_at: string;
    completed_at: string | null;
    duration_ms: number | null;
    request_payload: TriageRequest;
    events: RunLogEvent[];
    final_response: TriageFinalResponse | null;
}

// ---------------------------------------------------------------------------
// Form values (extracted from DOM for pure functions)
// ---------------------------------------------------------------------------

export interface FormValues {
    age: number | null;
    gestWeeks: number | null;
    gravidity: number | null;
    parity: number | null;
    bmiGroup: string;
    knownConditions: string;
    medications: string;
    systolicBp: number | null;
    diastolicBp: number | null;
    fastingGlucose: number | null;
    tempF: number | null;
    restingHr: number | null;
    spo2: number | null;
    hrv: number | null;
    headache: boolean;
    visionChanges: boolean;
    decreasedFetalMovement: boolean;
    freeText: string;
    requestId: string;
    backendUrl: string;
}

/** A map from required-rule id to the numeric value (or null). */
export type RequiredValuesMap = Record<string, number | null>;

/** A map from optional-rule id to the raw string value. */
export type OptionalValuesMap = Record<string, string>;

/** Injectable RNG function: returns a number in [0, 1). */
export type RngFunction = () => number;
