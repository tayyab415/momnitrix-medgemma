import Foundation

// Mirrors web-app/src/domain/types.ts → TriageRequest

struct TriageRequest: Codable, Sendable {
    var request_id: String
    var patient_context: PatientContext
    var vitals: Vitals
    var inputs: TriageInputs
    var metadata: TriageMetadata
}

struct PatientContext: Codable, Sendable {
    var age_years: Int
    var gestational_weeks: Int
    var known_conditions: [String]
    var medications: [String]
}

struct Vitals: Codable, Sendable {
    var systolic_bp: Int
    var diastolic_bp: Int
    var fasting_glucose: Double
    var hr: Int
    var spo2: Int
    var temp_c: Double
    var hrv: Double
}

struct TriageInputs: Codable, Sendable {
    var headache: Bool
    var vision_changes: Bool
    var decreased_fetal_movement: Bool
    var free_text: String
    var wound_image_b64: String?
    var skin_image_b64: String?
    var audio_b64: String?
}

struct TriageMetadata: Codable, Sendable {
    var source: String
    var composer_mode: String
    var response_composer_mode: String
    var medgemma_output_style: String
    var ui_mode: String
    var simulator: SimulatorMeta
}

struct SimulatorMeta: Codable, Sendable {
    var age_years: Int
    var bmi_group: String
    var gravidity: Int
    var parity: Int
    var temp_input_unit: String
}
