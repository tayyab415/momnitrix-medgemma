import Foundation

// Mirrors schemas.py:FinalTriageResponse exactly.

struct TriageFinalResponse: Codable, Sendable {
    var request_id: String?
    var trace_id: String?
    var timestamp: String?
    var risk_level: String?
    var policy_floor: String?
    var patient_message: String?
    var visit_prep_summary: String?
    var specialist_outputs: SpecialistOutputs?
    var medgemma_reasons: [String]?
    var action_items: [String]?
    var inference_diagnostics: InferenceDiagnostics?
    var latency_ms: LatencyMs?
    var artifact_refs: [String: String]?
}

struct SpecialistOutputs: Codable, Sendable {
    var transcript: String?
    var wound_scores: WoundScores?
    var skin_top3: [SkinTop3Item]?
}

struct WoundScores: Codable, Sendable {
    var urgency: Double
    var infection_risk: Double
    var erythema: Double
}

struct SkinTop3Item: Codable, Sendable {
    var condition: String
    var score: Double
}

struct InferenceDiagnostics: Codable, Sendable {
    var composer_mode: String?
    var medgemma_engine: String?
    var medgemma_fallback_used: Bool?
    var medgemma_prompt_profile: String?
    var medgemma_timing_breakdown: MedGemmaTiming?
    var latency_split_ms: LatencySplitMs?
    var latency_share_pct: LatencySharePct?
    var field_authorship: [String: String]?
    var router: RouterDiagnostics?
}

struct MedGemmaTiming: Codable, Sendable {
    var cold_start: Bool?
    var gpu_warmup_ms: Double?
    var medgemma_inference_ms: Double?
}

struct LatencySplitMs: Codable, Sendable {
    var medgemma: Int?
    var gemini: Int?
    var llm_total: Int?
}

struct LatencySharePct: Codable, Sendable {
    var medgemma: Double?
    var gemini: Double?
}

struct RouterDiagnostics: Codable, Sendable {
    var intent: String?
    var prompt_strategy: String?
    var planner_source: String?
    var ui_mode: String?
    var selected_specialists: [String]?
}

struct LatencyMs: Codable, Sendable {
    var medgemma: Int?
    var gemini: Int?
    var total: Int?
}
