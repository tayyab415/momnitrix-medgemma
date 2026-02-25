import Foundation

struct SSEEvent: Identifiable, Sendable {
    let id = UUID()
    let name: String
    let timestamp: Date
    // Store raw JSON string instead of [String: Any] for Sendable conformance
    let rawJSON: String

    // Parse the raw JSON on demand
    private var payload: [String: Any] {
        guard let data = rawJSON.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ["raw": rawJSON]
        }
        return json
    }

    var displayLabel: String {
        SSEEvent.labels[name] ?? name
    }

    var detail: String? {
        if let latency = payload["latency_ms"] as? Double {
            return "\(Int(latency))ms"
        }
        if let latency = payload["latency_ms"] as? Int {
            return "\(latency)ms"
        }
        if let risk = payload["risk_level"] as? String {
            return risk.uppercased()
        }
        if let err = payload["error"] as? String {
            return "error: \(err)"
        }
        return nil
    }

    nonisolated static let labels: [String: String] = [
        "request.accepted":              "Request Accepted",
        "request.rejected":              "❌ Request Rejected",
        "router.decision":               "Router Decision",
        "router.prompt_plan":            "Prompt Plan",
        "artifact.uploaded":             "Artifact Uploaded",
        "model.started":                 "Specialist Started",
        "model.completed":               "Specialist Completed",
        "model.failed":                  "Specialist Failed",
        "medgemma.started":              "MedGemma Started",
        "medgemma.completed":            "MedGemma Completed",
        "medgemma.delta":                "MedGemma Streaming…",
        "gemini.started":                "Gemini Started",
        "gemini.completed":              "Gemini Completed",
        "gemini.skipped":                "Gemini Skipped",
        "gemini.delta":                  "Gemini Streaming…",
        "diagnostics.inference_breakdown": "Inference Diagnostics",
        "triage.final":                  "✅ Final Response",
        "triage.error":                  "❌ Pipeline Error",
    ]
}
