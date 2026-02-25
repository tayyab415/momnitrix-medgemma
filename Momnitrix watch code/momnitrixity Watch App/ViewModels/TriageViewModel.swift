import Foundation
import Observation

// Central state machine for the entire app.

enum AppScreen: Sendable {
    case home
    case inputSheet
    case diagnosing
    case results
}

enum RiskLevel: String, Sendable {
    case green, yellow, red, unknown

    var color: String {
        switch self {
        case .green:   return "green"
        case .yellow:  return "yellow"
        case .red:     return "red"
        case .unknown: return "gray"
        }
    }

    var emoji: String {
        switch self {
        case .green:   return "🟢"
        case .yellow:  return "🟡"
        case .red:     return "🔴"
        case .unknown: return "⚪️"
        }
    }
}

@Observable
final class TriageViewModel {

    // Navigation
    var screen: AppScreen = .home

    // HealthKit live values — sensible defaults if Watch has no data
    var heartRate: Double = 75
    var hrv: Double = 40
    var spo2: Double = 97
    var tempC: Double = 36.8

    // Manual entries
    var age: Int = 25
    var gestationalWeeks: Int = 31
    var gravidity: Int = 2
    var parity: Int = 1
    var bmiGroup: String = "overweight"
    var knownConditions: String = ""
    var medications: String = "prenatal_vitamins"
    var systolicBp: Double = 130
    var diastolicBp: Double = 80
    var fastingGlucose: Double = 5.1

    // Symptom flags
    var headache: Bool = false
    var visionChanges: Bool = false
    var decreasedFetalMovement: Bool = false
    var freeText: String = ""

    // Media attachments (from AudioRecorderService / ImagePickerHelper)
    var audioB64: String?
    var woundImageB64: String?
    var skinImageB64: String?

    // SSE + result state
    var events: [SSEEvent] = []
    var finalResponse: TriageFinalResponse?
    var riskLevel: RiskLevel = .unknown
    var isStreaming: Bool = false
    var errorMessage: String?
    var validationErrors: [String] = []
    var backendHealthy: Bool = false

    private let apiClient = TriageAPIClient()

    // MARK: - Derived

    var canSubmit: Bool {
        validationErrors.isEmpty
    }

    var actionItems: [String] {
        finalResponse?.action_items ?? []
    }

    var medgemmaReasons: [String] {
        finalResponse?.medgemma_reasons ?? []
    }

    var patientMessage: String {
        finalResponse?.patient_message ?? ""
    }

    var visitSummary: String {
        finalResponse?.visit_prep_summary ?? ""
    }

    var totalLatencyMs: Double? {
        if let total = finalResponse?.latency_ms?.total {
            return Double(total)
        }
        return nil
    }

    // MARK: - Populate from HealthKit

    func populateFromHealthKit(_ reader: HealthReader) {
        // Only override defaults when HealthKit has real data
        if let hr = reader.heartRate, hr > 0   { heartRate = hr }
        if let h = reader.hrv, h > 0           { hrv = h }
        if reader.spo2Percent > 0              { spo2 = reader.spo2Percent }
        if let t = reader.wristTempC, t > 0    { tempC = t }
        revalidate()
    }

    // MARK: - Validation

    func revalidate() {
        validationErrors = ValidationService.validate(
            age: age,
            systolicBp: systolicBp,
            diastolicBp: diastolicBp,
            fastingGlucose: fastingGlucose,
            tempC: tempC,
            heartRate: heartRate,
            spo2: spo2
        )
    }

    // MARK: - Submit

    func submit() async {
        revalidate()
        guard canSubmit else { return }

        events = []
        finalResponse = nil
        riskLevel = .unknown
        errorMessage = nil
        isStreaming = true
        screen = .diagnosing

        let payload = PayloadComposer.compose(
            age: age,
            gestationalWeeks: gestationalWeeks,
            gravidity: gravidity,
            parity: parity,
            bmiGroup: bmiGroup,
            knownConditions: knownConditions
                .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) },
            medications: medications
                .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) },
            systolicBp: Int(systolicBp),
            diastolicBp: Int(diastolicBp),
            fastingGlucose: fastingGlucose,
            heartRate: Int(heartRate.rounded()),
            spo2Percent: Int(spo2.rounded()),
            tempC: tempC,
            hrv: hrv,
            headache: headache,
            visionChanges: visionChanges,
            decreasedFetalMovement: decreasedFetalMovement,
            freeText: freeText,
            audioB64: audioB64,
            woundImageB64: woundImageB64,
            skinImageB64: skinImageB64
        )

        apiClient.onEvent = { [weak self] event in
            self?.events.append(event)
        }

        apiClient.onFinalResponse = { [weak self] response in
            guard let self else { return }
            self.finalResponse = response
            self.riskLevel = RiskLevel(rawValue: response.risk_level?.lowercased() ?? "") ?? .unknown
            self.isStreaming = false
            self.screen = .results
        }

        apiClient.onError = { [weak self] message in
            self?.errorMessage = message
            self?.isStreaming = false
        }

        apiClient.onComplete = { [weak self] in
            self?.isStreaming = false
        }

        await apiClient.stream(request: payload)
    }

    // MARK: - Health check

    func checkBackendHealth() async {
        backendHealthy = await apiClient.checkHealth()
    }

    // MARK: - Reset

    func reset() {
        events = []
        finalResponse = nil
        riskLevel = .unknown
        isStreaming = false
        errorMessage = nil
        audioB64 = nil
        woundImageB64 = nil
        skinImageB64 = nil
        screen = .home
    }
}
