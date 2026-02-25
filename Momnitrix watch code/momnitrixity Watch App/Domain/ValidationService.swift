import Foundation

struct ValidationService {

    struct Rule {
        let label: String
        let min: Double
        let max: Double
    }

    // Ranges aligned with backend Pydantic schema minimums:
    // hr >= 20, spo2 >= 40, temp_c >= 30
    static let requiredRules: [String: Rule] = [
        "age":            Rule(label: "Age",             min: 13,  max: 50),
        "systolicBp":     Rule(label: "Systolic BP",     min: 80,  max: 180),
        "diastolicBp":    Rule(label: "Diastolic BP",    min: 45,  max: 120),
        "fastingGlucose": Rule(label: "Fasting Glucose", min: 3.0, max: 20.0),
        "tempC":          Rule(label: "Body Temp",       min: 30.0, max: 42.0),
        "heartRate":      Rule(label: "Heart Rate",      min: 20,  max: 200),
        "spo2":           Rule(label: "SpO2",            min: 40,  max: 100),
    ]

    static func validate(
        age: Int?,
        systolicBp: Double?,
        diastolicBp: Double?,
        fastingGlucose: Double?,
        tempC: Double?,
        heartRate: Double?,
        spo2: Double? = nil
    ) -> [String] {
        var errors: [String] = []

        func check(_ value: Double?, key: String) {
            guard let rule = requiredRules[key] else { return }
            guard let v = value else {
                errors.append("Missing: \(rule.label)")
                return
            }
            if v < rule.min || v > rule.max {
                errors.append("\(rule.label) must be \(rule.min)–\(rule.max)")
            }
        }

        check(age.map(Double.init), key: "age")
        check(systolicBp,           key: "systolicBp")
        check(diastolicBp,          key: "diastolicBp")
        check(fastingGlucose,       key: "fastingGlucose")
        check(tempC,                key: "tempC")
        check(heartRate,            key: "heartRate")
        check(spo2,                 key: "spo2")

        if let s = systolicBp, let d = diastolicBp, s <= d {
            errors.append("Systolic BP must be greater than diastolic BP")
        }

        return errors
    }
}
