import Foundation

struct PayloadComposer {

    static func compose(
        age: Int,
        gestationalWeeks: Int,
        gravidity: Int,
        parity: Int,
        bmiGroup: String,
        knownConditions: [String],
        medications: [String],
        systolicBp: Int,
        diastolicBp: Int,
        fastingGlucose: Double,
        heartRate: Int,
        spo2Percent: Int,
        tempC: Double,
        hrv: Double,
        headache: Bool,
        visionChanges: Bool,
        decreasedFetalMovement: Bool,
        freeText: String,
        audioB64: String? = nil,
        woundImageB64: String? = nil,
        skinImageB64: String? = nil
    ) -> TriageRequest {

        let requestId = "watch-\(Int(Date().timeIntervalSince1970))-\(Int.random(in: 100...999))"

        // Auto-derive uiMode from attached media
        let uiMode: String = {
            if audioB64 != nil { return "voice" }
            if woundImageB64 != nil || skinImageB64 != nil { return "image" }
            return "text"
        }()

        let profileSummary = """
        Patient profile:
        - Age: \(age) years
        - Obstetric history: G\(gravidity)P\(parity)
        - Gestational age: \(gestationalWeeks) weeks
        - BMI group: \(bmiGroup)
        """

        let fullNote = freeText.isEmpty
            ? profileSummary
            : profileSummary + "\n\nClinical note:\n" + freeText

        return TriageRequest(
            request_id: requestId,
            patient_context: PatientContext(
                age_years: age,
                gestational_weeks: gestationalWeeks,
                known_conditions: knownConditions.filter { $0.lowercased() != "none" },
                medications: medications.filter { $0.lowercased() != "none" }
            ),
            vitals: Vitals(
                systolic_bp: systolicBp,
                diastolic_bp: diastolicBp,
                fasting_glucose: fastingGlucose,
                hr: heartRate,
                spo2: spo2Percent,
                temp_c: tempC,
                hrv: hrv
            ),
            inputs: TriageInputs(
                headache: headache,
                vision_changes: visionChanges,
                decreased_fetal_movement: decreasedFetalMovement,
                free_text: fullNote,
                wound_image_b64: woundImageB64,
                skin_image_b64: skinImageB64,
                audio_b64: audioB64
            ),
            metadata: TriageMetadata(
                source: "momnitrix_watch_app",
                composer_mode: "medgemma_first",
                response_composer_mode: "medgemma_first",
                medgemma_output_style: "notebook",
                ui_mode: uiMode,
                simulator: SimulatorMeta(
                    age_years: age,
                    bmi_group: bmiGroup,
                    gravidity: gravidity,
                    parity: parity,
                    temp_input_unit: "degC"
                )
            )
        )
    }
}
