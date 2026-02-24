/**
 * Momnitrix QA Console – Payload Composition
 *
 * Pure functions to build the TriageRequest payload.  No DOM access.
 */

import type {
    TriageRequest,
    FormValues,
    Modality,
} from './types';
import {
    PAYLOAD_SOURCE,
    COMPOSER_MODE,
    OUTPUT_STYLE,
    TEMP_INPUT_UNIT,
} from './constants';
import { randomInt } from './randomizer';

/**
 * Convert a comma-separated string to a trimmed, non-empty list.
 */
export function parseCsvList(value: string): string[] {
    return String(value || '')
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean);
}

/**
 * Convert Fahrenheit to Celsius, rounded to 2 decimal places.
 */
export function celsiusFromFahrenheit(tempF: number): number {
    return Number((((tempF - 32) * 5) / 9).toFixed(2));
}

/**
 * Build the patient profile summary string for the free-text note.
 */
export function composeProfileSummary(
    age: number,
    gravidity: number,
    parity: number,
    gestWeeks: number,
    bmiGroup: string,
): string {
    return [
        'Patient profile:',
        `- Age: ${age} years`,
        `- Obstetric history: G${gravidity}P${parity}`,
        `- Gestational age: ${gestWeeks} weeks`,
        `- BMI group: ${bmiGroup}`,
    ].join('\n');
}

/**
 * Build the full TriageRequest payload from form values.
 *
 * @param values     – all extracted form field values
 * @param activeMode – currently selected modality
 * @returns a TriageRequest ready for JSON serialisation
 */
export function composePayloadSkeleton(
    values: FormValues,
    activeMode: Modality,
): TriageRequest {
    const knownConditions = parseCsvList(values.knownConditions).filter(
        (x) => x.toLowerCase() !== 'none',
    );
    const medications = parseCsvList(values.medications).filter(
        (x) => x.toLowerCase() !== 'none',
    );
    const note = (values.freeText || '').trim();
    const age = values.age ?? 0;
    const gravidity = values.gravidity ?? 0;
    const parity = values.parity ?? 0;
    const gestWeeks = values.gestWeeks ?? 0;
    const bmiGroup = values.bmiGroup || 'normal';

    const profileSummary = composeProfileSummary(
        age,
        gravidity,
        parity,
        gestWeeks,
        bmiGroup,
    );
    const tempF = values.tempF ?? 98;
    const requestId =
        (values.requestId || '').trim() ||
        `qa-${Date.now()}-${randomInt(100, 999)}`;

    const noteParts = [profileSummary];
    if (note) {
        noteParts.push(`Clinical note:\n${note}`);
    } else if (activeMode === 'voice') {
        noteParts.push(
            'Voice-first check-in. Prioritize transcript content from MedASR.',
        );
    } else if (activeMode === 'image') {
        noteParts.push(
            'Image-first check-in. Combine specialist image outputs with vitals.',
        );
    }

    return {
        request_id: requestId,
        patient_context: {
            age_years: age,
            gestational_weeks: gestWeeks,
            known_conditions: knownConditions,
            medications: medications,
        },
        vitals: {
            systolic_bp: values.systolicBp ?? 0,
            diastolic_bp: values.diastolicBp ?? 0,
            fasting_glucose: values.fastingGlucose ?? 0,
            hr: values.restingHr ?? 0,
            spo2: values.spo2 ?? 0,
            temp_c: celsiusFromFahrenheit(tempF),
            hrv: values.hrv ?? 0,
        },
        inputs: {
            headache: values.headache,
            vision_changes: values.visionChanges,
            decreased_fetal_movement: values.decreasedFetalMovement,
            free_text: noteParts.join('\n\n'),
            wound_image_b64: null,
            skin_image_b64: null,
            audio_b64: null,
        },
        metadata: {
            source: PAYLOAD_SOURCE,
            composer_mode: COMPOSER_MODE,
            medgemma_output_style: OUTPUT_STYLE,
            ui_mode: activeMode,
            simulator: {
                age_years: age,
                bmi_group: bmiGroup,
                gravidity: gravidity,
                parity: parity,
                temp_input_unit: TEMP_INPUT_UNIT,
            },
        },
    };
}
