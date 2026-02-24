/**
 * Momnitrix QA Console – Pure Validation Logic
 *
 * No DOM access.  Accepts extracted values and returns structured results.
 */

import type {
    ValidationResult,
    RequiredRule,
    OptionalNumericRule,
    RequiredValuesMap,
    OptionalValuesMap,
    Modality,
} from './types';
import { REQUIRED_RULES, OPTIONAL_NUMERIC_RULES } from './constants';

/**
 * Evaluate required vitals fields.
 *
 * @param values – a map from rule.id to the numeric value (or null if
 *   missing/invalid).
 * @param rules  – override for testing; defaults to REQUIRED_RULES.
 * @returns missing labels and validation error messages.
 */
export function evaluateRequiredInputs(
    values: RequiredValuesMap,
    rules: readonly RequiredRule[] = REQUIRED_RULES,
): ValidationResult {
    const missing: string[] = [];
    const errors: string[] = [];

    for (const rule of rules) {
        const value = values[rule.id] ?? null;
        if (value === null) {
            missing.push(rule.label);
            continue;
        }
        if (value < rule.min || value > rule.max) {
            errors.push(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
        }
    }

    const systolic = values['systolicBp'] ?? null;
    const diastolic = values['diastolicBp'] ?? null;
    if (systolic !== null && diastolic !== null && systolic <= diastolic) {
        errors.push('Systolic BP must be greater than diastolic BP.');
    }

    return { missing, errors };
}

/**
 * Full input validation (required + optional + backend URL + mode checks).
 *
 * @returns array of human-readable error strings.  Empty = valid.
 */
export function validateAllInputs(
    requiredValues: RequiredValuesMap,
    optionalValues: OptionalValuesMap,
    backendUrl: string,
    activeMode: Modality,
    hasAudio: boolean,
    requiredRules: readonly RequiredRule[] = REQUIRED_RULES,
    optionalRules: readonly OptionalNumericRule[] = OPTIONAL_NUMERIC_RULES,
): string[] {
    const errors: string[] = [];

    const required = evaluateRequiredInputs(requiredValues, requiredRules);
    errors.push(...required.errors);
    if (required.missing.length > 0) {
        errors.push(`Missing required inputs: ${required.missing.join(', ')}.`);
    }

    for (const rule of optionalRules) {
        const raw = (optionalValues[rule.id] ?? '').trim();
        if (!raw) continue;
        const value = Number(raw);
        if (Number.isNaN(value)) {
            errors.push(`${rule.label} must be a valid number.`);
            continue;
        }
        if (value < rule.min || value > rule.max) {
            errors.push(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
        }
    }

    if (!backendUrl.startsWith('http://') && !backendUrl.startsWith('https://')) {
        errors.push('Backend URL must start with http:// or https://');
    }

    if (activeMode === 'voice' && !hasAudio) {
        errors.push(
            'Voice mode selected but no audio attached. Upload audio or record in-app.',
        );
    }

    return errors;
}

/**
 * Parse a raw string value from an input field to a number or null.
 */
export function parseNumericInput(raw: string): number | null {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    const value = Number(trimmed);
    return Number.isNaN(value) ? null : value;
}
