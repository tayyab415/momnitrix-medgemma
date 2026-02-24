/**
 * Validation module tests.
 */

import { describe, it, expect } from 'vitest';
import {
    evaluateRequiredInputs,
    validateAllInputs,
    parseNumericInput,
} from '../domain/validation';
import { REQUIRED_RULES } from '../domain/constants';

describe('parseNumericInput', () => {
    it('returns null for empty string', () => {
        expect(parseNumericInput('')).toBeNull();
    });
    it('returns null for whitespace', () => {
        expect(parseNumericInput('   ')).toBeNull();
    });
    it('returns null for non-numeric', () => {
        expect(parseNumericInput('abc')).toBeNull();
    });
    it('parses integer', () => {
        expect(parseNumericInput('25')).toBe(25);
    });
    it('parses float', () => {
        expect(parseNumericInput('5.1')).toBe(5.1);
    });
});

describe('evaluateRequiredInputs', () => {
    it('reports all missing when values are null', () => {
        const values: Record<string, number | null> = {};
        for (const rule of REQUIRED_RULES) {
            values[rule.id] = null;
        }
        const result = evaluateRequiredInputs(values);
        expect(result.missing).toHaveLength(REQUIRED_RULES.length);
        expect(result.errors).toHaveLength(0);
    });

    it('reports no missing when all values are present and valid', () => {
        const result = evaluateRequiredInputs({
            age: 25,
            systolicBp: 130,
            diastolicBp: 80,
            fastingGlucose: 5.1,
            tempF: 98.0,
            restingHr: 86,
        });
        expect(result.missing).toHaveLength(0);
        expect(result.errors).toHaveLength(0);
    });

    it('reports out-of-range error', () => {
        const result = evaluateRequiredInputs({
            age: 5, // below min of 13
            systolicBp: 130,
            diastolicBp: 80,
            fastingGlucose: 5.1,
            tempF: 98.0,
            restingHr: 86,
        });
        expect(result.missing).toHaveLength(0);
        expect(result.errors).toHaveLength(1);
        expect(result.errors[0]).toContain('Age');
    });

    it('reports systolic <= diastolic error', () => {
        const result = evaluateRequiredInputs({
            age: 25,
            systolicBp: 80,
            diastolicBp: 80,
            fastingGlucose: 5.1,
            tempF: 98.0,
            restingHr: 86,
        });
        expect(result.errors).toContain(
            'Systolic BP must be greater than diastolic BP.',
        );
    });

    it('boundary values are valid', () => {
        const result = evaluateRequiredInputs({
            age: 13, // min
            systolicBp: 180, // max
            diastolicBp: 45, // min
            fastingGlucose: 3.0, // min
            tempF: 104, // max
            restingHr: 45, // min
        });
        expect(result.missing).toHaveLength(0);
        expect(result.errors).toHaveLength(0);
    });
});

describe('validateAllInputs', () => {
    const validRequired: Record<string, number | null> = {
        age: 25,
        systolicBp: 130,
        diastolicBp: 80,
        fastingGlucose: 5.1,
        tempF: 98.0,
        restingHr: 86,
    };

    it('returns empty array for valid inputs', () => {
        const errors = validateAllInputs(
            validRequired,
            {},
            'https://example.com',
            'text',
            false,
        );
        expect(errors).toHaveLength(0);
    });

    it('rejects bad backend URL', () => {
        const errors = validateAllInputs(
            validRequired,
            {},
            'ftp://example.com',
            'text',
            false,
        );
        expect(errors.some((e) => e.includes('Backend URL'))).toBe(true);
    });

    it('voice mode without audio produces error', () => {
        const errors = validateAllInputs(
            validRequired,
            {},
            'https://example.com',
            'voice',
            false,
        );
        expect(errors.some((e) => e.includes('Voice mode'))).toBe(true);
    });

    it('voice mode with audio is valid', () => {
        const errors = validateAllInputs(
            validRequired,
            {},
            'https://example.com',
            'voice',
            true,
        );
        expect(errors).toHaveLength(0);
    });

    it('optional out-of-range value is rejected', () => {
        const errors = validateAllInputs(
            validRequired,
            { gestWeeks: '50' }, // max is 42
            'https://example.com',
            'text',
            false,
        );
        expect(errors.some((e) => e.includes('Gestational weeks'))).toBe(true);
    });

    it('optional non-numeric value is rejected', () => {
        const errors = validateAllInputs(
            validRequired,
            { spo2: 'abc' },
            'https://example.com',
            'text',
            false,
        );
        expect(errors.some((e) => e.includes('SpO2'))).toBe(true);
    });
});
