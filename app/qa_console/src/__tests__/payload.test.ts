/**
 * Payload composition tests.
 */

import { describe, it, expect } from 'vitest';
import {
    parseCsvList,
    celsiusFromFahrenheit,
    composeProfileSummary,
    composePayloadSkeleton,
} from '../domain/payload';
import type { FormValues } from '../domain/types';

describe('parseCsvList', () => {
    it('splits comma-separated values', () => {
        expect(parseCsvList('a, b, c')).toEqual(['a', 'b', 'c']);
    });
    it('filters empty items', () => {
        expect(parseCsvList(',a,,b,')).toEqual(['a', 'b']);
    });
    it('returns empty array for empty string', () => {
        expect(parseCsvList('')).toEqual([]);
    });
});

describe('celsiusFromFahrenheit', () => {
    it('converts 98.6F to 37C', () => {
        expect(celsiusFromFahrenheit(98.6)).toBeCloseTo(37, 0);
    });
    it('converts 32F to 0C', () => {
        expect(celsiusFromFahrenheit(32)).toBe(0);
    });
    it('converts 212F to 100C', () => {
        expect(celsiusFromFahrenheit(212)).toBe(100);
    });
});

describe('composeProfileSummary', () => {
    it('includes all fields', () => {
        const summary = composeProfileSummary(25, 2, 1, 31, 'overweight');
        expect(summary).toContain('Age: 25');
        expect(summary).toContain('G2P1');
        expect(summary).toContain('31 weeks');
        expect(summary).toContain('overweight');
    });
});

describe('composePayloadSkeleton', () => {
    const baseValues: FormValues = {
        age: 25,
        gestWeeks: 31,
        gravidity: 2,
        parity: 1,
        bmiGroup: 'overweight',
        knownConditions: 'chronic_hypertension',
        medications: 'prenatal_vitamins',
        systolicBp: 130,
        diastolicBp: 80,
        fastingGlucose: 5.1,
        tempF: 98.0,
        restingHr: 86,
        spo2: 98,
        hrv: 42,
        headache: false,
        visionChanges: false,
        decreasedFetalMovement: false,
        freeText: 'A test note',
        requestId: 'test-123',
        backendUrl: 'https://example.com',
    };

    it('produces correct top-level structure', () => {
        const payload = composePayloadSkeleton(baseValues, 'text');
        expect(payload.request_id).toBe('test-123');
        expect(payload.patient_context.age_years).toBe(25);
        expect(payload.vitals.systolic_bp).toBe(130);
        expect(payload.inputs.headache).toBe(false);
        expect(payload.metadata.ui_mode).toBe('text');
    });

    it('converts temperature to Celsius', () => {
        const payload = composePayloadSkeleton(baseValues, 'text');
        expect(payload.vitals.temp_c).toBeCloseTo(36.67, 1);
    });

    it('filters out "none" from conditions', () => {
        const values = { ...baseValues, knownConditions: 'none' };
        const payload = composePayloadSkeleton(values, 'text');
        expect(payload.patient_context.known_conditions).toEqual([]);
    });

    it('includes clinical note in free_text', () => {
        const payload = composePayloadSkeleton(baseValues, 'text');
        expect(payload.inputs.free_text).toContain('A test note');
    });

    it('uses voice prompt when mode is voice and no note', () => {
        const values = { ...baseValues, freeText: '' };
        const payload = composePayloadSkeleton(values, 'voice');
        expect(payload.inputs.free_text).toContain('Voice-first');
    });

    it('uses image prompt when mode is image and no note', () => {
        const values = { ...baseValues, freeText: '' };
        const payload = composePayloadSkeleton(values, 'image');
        expect(payload.inputs.free_text).toContain('Image-first');
    });

    it('initialises media fields as null', () => {
        const payload = composePayloadSkeleton(baseValues, 'text');
        expect(payload.inputs.wound_image_b64).toBeNull();
        expect(payload.inputs.skin_image_b64).toBeNull();
        expect(payload.inputs.audio_b64).toBeNull();
    });

    it('preserves payload contract keys', () => {
        const payload = composePayloadSkeleton(baseValues, 'text');
        // Verify exact key names match backend expectations
        expect('request_id' in payload).toBe(true);
        expect('patient_context' in payload).toBe(true);
        expect('vitals' in payload).toBe(true);
        expect('inputs' in payload).toBe(true);
        expect('metadata' in payload).toBe(true);
        expect('age_years' in payload.patient_context).toBe(true);
        expect('systolic_bp' in payload.vitals).toBe(true);
        expect('diastolic_bp' in payload.vitals).toBe(true);
        expect('fasting_glucose' in payload.vitals).toBe(true);
        expect('hr' in payload.vitals).toBe(true);
        expect('temp_c' in payload.vitals).toBe(true);
        expect('wound_image_b64' in payload.inputs).toBe(true);
        expect('skin_image_b64' in payload.inputs).toBe(true);
        expect('audio_b64' in payload.inputs).toBe(true);
        expect('vision_changes' in payload.inputs).toBe(true);
        expect('decreased_fetal_movement' in payload.inputs).toBe(true);
    });
});
