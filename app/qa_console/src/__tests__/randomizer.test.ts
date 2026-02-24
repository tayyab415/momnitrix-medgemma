/**
 * Randomizer tests.
 */

import { describe, it, expect } from 'vitest';
import {
    randomInt,
    randomFloat,
    weightedPick,
    pick,
    generateRandomProfile,
} from '../domain/randomizer';
import { PROFILE_MODES } from '../domain/constants';

/** A deterministic RNG for testing: cycles through provided values. */
function seededRng(values: number[]): () => number {
    let index = 0;
    return () => {
        const val = values[index % values.length];
        index++;
        return val;
    };
}

describe('randomInt', () => {
    it('returns min when rng returns 0', () => {
        expect(randomInt(10, 20, () => 0)).toBe(10);
    });
    it('returns max when rng returns ~1', () => {
        expect(randomInt(10, 20, () => 0.999)).toBe(20);
    });
    it('stays within bounds over many calls', () => {
        for (let i = 0; i < 100; i++) {
            const v = randomInt(5, 15);
            expect(v).toBeGreaterThanOrEqual(5);
            expect(v).toBeLessThanOrEqual(15);
        }
    });
});

describe('randomFloat', () => {
    it('returns min when rng returns 0', () => {
        expect(randomFloat(4.0, 5.2, 0.1, () => 0)).toBe(4.0);
    });
    it('stays within bounds', () => {
        for (let i = 0; i < 100; i++) {
            const v = randomFloat(4.0, 5.2, 0.1);
            expect(v).toBeGreaterThanOrEqual(4.0);
            expect(v).toBeLessThanOrEqual(5.2);
        }
    });
});

describe('weightedPick', () => {
    it('picks first option when rng returns 0', () => {
        const options = [
            { name: 'a', weight: 1 },
            { name: 'b', weight: 1 },
        ];
        const result = weightedPick(options, () => 0);
        expect(result.name).toBe('a');
    });
    it('picks last option when rng returns ~1', () => {
        const options = [
            { name: 'a', weight: 1 },
            { name: 'b', weight: 1 },
        ];
        const result = weightedPick(options, () => 0.999);
        expect(result.name).toBe('b');
    });
});

describe('pick', () => {
    it('selects element from array', () => {
        const values = ['x', 'y', 'z'];
        const result = pick(values, () => 0);
        expect(values).toContain(result);
    });
});

describe('generateRandomProfile', () => {
    it('produces valid profile within required bounds', () => {
        for (let i = 0; i < 30; i++) {
            const profile = generateRandomProfile();

            // Check required fields are within at least one profile mode's ranges
            expect(profile.age).toBeGreaterThanOrEqual(13);
            expect(profile.age).toBeLessThanOrEqual(50);
            expect(profile.systolicBp).toBeGreaterThanOrEqual(80);
            expect(profile.systolicBp).toBeLessThanOrEqual(180);
            expect(profile.diastolicBp).toBeLessThan(profile.systolicBp);
            expect(profile.fastingGlucose).toBeGreaterThanOrEqual(3.0);
            expect(profile.fastingGlucose).toBeLessThanOrEqual(20.0);
            expect(profile.tempF).toBeGreaterThanOrEqual(95);
            expect(profile.tempF).toBeLessThanOrEqual(104);
            expect(profile.restingHr).toBeGreaterThanOrEqual(45);
            expect(profile.restingHr).toBeLessThanOrEqual(140);
        }
    });

    it('parity never exceeds gravidity', () => {
        for (let i = 0; i < 50; i++) {
            const profile = generateRandomProfile();
            expect(profile.parity).toBeLessThanOrEqual(profile.gravidity);
        }
    });

    it('assigns a valid profile mode name', () => {
        const validNames = PROFILE_MODES.map((m) => m.name);
        for (let i = 0; i < 20; i++) {
            const profile = generateRandomProfile();
            expect(validNames).toContain(profile.profileModeName);
        }
    });

    it('is deterministic with seeded RNG', () => {
        const rng1 = seededRng([0.3, 0.5, 0.1, 0.7, 0.2, 0.4]);
        const rng2 = seededRng([0.3, 0.5, 0.1, 0.7, 0.2, 0.4]);
        const profile1 = generateRandomProfile(rng1);
        const profile2 = generateRandomProfile(rng2);
        expect(profile1).toEqual(profile2);
    });
});
