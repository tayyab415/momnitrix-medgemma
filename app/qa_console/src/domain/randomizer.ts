/**
 * Momnitrix QA Console – Clinical Profile Randomizer
 *
 * All randomisation is injectable via an RNG function for deterministic
 * testing.  No DOM access.
 */

import type {
    ProfileMode,
    RandomProfileResult,
    RngFunction,
} from './types';
import { PROFILE_MODES } from './constants';

// ---------------------------------------------------------------------------
// Primitive helpers
// ---------------------------------------------------------------------------

/**
 * Random integer in [min, max] (inclusive).
 */
export function randomInt(
    min: number,
    max: number,
    rng: RngFunction = Math.random,
): number {
    return Math.floor(rng() * (max - min + 1)) + min;
}

/**
 * Random float in [min, max] with given step, rounded to 2 decimal places.
 */
export function randomFloat(
    min: number,
    max: number,
    step = 0.1,
    rng: RngFunction = Math.random,
): number {
    const span = Math.round((max - min) / step);
    const pick = randomInt(0, span, rng);
    return Number((min + pick * step).toFixed(2));
}

/**
 * Weighted random selection from an array of items with `.weight`.
 */
export function weightedPick<T extends { weight: number }>(
    options: readonly T[],
    rng: RngFunction = Math.random,
): T {
    const total = options.reduce((sum, item) => sum + item.weight, 0);
    let marker = rng() * total;
    for (const item of options) {
        marker -= item.weight;
        if (marker <= 0) return item;
    }
    return options[options.length - 1];
}

/**
 * Uniformly pick one element from an array.
 */
export function pick<T>(
    values: readonly T[],
    rng: RngFunction = Math.random,
): T {
    return values[randomInt(0, values.length - 1, rng)];
}

// ---------------------------------------------------------------------------
// Profile generation
// ---------------------------------------------------------------------------

/**
 * Generate a complete random patient profile using weighted profile modes.
 *
 * Returns a pure-data object (no DOM references).  The caller (UI layer)
 * is responsible for applying the values to form fields.
 */
export function generateRandomProfile(
    rng: RngFunction = Math.random,
    modes: readonly ProfileMode[] = PROFILE_MODES,
): RandomProfileResult {
    const mode = weightedPick(modes, rng);
    const r = mode.valueRanges;

    const age = randomInt(r.age[0], r.age[1], rng);
    const gestWeeks = randomInt(r.gestWeeks[0], r.gestWeeks[1], rng);
    const gravidity = randomInt(r.gravidity[0], r.gravidity[1], rng);
    const parityCandidate = randomInt(r.parity[0], r.parity[1], rng);
    const parity = Math.min(parityCandidate, gravidity);

    let systolicBp = randomInt(r.systolicBp[0], r.systolicBp[1], rng);
    let diastolicBp = randomInt(r.diastolicBp[0], r.diastolicBp[1], rng);
    if (diastolicBp >= systolicBp) {
        diastolicBp = systolicBp - randomInt(8, 25, rng);
    }

    const fastingGlucose = randomFloat(
        r.fastingGlucose[0],
        r.fastingGlucose[1],
        0.1,
        rng,
    );
    const tempF = randomFloat(r.tempF[0], r.tempF[1], 0.1, rng);
    const restingHr = randomInt(r.restingHr[0], r.restingHr[1], rng);
    const spo2 = randomInt(r.spo2[0], r.spo2[1], rng);
    const hrv = randomFloat(r.hrv[0], r.hrv[1], 0.1, rng);

    const headache = rng() < mode.booleans.headache;
    const visionChanges = rng() < mode.booleans.visionChanges;
    const decreasedFetalMovement = rng() < mode.booleans.decreasedFetalMovement;

    const bmiGroup = pick(mode.bmiGroup, rng);
    const selectedCondition = pick(mode.knownConditions, rng);
    const selectedMedication = pick(mode.medications, rng);
    const freeText = pick(mode.notes, rng);

    return {
        age,
        gestWeeks,
        gravidity,
        parity,
        bmiGroup,
        systolicBp,
        diastolicBp,
        fastingGlucose,
        tempF,
        restingHr,
        spo2,
        hrv,
        headache,
        visionChanges,
        decreasedFetalMovement,
        knownConditions: selectedCondition === 'none' ? '' : selectedCondition,
        medications: selectedMedication === 'none' ? '' : selectedMedication,
        freeText,
        profileModeName: mode.name,
    };
}
