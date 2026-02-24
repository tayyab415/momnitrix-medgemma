/**
 * Momnitrix QA Console – Centralized Constants
 *
 * Single source of truth for all hardcoded IDs, labels, ranges, and
 * configuration values.  No duplicated literals across files.
 */

import type {
    RequiredRule,
    OptionalNumericRule,
    ProfileMode,
    Modality,
} from './types';

// ---------------------------------------------------------------------------
// Required vitals rules (gate the Diagnose button)
// ---------------------------------------------------------------------------

export const REQUIRED_RULES: readonly RequiredRule[] = [
    { id: 'age', label: 'Age', min: 13, max: 50 },
    { id: 'systolicBp', label: 'Systolic BP', min: 80, max: 180 },
    { id: 'diastolicBp', label: 'Diastolic BP', min: 45, max: 120 },
    { id: 'fastingGlucose', label: 'Fasting glucose', min: 3.0, max: 20.0 },
    { id: 'tempF', label: 'Body temperature (degF)', min: 95, max: 104 },
    { id: 'restingHr', label: 'Resting heart rate', min: 45, max: 140 },
] as const;

// ---------------------------------------------------------------------------
// Optional numeric rules
// ---------------------------------------------------------------------------

export const OPTIONAL_NUMERIC_RULES: readonly OptionalNumericRule[] = [
    { id: 'gestWeeks', label: 'Gestational weeks', min: 4, max: 42 },
    { id: 'gravidity', label: 'Gravidity', min: 1, max: 10 },
    { id: 'parity', label: 'Parity', min: 0, max: 10 },
    { id: 'spo2', label: 'SpO2', min: 88, max: 100 },
    { id: 'hrv', label: 'HRV', min: 10, max: 140 },
] as const;

// ---------------------------------------------------------------------------
// SSE event labels
// ---------------------------------------------------------------------------

export const EVENT_LABELS: Readonly<Record<string, string>> = {
    'request.accepted': 'Request Accepted',
    'request.rejected': 'Request Rejected',
    'router.decision': 'Router Decision',
    'router.prompt_plan': 'Prompt Plan',
    'model.started': 'Specialist Started',
    'model.completed': 'Specialist Completed',
    'model.failed': 'Specialist Failed',
    'medgemma.started': 'MedGemma Started',
    'medgemma.completed': 'MedGemma Completed',
    'medgemma.delta': 'MedGemma Stream Chunk',
    'gemini.started': 'Gemini Started',
    'gemini.completed': 'Gemini Completed',
    'gemini.skipped': 'Gemini Skipped',
    'diagnostics.inference_breakdown': 'Inference Diagnostics',
    'triage.final': 'Final Response',
    'triage.error': 'Pipeline Error',
};

// ---------------------------------------------------------------------------
// Modalities
// ---------------------------------------------------------------------------

export const MODALITIES: readonly Modality[] = ['text', 'voice', 'image'];

// ---------------------------------------------------------------------------
// Result card ordering (matches HTML data-card attributes)
// ---------------------------------------------------------------------------

export const RESULT_CARD_ORDER: readonly string[] = [
    'overview',
    'patient_guidance',
    'visit_summary',
    'specialist_signals',
    'clinical_reasons',
    'action_plan',
    'raw_json',
];

// ---------------------------------------------------------------------------
// Default form field values (used by resetToDefaults)
// ---------------------------------------------------------------------------

export const DEFAULT_FORM_VALUES = {
    requestId: '',
    randomProfileMode: 'none yet',
    age: '25',
    gestWeeks: '31',
    gravidity: '2',
    parity: '1',
    bmiGroup: 'overweight',
    knownConditions: '',
    medications: 'prenatal_vitamins',
    systolicBp: '130',
    diastolicBp: '80',
    fastingGlucose: '5.1',
    tempF: '98.0',
    restingHr: '86',
    spo2: '98',
    hrv: '42.0',
    headache: false,
    visionChanges: false,
    decreasedFetalMovement: false,
    freeText: '',
} as const;

// ---------------------------------------------------------------------------
// Profile modes for clinical randomization
// ---------------------------------------------------------------------------

export const PROFILE_MODES: readonly ProfileMode[] = [
    {
        name: 'normal_monitoring',
        weight: 0.5,
        valueRanges: {
            age: [20, 34],
            gestWeeks: [20, 39],
            gravidity: [1, 4],
            parity: [0, 3],
            systolicBp: [104, 126],
            diastolicBp: [64, 82],
            fastingGlucose: [4.0, 5.2],
            tempF: [97.3, 99.0],
            restingHr: [62, 92],
            spo2: [96, 100],
            hrv: [28, 95],
        },
        bmiGroup: ['normal', 'overweight'],
        knownConditions: ['none'],
        medications: ['prenatal_vitamins'],
        booleans: {
            headache: 0.08,
            visionChanges: 0.04,
            decreasedFetalMovement: 0.05,
        },
        notes: [
            'Routine evening check-in, feeling mostly stable.',
            'No major concerns today, just regular tracking update.',
        ],
    },
    {
        name: 'borderline_risk',
        weight: 0.34,
        valueRanges: {
            age: [24, 40],
            gestWeeks: [26, 40],
            gravidity: [2, 6],
            parity: [1, 4],
            systolicBp: [128, 146],
            diastolicBp: [80, 94],
            fastingGlucose: [5.3, 7.8],
            tempF: [97.6, 99.7],
            restingHr: [76, 104],
            spo2: [94, 99],
            hrv: [18, 70],
        },
        bmiGroup: ['overweight', 'obese'],
        knownConditions: ['prior_gdm', 'chronic_hypertension', 'none'],
        medications: ['prenatal_vitamins', 'low_dose_aspirin', 'none'],
        booleans: {
            headache: 0.35,
            visionChanges: 0.12,
            decreasedFetalMovement: 0.16,
        },
        notes: [
            'Mild headache intermittently, requested triage review.',
            'Monitoring trend change over last 2 days.',
        ],
    },
    {
        name: 'red_flag',
        weight: 0.16,
        valueRanges: {
            age: [19, 43],
            gestWeeks: [28, 41],
            gravidity: [2, 8],
            parity: [1, 5],
            systolicBp: [150, 170],
            diastolicBp: [95, 112],
            fastingGlucose: [8.5, 16.0],
            tempF: [98.0, 101.5],
            restingHr: [88, 125],
            spo2: [92, 98],
            hrv: [10, 55],
        },
        bmiGroup: ['overweight', 'obese'],
        knownConditions: [
            'gestational_diabetes',
            'chronic_hypertension',
            'prior_preeclampsia',
        ],
        medications: ['insulin', 'labetalol', 'prenatal_vitamins'],
        booleans: {
            headache: 0.62,
            visionChanges: 0.4,
            decreasedFetalMovement: 0.28,
        },
        notes: [
            'Symptoms feel worse than baseline, requesting urgent review.',
            'Persistent headache and elevated values, worried about progression.',
        ],
    },
];

// ---------------------------------------------------------------------------
// Metadata constants sent in payload
// ---------------------------------------------------------------------------

export const PAYLOAD_SOURCE = 'qa_console_watch_prototype';
export const COMPOSER_MODE = 'medgemma_first';
export const OUTPUT_STYLE = 'notebook';
export const TEMP_INPUT_UNIT = 'degF';

// ---------------------------------------------------------------------------
// Audio MIME candidates for MediaRecorder (preference order)
// ---------------------------------------------------------------------------

export const AUDIO_MIME_CANDIDATES: readonly string[] = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
];
