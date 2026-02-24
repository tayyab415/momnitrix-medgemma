/**
 * Momnitrix QA Console – DOM / UI Orchestrator
 *
 * This is the ONLY module that touches `document` and `window`.
 * It reads form values, calls pure domain functions, and writes
 * results back to the DOM.
 */

import type {
    FormValues,
    Modality,
    RunLog,
    TriageFinalResponse,
    TriageRequest,
} from '../domain/types';

import {
    REQUIRED_RULES,
    OPTIONAL_NUMERIC_RULES,
    DEFAULT_FORM_VALUES,
    AUDIO_MIME_CANDIDATES,
} from '../domain/constants';

import { evaluateRequiredInputs, validateAllInputs, parseNumericInput } from '../domain/validation';
import { composePayloadSkeleton } from '../domain/payload';
import { createSseEventParser } from '../domain/sse';
import { generateRandomProfile } from '../domain/randomizer';
import { convertToWavIfPossible } from '../adapters/audio';
import { attachMedia } from '../adapters/media';
import { streamSseFromFetch } from '../adapters/streaming';
import {
    formatPatientGuidanceHtml,
    formatVisitSummaryHtml,
    formatSpecialistSummaryHtml,
    formatTimelineEntryText,
    stringifyJson,
} from '../domain/formatters';

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function el(id: string): HTMLElement {
    return document.getElementById(id)!;
}

function inputEl(id: string): HTMLInputElement {
    return document.getElementById(id) as HTMLInputElement;
}

function selectEl(id: string): HTMLSelectElement {
    return document.getElementById(id) as HTMLSelectElement;
}

function textareaEl(id: string): HTMLTextAreaElement {
    return document.getElementById(id) as HTMLTextAreaElement;
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

interface AppState {
    lastRun: RunLog | null;
    activeRun: RunLog | null;
    mediaRecorder: MediaRecorder | null;
    mediaStream: MediaStream | null;
    mediaChunks: Blob[];
    recordedAudioBlob: Blob | null;
    finalReceived: boolean;
    activeMode: Modality;
    resultCardIndex: number;
    inferenceTimer: ReturnType<typeof setInterval> | null;
    inferenceStartTime: number | null;
}

const state: AppState = {
    lastRun: null,
    activeRun: null,
    mediaRecorder: null,
    mediaStream: null,
    mediaChunks: [],
    recordedAudioBlob: null,
    finalReceived: false,
    activeMode: 'text',
    resultCardIndex: 0,
    inferenceTimer: null,
    inferenceStartTime: null,
};

// ---------------------------------------------------------------------------
// Form value extraction (DOM → pure types)
// ---------------------------------------------------------------------------

function getNumericValue(id: string): number | null {
    return parseNumericInput(inputEl(id).value);
}

function extractRequiredValues(): Record<string, number | null> {
    const values: Record<string, number | null> = {};
    for (const rule of REQUIRED_RULES) {
        values[rule.id] = getNumericValue(rule.id);
    }
    return values;
}

function extractOptionalValues(): Record<string, string> {
    const values: Record<string, string> = {};
    for (const rule of OPTIONAL_NUMERIC_RULES) {
        values[rule.id] = inputEl(rule.id).value;
    }
    return values;
}

function extractFormValues(): FormValues {
    return {
        age: getNumericValue('age'),
        gestWeeks: getNumericValue('gestWeeks'),
        gravidity: getNumericValue('gravidity'),
        parity: getNumericValue('parity'),
        bmiGroup: selectEl('bmiGroup').value,
        knownConditions: inputEl('knownConditions').value,
        medications: inputEl('medications').value,
        systolicBp: getNumericValue('systolicBp'),
        diastolicBp: getNumericValue('diastolicBp'),
        fastingGlucose: getNumericValue('fastingGlucose'),
        tempF: getNumericValue('tempF'),
        restingHr: getNumericValue('restingHr'),
        spo2: getNumericValue('spo2'),
        hrv: getNumericValue('hrv'),
        headache: inputEl('headache').checked,
        visionChanges: inputEl('visionChanges').checked,
        decreasedFetalMovement: inputEl('decreasedFetalMovement').checked,
        freeText: textareaEl('freeText').value,
        requestId: inputEl('requestId').value,
        backendUrl: inputEl('backendUrl').value,
    };
}

// ---------------------------------------------------------------------------
// Status / validation display
// ---------------------------------------------------------------------------

function updateStatus(message: string): void {
    el('statusLine').textContent = message;
}

function updateValidation(message: string): void {
    el('validationLine').textContent = message || '';
}

function setHealthLine(message: string, kind = ''): void {
    const node = el('healthLine');
    node.textContent = message;
    node.className = kind ? `health-line ${kind}` : 'health-line';
}

// ---------------------------------------------------------------------------
// Image preview
// ---------------------------------------------------------------------------

function setPreview(fileInput: HTMLInputElement, previewEl: HTMLImageElement): void {
    const file = fileInput.files?.[0];
    if (!file) {
        previewEl.style.display = 'none';
        previewEl.removeAttribute('src');
        return;
    }
    const objectUrl = URL.createObjectURL(file);
    previewEl.src = objectUrl;
    previewEl.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Required-field gate
// ---------------------------------------------------------------------------

export function updateRequiredGate(): void {
    const requiredValues = extractRequiredValues();
    const { missing, errors } = evaluateRequiredInputs(requiredValues);

    const requiredHint = el('requiredHint');
    const badge = el('requiredBadge');
    const vitalsChip = el('openVitalsBtn');
    const needsInput = missing.length > 0 || errors.length > 0;

    badge.textContent = String(missing.length);

    if (missing.length > 0) {
        requiredHint.textContent = `Missing required inputs: ${missing.join(', ')}.`;
        requiredHint.className = 'required-hint';
        vitalsChip.classList.remove('ok');
    } else if (errors.length > 0) {
        requiredHint.textContent = errors[0];
        requiredHint.className = 'required-hint';
        vitalsChip.classList.remove('ok');
    } else {
        requiredHint.textContent = 'All required vitals are present.';
        requiredHint.className = 'required-hint ok';
    }

    if (needsInput) {
        vitalsChip.classList.remove('ok');
        vitalsChip.classList.add('needs-input');
    } else {
        vitalsChip.classList.remove('needs-input');
        vitalsChip.classList.add('ok');
    }

    if (!state.activeRun) {
        (el('submitBtn') as HTMLButtonElement).disabled =
            missing.length > 0 || errors.length > 0;
    }
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

function setActiveMode(mode: Modality): void {
    state.activeMode = mode;

    // 1. Highlight the button on the home screen
    const modeBtnIds = {
        text: 'modeTextBtn',
        voice: 'modeVoiceBtn',
        image: 'modeImageBtn',
    };
    Object.entries(modeBtnIds).forEach(([m, id]) => {
        el(id).classList.toggle('active', m === mode);
    });

    // 2. Toggle the panel inside the sheet
    const panels: Array<{ id: string; mode: Modality }> = [
        { id: 'textPanel', mode: 'text' },
        { id: 'voicePanel', mode: 'voice' },
        { id: 'imagePanel', mode: 'image' },
    ];

    for (const item of panels) {
        const node = el(item.id);
        if (node) {
            node.classList.toggle('active', item.mode === mode);
        }
    }
    updateModeIndicators();
}

/**
 * Update the indicators (e.g. green dots) on the home screen cards
 * to show which modes currently have pending data (note, audio, or images).
 */
export function updateModeIndicators(): void {
    const hasNote = textareaEl('freeText').value.trim().length > 0;
    const hasAudio =
        inputEl('audioFile').files?.length || state.recordedAudioBlob;
    const hasImage =
        inputEl('woundImage').files?.length || inputEl('skinImage').files?.length;

    el('modeTextBtn').classList.toggle('has-data', hasNote);
    el('modeVoiceBtn').classList.toggle('has-data', !!hasAudio);
    el('modeImageBtn').classList.toggle('has-data', !!hasImage);

    // Also update Diagnose sub-label to show mode context
    const modeLabel = state.activeMode.charAt(0).toUpperCase() + state.activeMode.slice(1);
    const subLabel = el('submitBtn').querySelector('small');
    if (subLabel) {
        subLabel.textContent = `Run AI Triage (${modeLabel})`;
    }
}

// ---------------------------------------------------------------------------
// Inference loading indicator
// ---------------------------------------------------------------------------

function showInferenceLoader(): void {
    const loader = document.getElementById('inferenceLoader');
    if (!loader) return;
    loader.hidden = false;
    // Hide all result cards while loading
    const cards = Array.from(document.querySelectorAll('.result-card')) as HTMLElement[];
    for (const card of cards) card.style.display = 'none';
    const nav = document.querySelector('.carousel-bottom-nav') as HTMLElement | null;
    if (nav) nav.style.display = 'none';
    const pipelineSection = document.querySelector('#resultsWrapper > .panel') as HTMLElement | null;
    if (pipelineSection) pipelineSection.style.display = 'none';

    // Start elapsed timer
    state.inferenceStartTime = Date.now();
    const elapsedEl = document.getElementById('loaderElapsed');
    if (elapsedEl) elapsedEl.textContent = 'Elapsed: 0s';
    if (state.inferenceTimer) clearInterval(state.inferenceTimer);
    state.inferenceTimer = setInterval(() => {
        if (!state.inferenceStartTime) return;
        const secs = Math.floor((Date.now() - state.inferenceStartTime) / 1000);
        const mins = Math.floor(secs / 60);
        const remainSecs = secs % 60;
        if (elapsedEl) {
            elapsedEl.textContent = mins > 0
                ? `Elapsed: ${mins}m ${remainSecs}s`
                : `Elapsed: ${secs}s`;
        }
    }, 1000);
}

function hideInferenceLoader(): void {
    const loader = document.getElementById('inferenceLoader');
    if (loader) loader.hidden = true;
    // Restore result cards visibility
    const cards = Array.from(document.querySelectorAll('.result-card')) as HTMLElement[];
    for (const card of cards) card.style.display = '';
    const nav = document.querySelector('.carousel-bottom-nav') as HTMLElement | null;
    if (nav) nav.style.display = '';
    const pipelineSection = document.querySelector('#resultsWrapper > .panel') as HTMLElement | null;
    if (pipelineSection) pipelineSection.style.display = '';

    // Stop elapsed timer
    if (state.inferenceTimer) {
        clearInterval(state.inferenceTimer);
        state.inferenceTimer = null;
    }
    state.inferenceStartTime = null;
}

// ---------------------------------------------------------------------------
// Result carousel
// ---------------------------------------------------------------------------

function listResultCards(): HTMLElement[] {
    return Array.from(document.querySelectorAll('.result-card')) as HTMLElement[];
}

export function showResultCard(index: number): void {
    const cards = listResultCards();
    if (!cards.length) return;
    const safeIndex = Math.max(0, Math.min(index, cards.length - 1));
    state.resultCardIndex = safeIndex;

    for (let i = 0; i < cards.length; i += 1) {
        cards[i].classList.toggle('is-active', i === safeIndex);
    }

    const prev = el('prevCardBtn') as HTMLButtonElement | null;
    const next = el('nextCardBtn') as HTMLButtonElement | null;
    if (prev) prev.disabled = safeIndex <= 0;
    if (next) next.disabled = safeIndex >= cards.length - 1;
    el('cardPosition').textContent = `${safeIndex + 1} / ${cards.length}`;
}

function shiftResultCard(delta: number): void {
    showResultCard(state.resultCardIndex + delta);
    const wrapper = el('resultsWrapper');
    if (wrapper) wrapper.scrollTop = 0;
}

// ---------------------------------------------------------------------------
// List rendering
// ---------------------------------------------------------------------------

function renderList(id: string, values: unknown, emptyText: string): void {
    const node = el(id);
    node.innerHTML = '';
    const items = Array.isArray(values) ? (values as unknown[]) : [];
    if (!items.length) {
        const li = document.createElement('li');
        li.textContent = emptyText;
        node.appendChild(li);
        return;
    }
    for (const item of items) {
        const li = document.createElement('li');
        li.textContent = String(item);
        node.appendChild(li);
    }
}

// ---------------------------------------------------------------------------
// Run view reset
// ---------------------------------------------------------------------------

function resetRunViews(): void {
    el('eventTimeline').innerHTML = '';
    textareaEl('finalJson').value = '';
    el('riskLevel').textContent = '-';
    const riskEl = document.getElementById('riskLevel');
    if (riskEl) riskEl.className = 'risk-pill-hero';
    const pf = document.getElementById('policyFloor');
    if (pf) pf.textContent = '-';
    const dc = document.getElementById('diagComposer');
    if (dc) dc.textContent = '-';
    const dms = document.getElementById('diagModelShare');
    if (dms) dms.textContent = '-';
    const dcs = document.getElementById('diagColdStart');
    if (dcs) dcs.textContent = '-';
    const dwm = document.getElementById('diagWarmupMs');
    if (dwm) dwm.textContent = '-';
    const dmm = document.getElementById('diagMedgemmaMs');
    if (dmm) dmm.textContent = '-';
    el('diagTotalMs').textContent = '-';
    el('patientMessage').innerHTML = 'No result yet.';
    el('visitSummary').innerHTML = 'No result yet.';
    el('specialistSummary').innerHTML = 'No specialist outputs yet.';
    renderList('reasonsList', [], 'No reasons yet.');
    renderList('actionsList', [], 'No action plan yet.');
    showResultCard(0);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

function setDiagnosticsFromBreakdown(payload: Record<string, unknown>): void {
    const modelShare =
        typeof payload.model === 'string' ? payload.model : '-';
    const dms = document.getElementById('diagModelShare');
    if (dms) dms.textContent = modelShare;
    const dc = document.getElementById('diagComposer');
    if (dc)
        dc.textContent =
            typeof payload.composer_mode === 'string'
                ? payload.composer_mode
                : '-';
    if (payload.gpu_warmup_ms !== undefined) {
        const dwm = document.getElementById('diagWarmupMs');
        if (dwm) dwm.textContent = String(payload.gpu_warmup_ms);
    }
    if (payload.medgemma_inference_ms !== undefined) {
        const dmm = document.getElementById('diagMedgemmaMs');
        if (dmm) dmm.textContent = String(payload.medgemma_inference_ms);
    }
    if (payload.latency_ms !== undefined) {
        const dtm = document.getElementById('diagTotalMs');
        if (dtm) dtm.textContent = String(payload.latency_ms);
    }
}

function setDiagnosticsFromFinal(finalPayload: TriageFinalResponse): void {
    const diag = finalPayload.inference_diagnostics || {};
    const timing = diag.medgemma_timing_breakdown || {};
    const share = diag.latency_share_pct || {};
    const total = finalPayload.latency_ms?.total;

    const dc = document.getElementById('diagComposer');
    if (dc)
        dc.textContent = diag.composer_mode || dc.textContent || '-';

    if (share.medgemma !== undefined || share.gemini !== undefined) {
        const m =
            share.medgemma !== undefined
                ? Number(share.medgemma).toFixed(1)
                : '0.0';
        const g =
            share.gemini !== undefined ? Number(share.gemini).toFixed(1) : '0.0';
        const dms = document.getElementById('diagModelShare');
        if (dms) dms.textContent = `medgemma:${m}% | gemini:${g}%`;
    }
    if (timing.cold_start !== undefined) {
        const dcs = document.getElementById('diagColdStart');
        if (dcs) dcs.textContent = timing.cold_start ? 'YES' : 'NO';
    }
    if (timing.gpu_warmup_ms !== undefined) {
        const dwm = document.getElementById('diagWarmupMs');
        if (dwm) dwm.textContent = String(timing.gpu_warmup_ms);
    }
    if (timing.medgemma_inference_ms !== undefined) {
        const dmm = document.getElementById('diagMedgemmaMs');
        if (dmm) dmm.textContent = String(timing.medgemma_inference_ms);
    }
    if (total !== undefined) {
        const dtm = document.getElementById('diagTotalMs');
        if (dtm) dtm.textContent = String(total);
    }
}

// ---------------------------------------------------------------------------
// Final response rendering
// ---------------------------------------------------------------------------

function setFinalResponse(finalPayload: TriageFinalResponse): void {
    const riskLevel = String(finalPayload.risk_level || '-').toLowerCase();
    const riskEl = el('riskLevel');
    riskEl.textContent = riskLevel.toUpperCase();
    riskEl.className = 'risk-pill-hero ' + riskLevel;
    el('policyFloor').textContent = String(
        finalPayload.policy_floor || '-',
    ).toUpperCase();
    setDiagnosticsFromFinal(finalPayload);
    el('patientMessage').innerHTML = formatPatientGuidanceHtml(finalPayload);
    el('visitSummary').innerHTML = formatVisitSummaryHtml(
        finalPayload.visit_prep_summary,
    );
    const specialist = finalPayload.specialist_outputs || {};
    el('specialistSummary').innerHTML = formatSpecialistSummaryHtml(specialist);
    renderList(
        'reasonsList',
        finalPayload.medgemma_reasons,
        'No clinical reasons returned.',
    );
    renderList('actionsList', finalPayload.action_items, 'No actions returned.');
    textareaEl('finalJson').value = stringifyJson(finalPayload);
    showResultCard(0);
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function timelineEntry(
    eventName: string,
    payload: Record<string, unknown>,
): void {
    const li = document.createElement('li');
    li.textContent = formatTimelineEntryText(eventName, payload);
    el('eventTimeline').appendChild(li);
    li.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------------------
// Run log
// ---------------------------------------------------------------------------

function buildRunLog(url: string, payload: TriageRequest): void {
    state.activeRun = {
        backend_url: url,
        started_at: new Date().toISOString(),
        completed_at: null,
        duration_ms: null,
        request_payload: payload,
        events: [],
        final_response: null,
    };
    state.lastRun = null;
    state.finalReceived = false;
    (el('downloadLogBtn') as HTMLButtonElement).disabled = true;
}

function completeRunLog(): void {
    if (!state.activeRun) return;
    state.activeRun.completed_at = new Date().toISOString();
    const start = Date.parse(state.activeRun.started_at);
    const end = Date.parse(state.activeRun.completed_at);
    state.activeRun.duration_ms =
        Number.isNaN(start) || Number.isNaN(end) ? null : end - start;
    state.lastRun = state.activeRun;
    state.activeRun = null;
    (el('downloadLogBtn') as HTMLButtonElement).disabled = false;
}

// ---------------------------------------------------------------------------
// Backend health check
// ---------------------------------------------------------------------------

async function checkBackendHealth(): Promise<void> {
    updateValidation('');
    const baseUrl = inputEl('backendUrl')
        .value.trim()
        .replace(/\/$/, '');
    if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
        setHealthLine('Invalid backend URL.', 'err');
        return;
    }

    setHealthLine('Checking backend health...', '');
    try {
        const response = await fetch(`${baseUrl}/health?probe=1`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as Record<string, unknown>;
        const core = data.core_gpu_reachable === true ? 'core:up' : 'core:down';
        const derm = data.derm_reachable === true ? 'derm:up' : 'derm:down';
        const asr = data.medasr_reachable === true ? 'medasr:up' : 'medasr:down';
        const composerDefault =
            (data.default_response_composer_mode as string) ||
            (data.response_composer_mode as string) ||
            'unknown';
        setHealthLine(
            `Healthy | ${core} | ${derm} | ${asr} | default_mode=${composerDefault} (request can override)`,
            'ok',
        );
    } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setHealthLine(`Health check failed: ${msg}`, 'err');
    }
}

// ---------------------------------------------------------------------------
// Submit / stream
// ---------------------------------------------------------------------------

async function submitRequest(): Promise<void> {
    updateValidation('');
    const formValues = extractFormValues();
    const requiredValues = extractRequiredValues();
    const optionalValues = extractOptionalValues();

    const uploadedAudio = inputEl('audioFile').files?.[0] ?? null;
    const hasAudio = Boolean(uploadedAudio || state.recordedAudioBlob);

    const validationErrors = validateAllInputs(
        requiredValues,
        optionalValues,
        formValues.backendUrl.trim(),
        state.activeMode,
        hasAudio,
    );

    if (validationErrors.length) {
        updateValidation(validationErrors.join(' '));
        updateRequiredGate();
        openInputValuesPanel();
        return;
    }

    const payload: TriageRequest = composePayloadSkeleton(
        formValues,
        state.activeMode,
    );

    const mediaResult = await attachMedia(payload, {
        woundImage: inputEl('woundImage').files?.[0] ?? null,
        skinImage: inputEl('skinImage').files?.[0] ?? null,
        uploadedAudio,
        recordedAudioBlob: state.recordedAudioBlob,
    });

    if (mediaResult.audioStatusMessage) {
        el('recordStatus').textContent = mediaResult.audioStatusMessage;
    }

    if (state.activeMode === 'voice' && !payload.inputs.audio_b64) {
        updateValidation(
            'Voice mode selected but no audio payload could be attached.',
        );
        return;
    }

    const baseUrl = formValues.backendUrl.trim().replace(/\/$/, '');
    const endpoint = `${baseUrl}/v1/triage/stream`;
    buildRunLog(baseUrl, payload);
    resetRunViews();
    updateStatus('Submitting request and opening SSE stream...');

    const submitBtn = el('submitBtn') as HTMLButtonElement;
    submitBtn.disabled = true;
    openResultsPanel();
    showInferenceLoader();

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`HTTP ${response.status}: ${text}`);
        }

        const parser = createSseEventParser(
            (eventName: string, eventPayload: Record<string, unknown>) => {
                if (state.activeRun) {
                    state.activeRun.events.push({
                        event: eventName,
                        payload: eventPayload,
                    });
                }
                timelineEntry(eventName, eventPayload);

                if (eventName === 'diagnostics.inference_breakdown') {
                    setDiagnosticsFromBreakdown(eventPayload);
                }
                if (eventName === 'triage.final') {
                    state.finalReceived = true;
                    hideInferenceLoader();
                    if (state.activeRun)
                        state.activeRun.final_response =
                            eventPayload as unknown as TriageFinalResponse;
                    setFinalResponse(eventPayload as unknown as TriageFinalResponse);
                }
                if (eventName === 'triage.error') {
                    const msg =
                        typeof eventPayload.error === 'string'
                            ? eventPayload.error
                            : 'Unknown triage error';
                    updateValidation(`Triage error: ${msg}`);
                }
            },
        );

        await streamSseFromFetch(response, parser);
        completeRunLog();

        if (state.finalReceived) {
            updateStatus('Stream complete. Final triage received.');
        } else {
            updateStatus(
                'Stream complete, but no triage.final event was found.',
            );
        }
    } catch (err: unknown) {
        hideInferenceLoader();
        completeRunLog();
        updateStatus('Request failed.');
        const msg = err instanceof Error ? err.message : String(err);
        updateValidation(msg);
    } finally {
        updateRequiredGate();
    }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

export function resetToDefaults(): void {
    inputEl('requestId').value = DEFAULT_FORM_VALUES.requestId;
    inputEl('randomProfileMode').value = DEFAULT_FORM_VALUES.randomProfileMode;
    inputEl('age').value = DEFAULT_FORM_VALUES.age;
    inputEl('gestWeeks').value = DEFAULT_FORM_VALUES.gestWeeks;
    inputEl('gravidity').value = DEFAULT_FORM_VALUES.gravidity;
    inputEl('parity').value = DEFAULT_FORM_VALUES.parity;
    selectEl('bmiGroup').value = DEFAULT_FORM_VALUES.bmiGroup;
    inputEl('knownConditions').value = DEFAULT_FORM_VALUES.knownConditions;
    inputEl('medications').value = DEFAULT_FORM_VALUES.medications;

    inputEl('systolicBp').value = DEFAULT_FORM_VALUES.systolicBp;
    inputEl('diastolicBp').value = DEFAULT_FORM_VALUES.diastolicBp;
    inputEl('fastingGlucose').value = DEFAULT_FORM_VALUES.fastingGlucose;
    inputEl('tempF').value = DEFAULT_FORM_VALUES.tempF;
    inputEl('restingHr').value = DEFAULT_FORM_VALUES.restingHr;
    inputEl('spo2').value = DEFAULT_FORM_VALUES.spo2;
    inputEl('hrv').value = DEFAULT_FORM_VALUES.hrv;

    inputEl('headache').checked = DEFAULT_FORM_VALUES.headache;
    inputEl('visionChanges').checked = DEFAULT_FORM_VALUES.visionChanges;
    inputEl('decreasedFetalMovement').checked =
        DEFAULT_FORM_VALUES.decreasedFetalMovement;
    textareaEl('freeText').value = DEFAULT_FORM_VALUES.freeText;

    inputEl('woundImage').value = '';
    inputEl('skinImage').value = '';
    inputEl('audioFile').value = '';
    setPreview(
        inputEl('woundImage'),
        document.getElementById('woundPreview') as HTMLImageElement,
    );
    setPreview(
        inputEl('skinImage'),
        document.getElementById('skinPreview') as HTMLImageElement,
    );

    state.recordedAudioBlob = null;
    el('recordStatus').textContent = 'Not recording';
    setHealthLine('Not checked.');
    updateValidation('');
    updateStatus('Ready.');
    resetRunViews();
    state.lastRun = null;
    state.activeRun = null;
    (el('downloadLogBtn') as HTMLButtonElement).disabled = true;
    setActiveMode('text');
    updateRequiredGate();
    updateModeIndicators();
    closeResultsPanel();
}

// ---------------------------------------------------------------------------
// Audio status
// ---------------------------------------------------------------------------

function updateAudioStatus(): void {
    const uploadedAudio = inputEl('audioFile').files?.[0];
    if (uploadedAudio) {
        el('recordStatus').textContent = `Audio file selected: ${uploadedAudio.name}`;
        return;
    }
    if (state.recordedAudioBlob) {
        el('recordStatus').textContent = 'Recorded (stored for MedASR)';
        return;
    }
    el('recordStatus').textContent = 'Not recording';
}

// ---------------------------------------------------------------------------
// Randomizer
// ---------------------------------------------------------------------------

function fillFromRandomProfile(): void {
    const profile = generateRandomProfile();

    inputEl('age').value = String(profile.age);
    inputEl('gestWeeks').value = String(profile.gestWeeks);
    inputEl('gravidity').value = String(profile.gravidity);
    inputEl('parity').value = String(profile.parity);
    selectEl('bmiGroup').value = profile.bmiGroup;
    inputEl('systolicBp').value = String(profile.systolicBp);
    inputEl('diastolicBp').value = String(profile.diastolicBp);
    inputEl('fastingGlucose').value = String(profile.fastingGlucose);
    inputEl('tempF').value = String(profile.tempF);
    inputEl('restingHr').value = String(profile.restingHr);
    inputEl('spo2').value = String(profile.spo2);
    inputEl('hrv').value = String(profile.hrv);
    inputEl('headache').checked = profile.headache;
    inputEl('visionChanges').checked = profile.visionChanges;
    inputEl('decreasedFetalMovement').checked = profile.decreasedFetalMovement;
    inputEl('knownConditions').value = profile.knownConditions;
    inputEl('medications').value = profile.medications;
    textareaEl('freeText').value = profile.freeText;
    inputEl('randomProfileMode').value = profile.profileModeName;

    updateStatus(`Randomized using profile mode: ${profile.profileModeName}`);
    updateValidation('');
    updateRequiredGate();
    updateModeIndicators();
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

async function startRecording(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
        updateValidation('Browser microphone access is not available.');
        return;
    }
    try {
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: true,
        });
        const mimeType =
            AUDIO_MIME_CANDIDATES.find((m) =>
                MediaRecorder.isTypeSupported(m),
            ) || '';
        state.mediaChunks = [];
        state.mediaRecorder = mimeType
            ? new MediaRecorder(state.mediaStream, { mimeType })
            : new MediaRecorder(state.mediaStream);

        state.mediaRecorder.ondataavailable = (event: BlobEvent) => {
            if (event.data?.size > 0) {
                state.mediaChunks.push(event.data);
            }
        };

        state.mediaRecorder.onstop = async () => {
            const blobType = mimeType || 'audio/webm';
            const rawBlob = new Blob(state.mediaChunks, { type: blobType });
            const wavBlob = await convertToWavIfPossible(rawBlob);
            state.recordedAudioBlob = wavBlob || rawBlob;

            if (state.mediaStream) {
                state.mediaStream.getTracks().forEach((track) => track.stop());
            }

            state.mediaStream = null;
            state.mediaRecorder = null;
            state.mediaChunks = [];
            (el('recordStart') as HTMLButtonElement).disabled = false;
            (el('recordStop') as HTMLButtonElement).disabled = true;
            el('recordStatus').textContent = wavBlob
                ? 'Recorded (stored as WAV for MedASR)'
                : 'Recorded (original codec)';
            updateModeIndicators();
        };

        state.mediaRecorder.start();
        state.recordedAudioBlob = null;
        updateModeIndicators();
        (el('recordStart') as HTMLButtonElement).disabled = true;
        (el('recordStop') as HTMLButtonElement).disabled = false;
        el('recordStatus').textContent = 'Recording...';
        updateStatus('Microphone recording started.');
    } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        updateValidation(`Microphone error: ${msg}`);
    }
}

function stopRecording(): void {
    if (!state.mediaRecorder) return;
    state.mediaRecorder.stop();
    updateStatus('Recording stopped, processing audio...');
}

// ---------------------------------------------------------------------------
// Download run log
// ---------------------------------------------------------------------------

function downloadRunLog(): void {
    if (!state.lastRun) return;
    const fileName = `momnitrix-run-${Date.now()}.json`;
    const blob = new Blob([stringifyJson(state.lastRun)], {
        type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Panels / sheets
// ---------------------------------------------------------------------------

function openInputValuesPanel(): void {
    const panel = el('inputValuesPanel');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.hidden = false;
    panel.classList.add('is-open');
    backdrop.hidden = false;
    backdrop.classList.add('is-open');
    el('openVitalsBtn').setAttribute('aria-expanded', 'true');
    document.body.classList.add('sheet-open');
    const firstField = document.getElementById('age');
    if (firstField) firstField.focus();
}

function openInteractionPanel(): void {
    const panel = document.getElementById('interactionSheet');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add('sheet-open');
    // Force reflow for CSS transition
    void panel.offsetWidth;
    panel.classList.add('is-open');
    backdrop.classList.add('is-open');
}

function closeInteractionPanel(): void {
    const panel = document.getElementById('interactionSheet');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.classList.remove('is-open');
    backdrop.classList.remove('is-open');
    document.body.classList.remove('sheet-open');
    setTimeout(() => {
        if (!panel.classList.contains('is-open')) {
            panel.hidden = true;
            backdrop.hidden = true;
        }
    }, 250);
}

function closeInputValuesPanel(): void {
    const panel = el('inputValuesPanel');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.classList.remove('is-open');
    backdrop.classList.remove('is-open');
    document.body.classList.remove('sheet-open');
    el('openVitalsBtn').setAttribute('aria-expanded', 'false');
    el('openVitalsBtn').focus();
    setTimeout(() => {
        if (!panel.classList.contains('is-open')) {
            panel.hidden = true;
            backdrop.hidden = true;
        }
    }, 250);
}

function openResultsPanel(): void {
    const panel = el('resultsWrapper');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add('sheet-open');
    // Force reflow for CSS transition
    void panel.offsetWidth;
    panel.classList.add('is-open');
    backdrop.classList.add('is-open');
}

function closeResultsPanel(): void {
    const panel = el('resultsWrapper');
    const backdrop = el('inputValuesBackdrop');
    if (!panel || !backdrop) return;
    panel.classList.remove('is-open');
    backdrop.classList.remove('is-open');
    document.body.classList.remove('sheet-open');
    setTimeout(() => {
        if (!panel.classList.contains('is-open')) {
            panel.hidden = true;
            backdrop.hidden = true;
        }
    }, 250);
}

// ---------------------------------------------------------------------------
// Watch import placeholder
// ---------------------------------------------------------------------------

function importWatchPlaceholder(): void {
    updateStatus(
        'Watch import placeholder is active. Manual values remain enabled for prototype testing.',
    );
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

export function wireEvents(): void {
    el('modeTextBtn').addEventListener('click', () => {
        setActiveMode('text');
        openInteractionPanel();
    });
    el('modeVoiceBtn').addEventListener('click', () => {
        setActiveMode('voice');
        openInteractionPanel();
    });
    el('modeImageBtn').addEventListener('click', () => {
        setActiveMode('image');
        openInteractionPanel();
    });

    el('openVitalsBtn').addEventListener('click', openInputValuesPanel);
    el('closeVitalsBtn').addEventListener('click', closeInputValuesPanel);

    const closeResultsBtn = document.getElementById('closeResultsBtn');
    if (closeResultsBtn) {
        closeResultsBtn.addEventListener('click', closeResultsPanel);
    }
    const closeInteractionBtn = document.getElementById('closeInteractionBtn');
    if (closeInteractionBtn) {
        closeInteractionBtn.addEventListener('click', closeInteractionPanel);
    }

    el('inputValuesBackdrop').addEventListener('click', () => {
        closeInputValuesPanel();
        closeResultsPanel();
        closeInteractionPanel();
    });

    el('prevCardBtn').addEventListener('click', () => shiftResultCard(-1));
    el('nextCardBtn').addEventListener('click', () => shiftResultCard(1));
    el('importWatchBtn').addEventListener('click', importWatchPlaceholder);

    inputEl('woundImage').addEventListener('change', () => {
        setPreview(
            inputEl('woundImage'),
            document.getElementById('woundPreview') as HTMLImageElement,
        );
        updateModeIndicators();
    });
    inputEl('skinImage').addEventListener('change', () => {
        setPreview(
            inputEl('skinImage'),
            document.getElementById('skinPreview') as HTMLImageElement,
        );
        updateModeIndicators();
    });
    el('randomizeBtn').addEventListener('click', fillFromRandomProfile);
    el('submitBtn').addEventListener('click', () => void submitRequest());
    el('clearBtn').addEventListener('click', resetToDefaults);
    el('downloadLogBtn').addEventListener('click', downloadRunLog);
    el('recordStart').addEventListener('click', () => void startRecording());
    el('recordStop').addEventListener('click', stopRecording);
    el('checkHealthBtn').addEventListener('click', () =>
        void checkBackendHealth(),
    );
    inputEl('audioFile').addEventListener('change', () => {
        if (inputEl('audioFile').files?.[0]) {
            state.recordedAudioBlob = null;
        }
        updateAudioStatus();
        updateModeIndicators();
    });

    el('clearTextBtn').addEventListener('click', () => {
        textareaEl('freeText').value = '';
        updateModeIndicators();
    });

    el('clearVoiceBtn').addEventListener('click', () => {
        inputEl('audioFile').value = '';
        state.recordedAudioBlob = null;
        updateAudioStatus();
        updateModeIndicators();
    });

    el('clearImageBtn').addEventListener('click', () => {
        inputEl('woundImage').value = '';
        inputEl('skinImage').value = '';
        setPreview(inputEl('woundImage'), el('woundPreview') as HTMLImageElement);
        setPreview(inputEl('skinImage'), el('skinPreview') as HTMLImageElement);
        updateModeIndicators();
    });

    el('freeText').addEventListener('input', updateModeIndicators);

    const watchedFields = [
        ...REQUIRED_RULES.map((x) => x.id),
        ...OPTIONAL_NUMERIC_RULES.map((x) => x.id),
        'backendUrl',
    ];
    for (const id of watchedFields) {
        el(id).addEventListener('input', updateRequiredGate);
    }

    // Final check for indicators on boot
    updateModeIndicators();
    setActiveMode('text');

    document.addEventListener('keydown', (event: KeyboardEvent) => {
        const inputPanel = document.getElementById('inputValuesPanel');
        const resultsPanel = document.getElementById('resultsWrapper');
        const interactionPanel = document.getElementById('interactionSheet');
        if (event.key !== 'Escape') return;
        if (inputPanel && !inputPanel.hidden) closeInputValuesPanel();
        if (resultsPanel && !resultsPanel.hidden) closeResultsPanel();
        if (interactionPanel && !interactionPanel.hidden) closeInteractionPanel();
    });
}
