/**
 * Momnitrix QA Console – Pure Formatters
 *
 * All functions are string → string (or value → string).  No DOM access.
 * Used for rendering final triage results, timeline entries, and parsed
 * model output sections.
 */

import type {
    SectionedGuidance,
    TriageFinalResponse,
    SpecialistOutputs,
    InferenceBreakdownEvent,
} from './types';
import { EVENT_LABELS } from './constants';

// ---------------------------------------------------------------------------
// HTML escaping
// ---------------------------------------------------------------------------

export function escapeHtml(value: string | null | undefined): string {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ---------------------------------------------------------------------------
// Section-based guidance parser
// ---------------------------------------------------------------------------

const SECTION_KEY_MAP: Record<string, keyof SectionedGuidance> = {
    'RISK LEVEL': 'risk_level',
    'CLINICAL REASONING': 'clinical_reasoning',
    'POTENTIAL COMPLICATIONS': 'potential_complications',
    'RECOMMENDED ACTIONS': 'recommended_actions',
    'WARNING SIGNS': 'warning_signs',
};

const SECTION_HEADING_REGEX =
    /^\s*(RISK LEVEL|CLINICAL REASONING|POTENTIAL COMPLICATIONS|RECOMMENDED ACTIONS|WARNING SIGNS)\s*:\s*(.*)$/i;

/**
 * Parse model-generated text into structured guidance sections.
 */
export function splitSectionedGuidance(rawText: string | null | undefined): SectionedGuidance {
    const text = String(rawText || '').replace(/\r/g, '');
    const sections: SectionedGuidance = {
        risk_level: '',
        clinical_reasoning: [],
        potential_complications: [],
        recommended_actions: [],
        warning_signs: [],
    };

    let currentKey: keyof SectionedGuidance | '' = '';

    for (const line of text.split('\n')) {
        const match = line.match(SECTION_HEADING_REGEX);
        if (match) {
            const heading = match[1].toUpperCase();
            currentKey = SECTION_KEY_MAP[heading] || '';
            const tail = (match[2] || '').trim();
            if (tail && currentKey) {
                if (currentKey === 'risk_level') {
                    sections.risk_level = tail;
                } else {
                    (sections[currentKey] as string[]).push(tail);
                }
            }
            continue;
        }

        if (!currentKey) continue;
        const cleaned = line.trim();
        if (!cleaned) continue;
        if (currentKey === 'risk_level') {
            sections.risk_level = cleaned;
        } else {
            (sections[currentKey] as string[]).push(cleaned);
        }
    }

    return sections;
}

// ---------------------------------------------------------------------------
// List utilities
// ---------------------------------------------------------------------------

/**
 * Clean a list of raw lines: strip bullets, deduplicate, trim.
 */
export function toCleanList(lines: unknown): string[] {
    const out: string[] = [];
    const arr = Array.isArray(lines) ? (lines as unknown[]) : [];
    for (const line of arr) {
        const cleaned = String(line || '')
            .replace(/^\s*[-*•]+\s*/, '')
            .replace(/\s+/g, ' ')
            .trim();
        if (!cleaned) continue;
        if (!out.includes(cleaned)) out.push(cleaned);
    }
    return out;
}

/**
 * Join cleaned list items into a paragraph.
 */
export function toParagraph(lines: unknown): string {
    return toCleanList(lines).join(' ');
}

// ---------------------------------------------------------------------------
// Image evidence
// ---------------------------------------------------------------------------

/**
 * Build human-readable evidence lines from specialist image outputs.
 */
export function buildImageEvidenceLines(specialist: SpecialistOutputs | null | undefined): string[] {
    const lines: string[] = [];

    if (specialist?.wound_scores) {
        const wound = specialist.wound_scores;
        const urgency = Number(wound.urgency || 0);
        const infection = Number(wound.infection_risk || 0);
        if (urgency >= 0.6 || infection >= 0.7) {
            lines.push(
                `Wound-image specialist flagged elevated risk (urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}).`,
            );
        } else {
            lines.push(
                `Wound-image specialist reviewed the image with no high-risk threshold crossed (urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}).`,
            );
        }
    }

    if (specialist?.skin_top3 && Array.isArray(specialist.skin_top3) && specialist.skin_top3.length) {
        const labels = specialist.skin_top3
            .slice(0, 3)
            .map(
                (row) =>
                    `${String(row.condition || 'unknown').replace(/_/g, ' ')} (${Number(row.score || 0).toFixed(2)})`,
            );
        lines.push(`Skin specialist top findings: ${labels.join(', ')}.`);
    }

    return lines;
}

// ---------------------------------------------------------------------------
// HTML renderers (return HTML strings, no DOM mutation)
// ---------------------------------------------------------------------------

/**
 * Render the patient guidance card HTML.
 */
export function formatPatientGuidanceHtml(finalPayload: TriageFinalResponse | null | undefined): string {
    const fp = finalPayload || {};
    const sectioned = splitSectionedGuidance(fp.patient_message);
    const reasoning = toParagraph(sectioned.clinical_reasoning);
    const complications = toParagraph(sectioned.potential_complications);
    const actions = toCleanList(sectioned.recommended_actions);
    const warnings = toCleanList(sectioned.warning_signs);
    const specialist = fp.specialist_outputs || {};
    const imageEvidenceLines = buildImageEvidenceLines(specialist);
    const risk = String(fp.risk_level || sectioned.risk_level || '-')
        .toUpperCase()
        .replace('YELLOW', 'MID')
        .replace('GREEN', 'LOW')
        .replace('RED', 'HIGH');

    const hasSections =
        reasoning || complications || actions.length || warnings.length;
    if (!hasSections) {
        return `<p>${escapeHtml(fp.patient_message || 'No patient message generated.')}</p>`;
    }

    const riskClass =
        risk === 'HIGH' ? 'red' : risk === 'MID' ? 'yellow' : 'green';
    const actionsHtml = actions.length
        ? `<ul class="guidance-list">${actions.map((x) => `<li>${escapeHtml(x)}</li>`).join('')}</ul>`
        : `<p class="guidance-empty">No immediate actions listed.</p>`;
    const warningsHtml = warnings.length
        ? `<ul class="guidance-list">${warnings.map((x) => `<li>${escapeHtml(x)}</li>`).join('')}</ul>`
        : `<p class="guidance-empty">No urgent warning signs listed.</p>`;

    return `
    <div class="patient-guidance">
      <div class="guidance-head">
        <span class="guidance-title">For the patient</span>
        <span class="risk-pill ${riskClass}">${escapeHtml(risk)} RISK</span>
      </div>
      <div class="guidance-card">
        <h4>What this means now</h4>
        <p>${escapeHtml(
        reasoning ||
        'Current readings do not show severe immediate danger, but continue close monitoring.',
    )}</p>
      </div>
      <div class="guidance-card">
        <h4>Possible complications if this continues</h4>
        <p>${escapeHtml(
        complications ||
        'Complication risk depends on future trends and symptom progression.',
    )}</p>
      </div>
      ${imageEvidenceLines.length
            ? `<div class="guidance-card">
              <h4>What image analysis found</h4>
              <ul class="guidance-list">${imageEvidenceLines.map((line) => `<li>${escapeHtml(line)}</li>`).join('')}</ul>
            </div>`
            : ''
        }
      <div class="guidance-card">
        <h4>What to do next</h4>
        ${actionsHtml}
      </div>
      <div class="guidance-card warning">
        <h4>Seek urgent care now if</h4>
        ${warningsHtml}
      </div>
    </div>
  `;
}

/**
 * Render visit summary HTML.
 */
export function formatVisitSummaryHtml(summaryText: string | null | undefined): string {
    const text = String(summaryText || '').trim();
    if (!text) return 'No visit summary generated.';
    const lines = text
        .split('\n')
        .map((x) => x.trim())
        .filter(Boolean);
    const bullets = lines
        .filter((line) => line.startsWith('-'))
        .map((line) => line.replace(/^-+\s*/, '').trim());
    if (!bullets.length) {
        return `<p>${escapeHtml(text)}</p>`;
    }
    return `<ul class="visit-list">${bullets.map((line) => `<li>${escapeHtml(line)}</li>`).join('')}</ul>`;
}

/**
 * Render specialist summary HTML.
 */
export function formatSpecialistSummaryHtml(
    specialist: SpecialistOutputs | null | undefined,
): string {
    const parts: string[] = [];

    if (specialist?.transcript) {
        const full = String(specialist.transcript).replace(/\s+/g, ' ').trim();
        const clipped = full.length > 170 ? `${full.slice(0, 170)}...` : full;
        parts.push(`
      <div class="specialist-chip">
        <strong>What the app heard:</strong> ${escapeHtml(clipped)}
      </div>
    `);
    }

    if (specialist?.wound_scores) {
        const wound = specialist.wound_scores;
        const urgency = Number(wound.urgency || 0);
        const infection = Number(wound.infection_risk || 0);
        const erythema = Number(wound.erythema || 0);
        parts.push(
            `<div class="specialist-chip"><strong>Wound model:</strong> urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}, erythema ${erythema.toFixed(2)}.</div>`,
        );
    }

    if (specialist?.skin_top3 || specialist?.skin_scores) {
        if (
            Array.isArray(specialist?.skin_top3) &&
            specialist.skin_top3.length
        ) {
            const labels = specialist.skin_top3
                .slice(0, 3)
                .map(
                    (row) =>
                        `${String(row.condition || 'unknown').replace(/_/g, ' ')} (${Number(row.score || 0).toFixed(2)})`,
                );
            parts.push(
                `<div class="specialist-chip"><strong>Skin model:</strong> ${escapeHtml(labels.join(', '))}.</div>`,
            );
        } else {
            parts.push(
                '<div class="specialist-chip">Skin specialist was used.</div>',
            );
        }
    }

    if (!parts.length) {
        return 'No specialist outputs were used in this run.';
    }
    return `<div class="specialist-stack">${parts.join('')}</div>`;
}

// ---------------------------------------------------------------------------
// Timeline entry text
// ---------------------------------------------------------------------------

/**
 * Format a single SSE timeline entry as a display string.
 */
export function formatTimelineEntryText(
    eventName: string,
    payload: InferenceBreakdownEvent | Record<string, unknown> | null,
): string {
    const p = (payload || {}) as Record<string, unknown>;
    const time =
        typeof p.timestamp === 'string' ? p.timestamp : new Date().toISOString();
    const label = EVENT_LABELS[eventName] || eventName;
    const details: string[] = [];

    if (p.model) details.push(`model=${p.model}`);
    if (p.intent) details.push(`intent=${p.intent}`);
    if (p.prompt_strategy) details.push(`strategy=${p.prompt_strategy}`);
    if (p.risk_level) details.push(`risk=${p.risk_level}`);
    if (p.latency_ms !== undefined) details.push(`latency=${p.latency_ms}ms`);
    if (p.gpu_warmup_ms !== undefined) details.push(`warmup=${p.gpu_warmup_ms}ms`);
    if (p.medgemma_inference_ms !== undefined) {
        details.push(`medgemma_infer=${p.medgemma_inference_ms}ms`);
    }
    if (p.composer_mode) details.push(`composer=${p.composer_mode}`);
    if (p.reason) details.push(`reason=${p.reason}`);
    if (p.error) details.push(`error=${p.error}`);

    return `${time} | ${label}${details.length ? ` | ${details.join(' | ')}` : ''}`;
}

// ---------------------------------------------------------------------------
// JSON stringification
// ---------------------------------------------------------------------------

export function stringifyJson(data: unknown): string {
    return JSON.stringify(data, null, 2);
}
