/**
 * Formatter tests.
 */

import { describe, it, expect } from 'vitest';
import {
    escapeHtml,
    splitSectionedGuidance,
    toCleanList,
    toParagraph,
    buildImageEvidenceLines,
    formatVisitSummaryHtml,
    formatSpecialistSummaryHtml,
    formatTimelineEntryText,
} from '../domain/formatters';

describe('escapeHtml', () => {
    it('escapes special characters', () => {
        expect(escapeHtml('<b>"hi" & \'there\'</b>')).toBe(
            '&lt;b&gt;&quot;hi&quot; &amp; &#39;there&#39;&lt;/b&gt;',
        );
    });
    it('handles null/undefined', () => {
        expect(escapeHtml(null)).toBe('');
        expect(escapeHtml(undefined)).toBe('');
    });
});

describe('splitSectionedGuidance', () => {
    it('parses all sections', () => {
        const text = [
            'RISK LEVEL: HIGH',
            'CLINICAL REASONING: Something important',
            'More reasoning',
            'POTENTIAL COMPLICATIONS: Comp1',
            'RECOMMENDED ACTIONS: Action1',
            '- Action2',
            'WARNING SIGNS: Warn1',
        ].join('\n');

        const result = splitSectionedGuidance(text);
        expect(result.risk_level).toBe('HIGH');
        expect(result.clinical_reasoning).toEqual([
            'Something important',
            'More reasoning',
        ]);
        expect(result.potential_complications).toEqual(['Comp1']);
        expect(result.recommended_actions).toEqual(['Action1', '- Action2']);
        expect(result.warning_signs).toEqual(['Warn1']);
    });

    it('handles empty input', () => {
        const result = splitSectionedGuidance('');
        expect(result.risk_level).toBe('');
        expect(result.clinical_reasoning).toEqual([]);
    });

    it('handles null input', () => {
        const result = splitSectionedGuidance(null);
        expect(result.risk_level).toBe('');
    });
});

describe('toCleanList', () => {
    it('strips bullets and deduplicates', () => {
        const items = ['- foo', '* bar', '• baz', '- foo'];
        expect(toCleanList(items)).toEqual(['foo', 'bar', 'baz']);
    });
    it('filters empty strings', () => {
        expect(toCleanList(['', '  ', 'ok'])).toEqual(['ok']);
    });
    it('handles non-array', () => {
        expect(toCleanList(null)).toEqual([]);
        expect(toCleanList(undefined)).toEqual([]);
    });
});

describe('toParagraph', () => {
    it('joins cleaned items into single string', () => {
        expect(toParagraph(['- a', '- b'])).toBe('a b');
    });
});

describe('buildImageEvidenceLines', () => {
    it('reports elevated wound risk', () => {
        const lines = buildImageEvidenceLines({
            wound_scores: { urgency: 0.8, infection_risk: 0.9, erythema: 0.3 },
        });
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('elevated risk');
    });

    it('reports no high-risk for low values', () => {
        const lines = buildImageEvidenceLines({
            wound_scores: { urgency: 0.2, infection_risk: 0.3, erythema: 0.1 },
        });
        expect(lines[0]).toContain('no high-risk');
    });

    it('includes skin findings', () => {
        const lines = buildImageEvidenceLines({
            skin_top3: [
                { condition: 'eczema', score: 0.85 },
                { condition: 'psoriasis', score: 0.45 },
            ],
        });
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('eczema');
        expect(lines[0]).toContain('psoriasis');
    });

    it('returns empty for null specialist', () => {
        expect(buildImageEvidenceLines(null)).toEqual([]);
    });
});

describe('formatVisitSummaryHtml', () => {
    it('renders bullets as list', () => {
        const html = formatVisitSummaryHtml('- item1\n- item2');
        expect(html).toContain('<ul');
        expect(html).toContain('item1');
        expect(html).toContain('item2');
    });
    it('wraps non-bullet text in paragraph', () => {
        const html = formatVisitSummaryHtml('just plain text');
        expect(html).toContain('<p>');
    });
    it('handles empty input', () => {
        expect(formatVisitSummaryHtml('')).toBe('No visit summary generated.');
    });
});

describe('formatSpecialistSummaryHtml', () => {
    it('shows transcript chip', () => {
        const html = formatSpecialistSummaryHtml({ transcript: 'Hello world' });
        expect(html).toContain('What the app heard');
    });
    it('returns empty message for null', () => {
        expect(formatSpecialistSummaryHtml(null)).toContain('No specialist');
    });
});

describe('formatTimelineEntryText', () => {
    it('includes label and details', () => {
        const text = formatTimelineEntryText('triage.final', {
            risk_level: 'high',
            timestamp: '2025-01-01T00:00:00Z',
        });
        expect(text).toContain('Final Response');
        expect(text).toContain('risk=high');
    });
    it('uses event name as fallback label', () => {
        const text = formatTimelineEntryText('unknown.event', {});
        expect(text).toContain('unknown.event');
    });
});
