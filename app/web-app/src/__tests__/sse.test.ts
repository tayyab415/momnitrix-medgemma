/**
 * SSE parser tests.
 */

import { describe, it, expect } from 'vitest';
import { parseBlock, createSseEventParser } from '../domain/sse';

describe('parseBlock', () => {
    it('parses event name and JSON data', () => {
        const block = 'event: triage.final\ndata: {"risk_level":"high"}';
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        parseBlock(block, (name, payload) => events.push({ name, payload }));
        expect(events).toHaveLength(1);
        expect(events[0].name).toBe('triage.final');
        expect(events[0].payload).toEqual({ risk_level: 'high' });
    });

    it('defaults event name to "message"', () => {
        const block = 'data: {"key":"val"}';
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        parseBlock(block, (name, payload) => events.push({ name, payload }));
        expect(events[0].name).toBe('message');
    });

    it('ignores comment lines', () => {
        const block = ': this is a comment\nevent: test\ndata: {"ok":true}';
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        parseBlock(block, (name, payload) => events.push({ name, payload }));
        expect(events).toHaveLength(1);
        expect(events[0].name).toBe('test');
    });

    it('handles non-JSON data gracefully', () => {
        const block = 'event: delta\ndata: just some text';
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        parseBlock(block, (name, payload) => events.push({ name, payload }));
        expect(events[0].payload).toEqual({ raw: 'just some text' });
    });

    it('handles multi-line data', () => {
        const block = 'event: test\ndata: {"line1":\ndata: "val"}';
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        parseBlock(block, (name, payload) => events.push({ name, payload }));
        // Multi-line data joined with \n, then parsed as valid JSON
        expect(events[0].payload).toEqual({ line1: 'val' });
    });
});

describe('createSseEventParser', () => {
    it('emits events from chunked input', () => {
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        const parser = createSseEventParser((name, payload) =>
            events.push({ name, payload }),
        );

        parser('event: a\ndata: {"n":1}\n\nevent: b\ndata: {"n":2}\n\n');
        expect(events).toHaveLength(2);
        expect(events[0].name).toBe('a');
        expect(events[1].name).toBe('b');
    });

    it('buffers incomplete blocks', () => {
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        const parser = createSseEventParser((name, payload) =>
            events.push({ name, payload }),
        );

        parser('event: test\n');
        expect(events).toHaveLength(0);

        parser('data: {"ok":true}\n\n');
        expect(events).toHaveLength(1);
        expect(events[0].name).toBe('test');
    });

    it('flushes remaining buffer', () => {
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        const parser = createSseEventParser((name, payload) =>
            events.push({ name, payload }),
        );

        parser('event: last\ndata: {"end":true}');
        expect(events).toHaveLength(0); // no double-newline yet

        parser('', true); // flush
        expect(events).toHaveLength(1);
        expect(events[0].name).toBe('last');
    });

    it('strips \\r characters', () => {
        const events: Array<{ name: string; payload: Record<string, unknown> }> = [];
        const parser = createSseEventParser((name, payload) =>
            events.push({ name, payload }),
        );
        parser('event: x\r\ndata: {"v":1}\r\n\r\n');
        expect(events).toHaveLength(1);
        expect(events[0].name).toBe('x');
    });
});
