/**
 * Momnitrix QA Console – SSE Stream Parser
 *
 * Pure functions for server-sent events parsing.  No DOM access.
 */

/**
 * Callback invoked for each parsed SSE event.
 */
export type SseEventHandler = (
    eventName: string,
    payload: Record<string, unknown>,
) => void;

/**
 * Parse a single SSE text block (between double-newlines) and invoke the
 * handler with the parsed event name and payload.
 */
export function parseBlock(block: string, onEvent: SseEventHandler): void {
    const lines = block.split('\n');
    let eventName = 'message';
    const dataLines: string[] = [];

    for (const line of lines) {
        if (!line || line.startsWith(':')) continue;
        if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
            continue;
        }
        if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
        }
    }

    const dataRaw = dataLines.join('\n');
    let payload: Record<string, unknown> = { raw: dataRaw };
    if (dataRaw) {
        try {
            payload = JSON.parse(dataRaw) as Record<string, unknown>;
        } catch {
            payload = { raw: dataRaw };
        }
    }
    onEvent(eventName, payload);
}

/**
 * Create a stateful SSE parser that buffers incoming text chunks and emits
 * complete events via `onEvent`.
 *
 * @returns A function accepting `(chunk: string, flush?: boolean)`.
 *   Call with `flush = true` after the stream ends to emit any remaining
 *   buffered data.
 */
export function createSseEventParser(
    onEvent: SseEventHandler,
): (chunk: string, flush?: boolean) => void {
    let buffer = '';

    return (chunk: string, flush = false): void => {
        buffer += chunk.replace(/\r/g, '');
        while (true) {
            const splitAt = buffer.indexOf('\n\n');
            if (splitAt === -1) break;
            const block = buffer.slice(0, splitAt);
            buffer = buffer.slice(splitAt + 2);
            parseBlock(block, onEvent);
        }
        if (flush && buffer.trim()) {
            parseBlock(buffer, onEvent);
            buffer = '';
        }
    };
}
