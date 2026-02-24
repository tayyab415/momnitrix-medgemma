/**
 * Momnitrix QA Console – SSE Streaming Adapter
 *
 * Browser-only: reads a fetch Response body as a text stream and pipes
 * chunks into the SSE parser.
 */

/**
 * Consume a streaming fetch response body and pipe decoded text chunks
 * to the given chunk handler.
 *
 * @param response – a fetch Response with a readable body
 * @param onChunk  – callback receiving `(text, flush)` for each decoded chunk
 */
export async function streamSseFromFetch(
    response: Response,
    onChunk: (text: string, flush: boolean) => void,
): Promise<void> {
    if (!response.body) {
        throw new Error('Streaming response body is missing.');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        onChunk(decoder.decode(value, { stream: true }), false);
    }
    onChunk('', true);
}
