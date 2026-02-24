/**
 * Momnitrix QA Console – Base64 Adapter
 *
 * Browser-only: uses FileReader API.
 */

/**
 * Read a Blob as a base64-encoded string (without the data-URL prefix).
 */
export function base64FromBlob(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const value = String(reader.result || '');
            const split = value.split(',');
            resolve(split.length > 1 ? split[1] : value);
        };
        reader.onerror = () =>
            reject(new Error('Failed to read file for base64 conversion.'));
        reader.readAsDataURL(blob);
    });
}
