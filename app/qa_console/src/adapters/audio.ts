/**
 * Momnitrix QA Console – Audio Adapter
 *
 * Browser-only: uses AudioContext and MediaRecorder APIs.
 */

/**
 * Encode an AudioBuffer into a WAV ArrayBuffer.
 */
export function encodeWavFromAudioBuffer(audioBuffer: AudioBuffer): ArrayBuffer {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const samplesPerChannel = audioBuffer.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = samplesPerChannel * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(offset: number, text: string): void {
        for (let i = 0; i < text.length; i += 1) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
    }

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samplesPerChannel; i += 1) {
        for (let channel = 0; channel < numChannels; channel += 1) {
            const sample = audioBuffer.getChannelData(channel)[i];
            const clamped = Math.max(-1, Math.min(1, sample));
            const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
            view.setInt16(offset, int16, true);
            offset += 2;
        }
    }
    return buffer;
}

/**
 * Try to convert a Blob to WAV format using the Web Audio API.
 * Returns null if conversion fails or AudioContext is unavailable.
 */
export async function convertToWavIfPossible(
    blob: Blob,
): Promise<Blob | null> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const AudioCtor = (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtor) return null;

    try {
        const audioContext = new AudioCtor() as AudioContext;
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        await audioContext.close();
        const wavBuffer = encodeWavFromAudioBuffer(audioBuffer);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    } catch {
        return null;
    }
}
