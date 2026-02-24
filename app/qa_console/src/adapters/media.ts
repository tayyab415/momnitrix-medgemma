/**
 * Momnitrix QA Console – Media Attachment Adapter
 *
 * Browser-only: reads File inputs for wound/skin images and audio.
 */

import type { TriageRequest } from '../domain/types';
import { base64FromBlob } from './base64';
import { convertToWavIfPossible } from './audio';

export interface MediaFiles {
    woundImage: File | null;
    skinImage: File | null;
    uploadedAudio: File | null;
    recordedAudioBlob: Blob | null;
}

export interface AttachMediaResult {
    audioStatusMessage: string | null;
}

/**
 * Attach media payloads (images, audio) to the triage request.
 *
 * Mutates `payload.inputs` in place for base64 fields.
 * Returns an optional status message for the audio path.
 */
export async function attachMedia(
    payload: TriageRequest,
    files: MediaFiles,
): Promise<AttachMediaResult> {
    let audioStatusMessage: string | null = null;

    if (files.woundImage) {
        payload.inputs.wound_image_b64 = await base64FromBlob(files.woundImage);
    }

    if (files.skinImage) {
        payload.inputs.skin_image_b64 = await base64FromBlob(files.skinImage);
    }

    if (files.uploadedAudio) {
        const wavBlob = await convertToWavIfPossible(files.uploadedAudio);
        const blobToSend = wavBlob || files.uploadedAudio;
        payload.inputs.audio_b64 = await base64FromBlob(blobToSend);
        audioStatusMessage = wavBlob
            ? 'Uploaded audio converted to WAV for MedASR'
            : `Uploaded audio kept as ${files.uploadedAudio.type || 'original format'}`;
        return { audioStatusMessage };
    }

    if (files.recordedAudioBlob) {
        payload.inputs.audio_b64 = await base64FromBlob(files.recordedAudioBlob);
    }

    return { audioStatusMessage };
}
