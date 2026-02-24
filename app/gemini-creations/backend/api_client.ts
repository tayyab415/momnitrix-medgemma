import { VitalsPayload } from '../frontend/utils/randomizer';

interface ClientOptions {
  apiUrl?: string;
}

export class ModalApiClient {
  private apiUrl: string;

  constructor(options?: ClientOptions) {
    // Defaults to local Next.js proxy or direct Modal URL depending on env
    this.apiUrl = options?.apiUrl || process.env.NEXT_PUBLIC_MODAL_API_URL || '';
  }

  /**
   * Sends the simulated vitals and optional media to the Modal backend.
   * Uses SSE (Server-Sent Events) to stream the response back.
   */
  async streamTriage(
    vitals: VitalsPayload,
    patientContext: any,
    inputs: { [key: string]: any },
    onMessage: (data: any) => void,
    onError: (error: Error) => void,
    onComplete: () => void
  ) {
    if (!this.apiUrl) {
      onError(new Error("Modal API URL is not configured."));
      return;
    }

    try {
      const response = await fetch(`${this.apiUrl}/v1/triage/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          patient_context: patientContext,
          vitals: vitals,
          inputs: inputs
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No readable stream available in response");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          onComplete();
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\\n');
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            if (dataStr) {
              if (dataStr === '[DONE]') {
                continue;
              }
              try {
                const data = JSON.parse(dataStr);
                onMessage(data);
              } catch (e) {
                console.warn("Failed to parse SSE data chunk", dataStr);
              }
            }
          }
        }
      }
    } catch (error: any) {
      onError(error);
    }
  }
}
