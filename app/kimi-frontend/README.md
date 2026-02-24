# Momnitrix Frontend - Watch Simulation Console

A modern, responsive frontend for testing the Momnitrix maternal health triage system. Simulates watch sensor inputs through manual entry and randomized clinical profiles.

## Features

### 🎛️ Manual Input Panel
- **Patient Profile**: Age, gestational weeks, obstetric history (G/P), BMI group
- **Vitals**: Blood pressure, glucose, temperature, heart rate, SpO2, HRV
- **All inputs have medically-validated min/max bounds** - no crazy values like 1000 bpm heart rate!

### 🎲 Smart Randomization
Three clinical profiles with realistic value distributions:
- **Normal Monitoring** (50%) - Healthy baseline values
- **Borderline Risk** (35%) - Elevated but non-critical values  
- **Red Flag** (15%) - Critical values requiring urgent attention

### 🎤 Multimodal Inputs
- **Wound Images** → MedSigLIP analysis
- **Skin Images** → Derm Foundation analysis
- **Audio** → MedASR transcription (upload or mic recording)
- **Text Notes** → MedGemma clinical reasoning

### ⚡ Real-time Orchestration
- SSE streaming events from the orchestrator
- Live pipeline timeline showing each model invocation
- Final triage results with risk level and action items

### 🔧 Developer Tools
- Health check button to verify backend status
- Download run logs as JSON for debugging
- Copy-paste backend URL configuration

## Quick Start

```bash
# Navigate to the frontend folder
cd kimi-frontend

# Start a local server (using a port different from other frontends)
python3 -m http.server 8080

# Open in browser
http://localhost:8080
```

## Backend Configuration

The frontend connects to your Modal backend. Set the URL in the configuration panel:

```
Default: https://tayyabkhn343--momnitrix-api-v2-web.modal.run
```

To check if your backend is healthy, click **"Check Health"** - it will show:
- Gemini model status
- Core GPU (MedGemma/MedSigLIP) availability
- Derm service availability
- MedASR service availability

## Input Validation Bounds

All numeric inputs are validated against clinically realistic bounds:

| Field | Min | Max | Unit |
|-------|-----|-----|------|
| Age | 13 | 55 | years |
| Gestational Weeks | 1 | 45 | weeks |
| Systolic BP | 50 | 260 | mmHg |
| Diastolic BP | 30 | 160 | mmHg |
| Fasting Glucose | 1.0 | 40.0 | mmol/L |
| Heart Rate | 20 | 260 | bpm |
| SpO2 | 40 | 100 | % |
| Temperature | 90 | 110 | °F |
| HRV | 0 | 300 | ms |

## API Contract

The frontend sends `POST /v1/triage/stream` with this JSON structure:

```json
{
  "request_id": "kimi-1234567890-123",
  "patient_context": {
    "gestational_weeks": 32,
    "known_conditions": ["chronic_hypertension"],
    "medications": ["labetalol"]
  },
  "vitals": {
    "systolic_bp": 125,
    "diastolic_bp": 78,
    "fasting_glucose_mmol_l": 5.2,
    "hr": 82,
    "spo2": 98,
    "temp_c": 36.9,
    "hrv": 45
  },
  "inputs": {
    "headache": false,
    "vision_changes": false,
    "decreased_fetal_movement": false,
    "free_text": "Patient feeling well...",
    "wound_image_b64": "...base64...",
    "skin_image_b64": "...base64...",
    "audio_b64": "...base64..."
  },
  "metadata": {
    "source": "kimi_frontend",
    "simulator": { ... }
  }
}
```

## File Structure

```
kimi-frontend/
├── index.html      # Main UI markup
├── app.js          # Application logic
├── styles.css      # Styling
└── README.md       # This file
```

## No Dependencies!

This frontend is built with vanilla HTML, CSS, and JavaScript - no build steps, no npm, no frameworks. Just open and use.

## Port Configuration

To avoid conflicts with other frontends (like `qa_console` on port 8000):

```bash
# This frontend defaults to port 8080
python3 -m http.server 8080

# Or use any other available port
python3 -m http.server 3000
python3 -m http.server 5000
```

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari

Requires:
- ES6+ JavaScript support
- Fetch API
- MediaRecorder API (for mic recording)
- FileReader API (for file uploads)
