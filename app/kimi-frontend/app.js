/**
 * Momnitrix Frontend - Watch Simulation Console
 * Handles manual input, randomization, multimodal uploads, and SSE streaming
 */

// DOM Helpers
const $ = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

// State
const state = {
  isRecording: false,
  mediaRecorder: null,
  mediaStream: null,
  recordedBlob: null,
  selectedFiles: {
    wound: null,
    skin: null,
    audio: null
  },
  runLog: null,
  eventCount: 0
};

// Clinical Profile Definitions for Randomization
const CLINICAL_PROFILES = {
  normal: {
    name: 'Normal Monitoring',
    weight: 0.5,
    ranges: {
      age: [20, 35],
      gestWeeks: [20, 38],
      gravidity: [1, 4],
      parity: [0, 3],
      systolicBp: [105, 125],
      diastolicBp: [65, 80],
      fastingGlucose: [4.0, 5.3],
      tempF: [97.5, 99.0],
      restingHr: [65, 90],
      spo2: [96, 100],
      hrv: [30, 90]
    },
    bmi: ['normal', 'overweight'],
    conditions: ['none'],
    medications: ['prenatal_vitamins', 'none'],
    symptomProb: { headache: 0.1, visionChanges: 0.05, decreasedFetalMovement: 0.05 },
    notes: [
      'Routine check-in. Feeling well overall.',
      'No concerns today. Regular monitoring.',
      'Stable vitals, no new symptoms reported.',
      'Evening check completed. All parameters normal.'
    ]
  },
  borderline: {
    name: 'Borderline Risk',
    weight: 0.35,
    ranges: {
      age: [25, 42],
      gestWeeks: [26, 40],
      gravidity: [2, 6],
      parity: [1, 4],
      systolicBp: [128, 145],
      diastolicBp: [82, 94],
      fastingGlucose: [5.4, 7.5],
      tempF: [98.0, 99.8],
      restingHr: [78, 105],
      spo2: [94, 98],
      hrv: [20, 65]
    },
    bmi: ['overweight', 'obese'],
    conditions: ['gestational_diabetes', 'chronic_hypertension', 'none'],
    medications: ['low_dose_aspirin', 'labetalol', 'prenatal_vitamins'],
    symptomProb: { headache: 0.4, visionChanges: 0.15, decreasedFetalMovement: 0.18 },
    notes: [
      'Mild headache reported. Monitoring for changes.',
      'BP trending higher than baseline. Watching closely.',
      'Patient reports mild discomfort. Non-urgent review.',
      'Some swelling noted. Advised to rest and monitor.'
    ]
  },
  redFlag: {
    name: 'Red Flag',
    weight: 0.15,
    ranges: {
      age: [18, 45],
      gestWeeks: [28, 41],
      gravidity: [2, 8],
      parity: [1, 5],
      systolicBp: [150, 175],
      diastolicBp: [96, 115],
      fastingGlucose: [8.0, 16.0],
      tempF: [99.0, 102.0],
      restingHr: [90, 130],
      spo2: [92, 97],
      hrv: [12, 50]
    },
    bmi: ['overweight', 'obese'],
    conditions: ['severe_preeclampsia', 'gestational_diabetes', 'chronic_hypertension'],
    medications: ['insulin', 'labetalol', 'methyldopa'],
    symptomProb: { headache: 0.7, visionChanges: 0.45, decreasedFetalMovement: 0.35 },
    notes: [
      'Severe headache with visual disturbances. Urgent review needed.',
      'BP critically elevated. Possible pre-eclampsia.',
      'Patient reports decreased fetal movement. Immediate attention.',
      'Multiple red flag symptoms present. High priority.'
    ]
  }
};

// Utility Functions
const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const randomFloat = (min, max, decimals = 1) => {
  const val = Math.random() * (max - min) + min;
  return Number(val.toFixed(decimals));
};
const randomBool = (prob) => Math.random() < prob;
const pick = (arr) => arr[randomInt(0, arr.length - 1)];
const weightedPick = (profiles) => {
  const total = Object.values(profiles).reduce((sum, p) => sum + p.weight, 0);
  let rand = Math.random() * total;
  for (const [key, profile] of Object.entries(profiles)) {
    rand -= profile.weight;
    if (rand <= 0) return { key, ...profile };
  }
  return { key: 'normal', ...profiles.normal };
};

const fahrenheitToCelsius = (f) => Number(((f - 32) * 5 / 9).toFixed(2));

const showStatus = (message, type = 'info') => {
  const container = $('statusMessages');
  const el = document.createElement('div');
  el.className = `status-message status-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 5000);
};

const clearStatus = () => {
  $('statusMessages').innerHTML = '';
};

// File Handling
const handleFileSelect = (inputId, previewId, containerId, type) => {
  const input = $(inputId);
  const preview = $(previewId);
  const container = $(containerId);
  
  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    state.selectedFiles[type] = file;
    
    if (type === 'audio') {
      $('audioFileName').textContent = file.name;
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.src = ev.target.result;
      container.classList.add('has-preview');
    };
    reader.readAsDataURL(file);
  });
};

const clearFile = (inputId) => {
  const input = $(inputId);
  const type = inputId === 'woundImage' ? 'wound' : 'skin';
  input.value = '';
  state.selectedFiles[type] = null;
  
  if (type === 'wound') {
    $('woundPreview').src = '';
    $('woundPreviewContainer').classList.remove('has-preview');
  } else {
    $('skinPreview').src = '';
    $('skinPreviewContainer').classList.remove('has-preview');
  }
};

// Audio Recording
const initAudioRecording = () => {
  const startBtn = $('recordStartBtn');
  const stopBtn = $('recordStopBtn');
  const status = $('recordStatus');
  
  startBtn.addEventListener('click', async () => {
    try {
      state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 
                       MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
      
      state.mediaRecorder = new MediaRecorder(state.mediaStream, mimeType ? { mimeType } : {});
      state.recordedChunks = [];
      
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
      };
      
      state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: mimeType || 'audio/webm' });
        state.recordedBlob = blob;
        state.selectedFiles.audio = blob;
        $('audioFileName').textContent = 'Recorded audio';
        status.textContent = 'Recorded';
        status.classList.add('recorded');
        
        state.mediaStream.getTracks().forEach(t => t.stop());
        state.mediaRecorder = null;
        state.mediaStream = null;
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
      };
      
      state.mediaRecorder.start();
      state.isRecording = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      status.textContent = 'Recording...';
      status.classList.add('recording');
      
    } catch (err) {
      showStatus('Microphone access denied or unavailable', 'error');
    }
  });
  
  stopBtn.addEventListener('click', () => {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
      state.isRecording = false;
      status.classList.remove('recording');
    }
  });
};

// Randomization
const randomizeProfile = () => {
  const profile = weightedPick(CLINICAL_PROFILES);
  const r = profile.ranges;
  
  // Set values
  $('age').value = randomInt(r.age[0], r.age[1]);
  $('gestWeeks').value = randomInt(r.gestWeeks[0], r.gestWeeks[1]);
  $('gravidity').value = randomInt(r.gravidity[0], r.gravidity[1]);
  
  // Parity should not exceed gravidity
  const gravidity = parseInt($('gravidity').value);
  $('parity').value = randomInt(0, Math.min(r.parity[1], gravidity));
  
  $('bmiGroup').value = pick(profile.bmi);
  
  // Vitals
  $('systolicBp').value = randomInt(r.systolicBp[0], r.systolicBp[1]);
  let diastolic = randomInt(r.diastolicBp[0], r.diastolicBp[1]);
  const systolic = parseInt($('systolicBp').value);
  if (diastolic >= systolic) diastolic = systolic - randomInt(10, 25);
  $('diastolicBp').value = Math.max(30, diastolic);
  
  $('fastingGlucose').value = randomFloat(r.fastingGlucose[0], r.fastingGlucose[1]);
  $('tempF').value = randomFloat(r.tempF[0], r.tempF[1]);
  $('restingHr').value = randomInt(r.restingHr[0], r.restingHr[1]);
  $('spo2').value = randomInt(r.spo2[0], r.spo2[1]);
  $('hrv').value = randomFloat(r.hrv[0], r.hrv[1]);
  
  // Symptoms
  $('headache').checked = randomBool(profile.symptomProb.headache);
  $('visionChanges').checked = randomBool(profile.symptomProb.visionChanges);
  $('decreasedFetalMovement').checked = randomBool(profile.symptomProb.decreasedFetalMovement);
  
  // Conditions & Medications
  const condition = pick(profile.conditions);
  $('knownConditions').value = condition === 'none' ? '' : condition;
  
  const medication = pick(profile.medications);
  $('medications').value = medication === 'none' ? '' : medication;
  
  // Notes
  $('freeText').value = pick(profile.notes);
  
  showStatus(`Randomized: ${profile.name} profile`, 'success');
};

// Validation
const validateInputs = () => {
  const errors = [];
  const fields = [
    { id: 'age', name: 'Age', min: 13, max: 55 },
    { id: 'gestWeeks', name: 'Gestational Weeks', min: 1, max: 45 },
    { id: 'gravidity', name: 'Gravidity', min: 1, max: 12 },
    { id: 'parity', name: 'Parity', min: 0, max: 12 },
    { id: 'systolicBp', name: 'Systolic BP', min: 50, max: 260 },
    { id: 'diastolicBp', name: 'Diastolic BP', min: 30, max: 160 },
    { id: 'restingHr', name: 'Heart Rate', min: 20, max: 260 },
    { id: 'spo2', name: 'SpO2', min: 40, max: 100 }
  ];
  
  fields.forEach(f => {
    const val = parseFloat($(f.id).value);
    if (isNaN(val) || val < f.min || val > f.max) {
      errors.push(`${f.name} must be between ${f.min} and ${f.max}`);
    }
  });
  
  const systolic = parseInt($('systolicBp').value);
  const diastolic = parseInt($('diastolicBp').value);
  if (systolic <= diastolic) {
    errors.push('Systolic BP must be greater than Diastolic BP');
  }
  
  const glucose = parseFloat($('fastingGlucose').value);
  if (isNaN(glucose) || glucose < 1 || glucose > 40) {
    errors.push('Fasting Glucose must be between 1.0 and 40.0 mmol/L');
  }
  
  const tempF = parseFloat($('tempF').value);
  if (isNaN(tempF) || tempF < 90 || tempF > 110) {
    errors.push('Temperature must be between 90°F and 110°F');
  }
  
  const hrv = parseFloat($('hrv').value);
  if (isNaN(hrv) || hrv < 0 || hrv > 300) {
    errors.push('HRV must be between 0 and 300 ms');
  }
  
  return errors;
};

// API Communication
const checkHealth = async () => {
  const url = $('backendUrl').value.trim();
  if (!url) {
    showStatus('Please enter a backend URL', 'error');
    return;
  }
  
  try {
    const resp = await fetch(`${url.replace(/\/$/, '')}/health`);
    const data = await resp.json();
    
    const status = $('healthStatus');
    if (data.status === 'ok') {
      status.innerHTML = `● Healthy | Gemini: ${data.gemini_model || 'N/A'} | Core: ${data.core_gpu_configured ? '✓' : '✗'} | Derm: ${data.derm_configured ? '✓' : '✗'} | ASR: ${data.medasr_configured ? '✓' : '✗'}`;
      status.className = 'status-indicator status-ok';
      showStatus('Backend is healthy', 'success');
    } else {
      status.textContent = '● Unhealthy';
      status.className = 'status-indicator status-error';
    }
  } catch (err) {
    $('healthStatus').textContent = '● Unreachable';
    $('healthStatus').className = 'status-indicator status-error';
    showStatus('Failed to connect to backend', 'error');
  }
};

const fileToBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => {
    const base64 = reader.result.split(',')[1];
    resolve(base64);
  };
  reader.onerror = reject;
  reader.readAsDataURL(file);
});

const buildPayload = async () => {
  const knownConditions = $('knownConditions').value
    .split(',').map(s => s.trim()).filter(Boolean);
  const medications = $('medications').value
    .split(',').map(s => s.trim()).filter(Boolean);
  
  const tempF = parseFloat($('tempF').value);
  
  const payload = {
    request_id: $('requestId').value.trim() || `kimi-${Date.now()}-${randomInt(100, 999)}`,
    patient_context: {
      gestational_weeks: parseInt($('gestWeeks').value),
      known_conditions: knownConditions,
      medications: medications
    },
    vitals: {
      systolic_bp: parseInt($('systolicBp').value),
      diastolic_bp: parseInt($('diastolicBp').value),
      fasting_glucose_mmol_l: parseFloat($('fastingGlucose').value),
      hr: parseInt($('restingHr').value),
      spo2: parseInt($('spo2').value),
      temp_c: fahrenheitToCelsius(tempF),
      hrv: parseFloat($('hrv').value)
    },
    inputs: {
      headache: $('headache').checked,
      vision_changes: $('visionChanges').checked,
      decreased_fetal_movement: $('decreasedFetalMovement').checked,
      free_text: $('freeText').value.trim() || null,
      wound_image_b64: null,
      skin_image_b64: null,
      audio_b64: null
    },
    metadata: {
      source: 'kimi_frontend',
      composer_mode: 'medgemma_first',
      medgemma_output_style: 'notebook',
      simulator: {
        age_years: parseInt($('age').value),
        bmi_group: $('bmiGroup').value,
        gravidity: parseInt($('gravidity').value),
        parity: parseInt($('parity').value),
        temp_input_unit: 'degF'
      }
    }
  };
  
  // Attach files
  if (state.selectedFiles.wound) {
    payload.inputs.wound_image_b64 = await fileToBase64(state.selectedFiles.wound);
  }
  if (state.selectedFiles.skin) {
    payload.inputs.skin_image_b64 = await fileToBase64(state.selectedFiles.skin);
  }
  if (state.selectedFiles.audio) {
    payload.inputs.audio_b64 = await fileToBase64(state.selectedFiles.audio);
  }
  
  return payload;
};

// SSE Parsing
const createSSEParser = (onEvent) => {
  let buffer = '';
  return (chunk, isFinal = false) => {
    buffer += chunk.replace(/\r/g, '');
    
    while (true) {
      const idx = buffer.indexOf('\n\n');
      if (idx === -1) break;
      
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      parseSSEBlock(block, onEvent);
    }
    
    if (isFinal && buffer.trim()) {
      parseSSEBlock(buffer, onEvent);
    }
  };
};

const parseSSEBlock = (block, onEvent) => {
  const lines = block.split('\n');
  let eventName = 'message';
  const dataLines = [];
  
  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    }
  }
  
  const dataRaw = dataLines.join('\n');
  let payload = { raw: dataRaw };
  if (dataRaw) {
    try {
      payload = JSON.parse(dataRaw);
    } catch (e) {}
  }
  onEvent(eventName, payload);
};

// UI Updates
const addTimelineEvent = (eventName, payload) => {
  const timeline = $('eventTimeline');
  
  // Remove empty state
  if (state.eventCount === 0) {
    timeline.innerHTML = '';
  }
  state.eventCount++;
  
  const item = document.createElement('div');
  item.className = 'timeline-item';
  
  const time = payload.timestamp ? new Date(payload.timestamp).toLocaleTimeString() : 
               new Date().toLocaleTimeString();
  
  let details = '';
  if (payload.model) details += ` model=${payload.model}`;
  if (payload.risk_level) details += ` risk=${payload.risk_level}`;
  if (payload.latency_ms !== undefined) details += ` ${payload.latency_ms}ms`;
  if (payload.gpu_warmup_ms !== undefined) details += ` warmup=${payload.gpu_warmup_ms}ms`;
  if (payload.medgemma_inference_ms !== undefined) details += ` infer=${payload.medgemma_inference_ms}ms`;
  
  item.innerHTML = `
    <span class="timeline-time">${time}</span>
    <span class="timeline-event">${eventName}</span>
    <span class="timeline-details">${details}</span>
  `;
  
  timeline.appendChild(item);
  item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

const displayResult = (payload) => {
  const container = $('resultContent');
  const risk = payload.risk_level || 'unknown';
  const floor = payload.policy_floor || 'unknown';
  
  const riskClass = `risk-${risk}`;
  const floorClass = `risk-${floor}`;
  
  container.innerHTML = `
    <div class="result-grid">
      <div class="result-item">
        <span class="result-label">Risk Level</span>
        <span class="result-value ${riskClass}">${risk.toUpperCase()}</span>
      </div>
      <div class="result-item">
        <span class="result-label">Policy Floor</span>
        <span class="result-value ${floorClass}">${floor.toUpperCase()}</span>
      </div>
    </div>
    <div class="result-section">
      <h4>Patient Message</h4>
      <p>${payload.patient_message || 'N/A'}</p>
    </div>
    <div class="result-section">
      <h4>Action Items</h4>
      <ul>${(payload.action_items || []).map(i => `<li>${i}</li>`).join('')}</ul>
    </div>
  `;
  
  $('finalJson').value = JSON.stringify(payload, null, 2);
  $('downloadLogBtn').disabled = false;
};

// Submission
const submitTriage = async () => {
  clearStatus();
  
  const errors = validateInputs();
  if (errors.length > 0) {
    errors.forEach(e => showStatus(e, 'error'));
    return;
  }
  
  const baseUrl = $('backendUrl').value.trim().replace(/\/$/, '');
  if (!baseUrl) {
    showStatus('Please enter a backend URL', 'error');
    return;
  }
  
  const submitBtn = $('submitBtn');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing...';
  
  try {
    const payload = await buildPayload();
    
    // Reset timeline
    state.eventCount = 0;
    $('eventTimeline').innerHTML = '<div class="timeline-empty">Waiting for events...</div>';
    $('resultContent').innerHTML = '<div class="result-empty">Processing...</div>';
    $('finalJson').value = '';
    $('downloadLogBtn').disabled = true;
    
    // Store for logging
    state.runLog = {
      started_at: new Date().toISOString(),
      backend_url: baseUrl,
      request: payload,
      events: [],
      response: null
    };
    
    showStatus('Sending request to orchestrator...', 'info');
    
    const response = await fetch(`${baseUrl}/v1/triage/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const parser = createSSEParser((eventName, eventPayload) => {
      state.runLog.events.push({ event: eventName, payload: eventPayload });
      addTimelineEvent(eventName, eventPayload);
      
      if (eventName === 'triage.final') {
        state.runLog.response = eventPayload;
        state.runLog.completed_at = new Date().toISOString();
        displayResult(eventPayload);
        showStatus('Triage complete!', 'success');
      }
      
      if (eventName === 'triage.error') {
        showStatus(`Error: ${eventPayload.error || 'Unknown error'}`, 'error');
      }
    });
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      parser(decoder.decode(value, { stream: true }));
    }
    parser('', true);
    
  } catch (err) {
    showStatus(`Request failed: ${err.message}`, 'error');
    $('resultContent').innerHTML = `<div class="result-empty error">Error: ${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Triage';
  }
};

// Clear All
const clearAll = () => {
  // Reset to defaults
  $('age').value = 28;
  $('gestWeeks').value = 32;
  $('gravidity').value = 2;
  $('parity').value = 1;
  $('bmiGroup').value = 'normal';
  $('knownConditions').value = '';
  $('medications').value = '';
  
  $('systolicBp').value = 125;
  $('diastolicBp').value = 78;
  $('fastingGlucose').value = 5.2;
  $('tempF').value = 98.4;
  $('restingHr').value = 82;
  $('spo2').value = 98;
  $('hrv').value = 45;
  
  $('headache').checked = false;
  $('visionChanges').checked = false;
  $('decreasedFetalMovement').checked = false;
  $('freeText').value = '';
  
  clearFile('woundImage');
  clearFile('skinImage');
  $('audioFile').value = '';
  $('audioFileName').textContent = '';
  state.selectedFiles.audio = null;
  state.recordedBlob = null;
  $('recordStatus').textContent = 'Ready';
  $('recordStatus').className = 'record-status';
  
  state.eventCount = 0;
  $('eventTimeline').innerHTML = '<div class="timeline-empty">Submit a triage request to see the orchestration pipeline in action.</div>';
  $('resultContent').innerHTML = '<div class="result-empty">Waiting for triage completion...</div>';
  $('finalJson').value = '';
  $('downloadLogBtn').disabled = true;
  
  clearStatus();
  showStatus('Form cleared', 'info');
};

// Download Log
const downloadLog = () => {
  if (!state.runLog) return;
  
  const blob = new Blob([JSON.stringify(state.runLog, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `momnitrix-run-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

// Initialization
const init = () => {
  // File inputs
  handleFileSelect('woundImage', 'woundPreview', 'woundPreviewContainer', 'wound');
  handleFileSelect('skinImage', 'skinPreview', 'skinPreviewContainer', 'skin');
  handleFileSelect('audioFile', null, null, 'audio');
  
  // Audio recording
  initAudioRecording();
  
  // Buttons
  $('randomizeBtn').addEventListener('click', randomizeProfile);
  $('submitBtn').addEventListener('click', submitTriage);
  $('clearBtn').addEventListener('click', clearAll);
  $('checkHealthBtn').addEventListener('click', checkHealth);
  $('downloadLogBtn').addEventListener('click', downloadLog);
  
  // Check health on load
  setTimeout(checkHealth, 500);
};

document.addEventListener('DOMContentLoaded', init);
