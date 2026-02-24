"use strict";
(() => {
  // src/domain/constants.ts
  var REQUIRED_RULES = [
    { id: "age", label: "Age", min: 13, max: 50 },
    { id: "systolicBp", label: "Systolic BP", min: 80, max: 180 },
    { id: "diastolicBp", label: "Diastolic BP", min: 45, max: 120 },
    { id: "fastingGlucose", label: "Fasting glucose", min: 3, max: 20 },
    { id: "tempF", label: "Body temperature (degF)", min: 95, max: 104 },
    { id: "restingHr", label: "Resting heart rate", min: 45, max: 140 }
  ];
  var OPTIONAL_NUMERIC_RULES = [
    { id: "gestWeeks", label: "Gestational weeks", min: 4, max: 42 },
    { id: "gravidity", label: "Gravidity", min: 1, max: 10 },
    { id: "parity", label: "Parity", min: 0, max: 10 },
    { id: "spo2", label: "SpO2", min: 88, max: 100 },
    { id: "hrv", label: "HRV", min: 10, max: 140 }
  ];
  var EVENT_LABELS = {
    "request.accepted": "Request Accepted",
    "request.rejected": "Request Rejected",
    "router.decision": "Router Decision",
    "router.prompt_plan": "Prompt Plan",
    "model.started": "Specialist Started",
    "model.completed": "Specialist Completed",
    "model.failed": "Specialist Failed",
    "medgemma.started": "MedGemma Started",
    "medgemma.completed": "MedGemma Completed",
    "medgemma.delta": "MedGemma Stream Chunk",
    "gemini.started": "Gemini Started",
    "gemini.completed": "Gemini Completed",
    "gemini.skipped": "Gemini Skipped",
    "diagnostics.inference_breakdown": "Inference Diagnostics",
    "triage.final": "Final Response",
    "triage.error": "Pipeline Error"
  };
  var DEFAULT_FORM_VALUES = {
    requestId: "",
    randomProfileMode: "none yet",
    age: "25",
    gestWeeks: "31",
    gravidity: "2",
    parity: "1",
    bmiGroup: "overweight",
    knownConditions: "",
    medications: "prenatal_vitamins",
    systolicBp: "130",
    diastolicBp: "80",
    fastingGlucose: "5.1",
    tempF: "98.0",
    restingHr: "86",
    spo2: "98",
    hrv: "42.0",
    headache: false,
    visionChanges: false,
    decreasedFetalMovement: false,
    freeText: ""
  };
  var PROFILE_MODES = [
    {
      name: "normal_monitoring",
      weight: 0.5,
      valueRanges: {
        age: [20, 34],
        gestWeeks: [20, 39],
        gravidity: [1, 4],
        parity: [0, 3],
        systolicBp: [104, 126],
        diastolicBp: [64, 82],
        fastingGlucose: [4, 5.2],
        tempF: [97.3, 99],
        restingHr: [62, 92],
        spo2: [96, 100],
        hrv: [28, 95]
      },
      bmiGroup: ["normal", "overweight"],
      knownConditions: ["none"],
      medications: ["prenatal_vitamins"],
      booleans: {
        headache: 0.08,
        visionChanges: 0.04,
        decreasedFetalMovement: 0.05
      },
      notes: [
        "Routine evening check-in, feeling mostly stable.",
        "No major concerns today, just regular tracking update."
      ]
    },
    {
      name: "borderline_risk",
      weight: 0.34,
      valueRanges: {
        age: [24, 40],
        gestWeeks: [26, 40],
        gravidity: [2, 6],
        parity: [1, 4],
        systolicBp: [128, 146],
        diastolicBp: [80, 94],
        fastingGlucose: [5.3, 7.8],
        tempF: [97.6, 99.7],
        restingHr: [76, 104],
        spo2: [94, 99],
        hrv: [18, 70]
      },
      bmiGroup: ["overweight", "obese"],
      knownConditions: ["prior_gdm", "chronic_hypertension", "none"],
      medications: ["prenatal_vitamins", "low_dose_aspirin", "none"],
      booleans: {
        headache: 0.35,
        visionChanges: 0.12,
        decreasedFetalMovement: 0.16
      },
      notes: [
        "Mild headache intermittently, requested triage review.",
        "Monitoring trend change over last 2 days."
      ]
    },
    {
      name: "red_flag",
      weight: 0.16,
      valueRanges: {
        age: [19, 43],
        gestWeeks: [28, 41],
        gravidity: [2, 8],
        parity: [1, 5],
        systolicBp: [150, 170],
        diastolicBp: [95, 112],
        fastingGlucose: [8.5, 16],
        tempF: [98, 101.5],
        restingHr: [88, 125],
        spo2: [92, 98],
        hrv: [10, 55]
      },
      bmiGroup: ["overweight", "obese"],
      knownConditions: [
        "gestational_diabetes",
        "chronic_hypertension",
        "prior_preeclampsia"
      ],
      medications: ["insulin", "labetalol", "prenatal_vitamins"],
      booleans: {
        headache: 0.62,
        visionChanges: 0.4,
        decreasedFetalMovement: 0.28
      },
      notes: [
        "Symptoms feel worse than baseline, requesting urgent review.",
        "Persistent headache and elevated values, worried about progression."
      ]
    }
  ];
  var PAYLOAD_SOURCE = "qa_console_watch_prototype";
  var COMPOSER_MODE = "medgemma_first";
  var OUTPUT_STYLE = "notebook";
  var TEMP_INPUT_UNIT = "degF";
  var AUDIO_MIME_CANDIDATES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4"
  ];

  // src/domain/validation.ts
  function evaluateRequiredInputs(values, rules = REQUIRED_RULES) {
    const missing = [];
    const errors = [];
    for (const rule of rules) {
      const value = values[rule.id] ?? null;
      if (value === null) {
        missing.push(rule.label);
        continue;
      }
      if (value < rule.min || value > rule.max) {
        errors.push(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
      }
    }
    const systolic = values["systolicBp"] ?? null;
    const diastolic = values["diastolicBp"] ?? null;
    if (systolic !== null && diastolic !== null && systolic <= diastolic) {
      errors.push("Systolic BP must be greater than diastolic BP.");
    }
    return { missing, errors };
  }
  function validateAllInputs(requiredValues, optionalValues, backendUrl, activeMode, hasAudio, requiredRules = REQUIRED_RULES, optionalRules = OPTIONAL_NUMERIC_RULES) {
    const errors = [];
    const required = evaluateRequiredInputs(requiredValues, requiredRules);
    errors.push(...required.errors);
    if (required.missing.length > 0) {
      errors.push(`Missing required inputs: ${required.missing.join(", ")}.`);
    }
    for (const rule of optionalRules) {
      const raw = (optionalValues[rule.id] ?? "").trim();
      if (!raw) continue;
      const value = Number(raw);
      if (Number.isNaN(value)) {
        errors.push(`${rule.label} must be a valid number.`);
        continue;
      }
      if (value < rule.min || value > rule.max) {
        errors.push(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
      }
    }
    if (!backendUrl.startsWith("http://") && !backendUrl.startsWith("https://")) {
      errors.push("Backend URL must start with http:// or https://");
    }
    if (activeMode === "voice" && !hasAudio) {
      errors.push(
        "Voice mode selected but no audio attached. Upload audio or record in-app."
      );
    }
    return errors;
  }
  function parseNumericInput(raw) {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    const value = Number(trimmed);
    return Number.isNaN(value) ? null : value;
  }

  // src/domain/randomizer.ts
  function randomInt(min, max, rng = Math.random) {
    return Math.floor(rng() * (max - min + 1)) + min;
  }
  function randomFloat(min, max, step = 0.1, rng = Math.random) {
    const span = Math.round((max - min) / step);
    const pick2 = randomInt(0, span, rng);
    return Number((min + pick2 * step).toFixed(2));
  }
  function weightedPick(options, rng = Math.random) {
    const total = options.reduce((sum, item) => sum + item.weight, 0);
    let marker = rng() * total;
    for (const item of options) {
      marker -= item.weight;
      if (marker <= 0) return item;
    }
    return options[options.length - 1];
  }
  function pick(values, rng = Math.random) {
    return values[randomInt(0, values.length - 1, rng)];
  }
  function generateRandomProfile(rng = Math.random, modes = PROFILE_MODES) {
    const mode = weightedPick(modes, rng);
    const r = mode.valueRanges;
    const age = randomInt(r.age[0], r.age[1], rng);
    const gestWeeks = randomInt(r.gestWeeks[0], r.gestWeeks[1], rng);
    const gravidity = randomInt(r.gravidity[0], r.gravidity[1], rng);
    const parityCandidate = randomInt(r.parity[0], r.parity[1], rng);
    const parity = Math.min(parityCandidate, gravidity);
    let systolicBp = randomInt(r.systolicBp[0], r.systolicBp[1], rng);
    let diastolicBp = randomInt(r.diastolicBp[0], r.diastolicBp[1], rng);
    if (diastolicBp >= systolicBp) {
      diastolicBp = systolicBp - randomInt(8, 25, rng);
    }
    const fastingGlucose = randomFloat(
      r.fastingGlucose[0],
      r.fastingGlucose[1],
      0.1,
      rng
    );
    const tempF = randomFloat(r.tempF[0], r.tempF[1], 0.1, rng);
    const restingHr = randomInt(r.restingHr[0], r.restingHr[1], rng);
    const spo2 = randomInt(r.spo2[0], r.spo2[1], rng);
    const hrv = randomFloat(r.hrv[0], r.hrv[1], 0.1, rng);
    const headache = rng() < mode.booleans.headache;
    const visionChanges = rng() < mode.booleans.visionChanges;
    const decreasedFetalMovement = rng() < mode.booleans.decreasedFetalMovement;
    const bmiGroup = pick(mode.bmiGroup, rng);
    const selectedCondition = pick(mode.knownConditions, rng);
    const selectedMedication = pick(mode.medications, rng);
    const freeText = pick(mode.notes, rng);
    return {
      age,
      gestWeeks,
      gravidity,
      parity,
      bmiGroup,
      systolicBp,
      diastolicBp,
      fastingGlucose,
      tempF,
      restingHr,
      spo2,
      hrv,
      headache,
      visionChanges,
      decreasedFetalMovement,
      knownConditions: selectedCondition === "none" ? "" : selectedCondition,
      medications: selectedMedication === "none" ? "" : selectedMedication,
      freeText,
      profileModeName: mode.name
    };
  }

  // src/domain/payload.ts
  function parseCsvList(value) {
    return String(value || "").split(",").map((x) => x.trim()).filter(Boolean);
  }
  function celsiusFromFahrenheit(tempF) {
    return Number(((tempF - 32) * 5 / 9).toFixed(2));
  }
  function composeProfileSummary(age, gravidity, parity, gestWeeks, bmiGroup) {
    return [
      "Patient profile:",
      `- Age: ${age} years`,
      `- Obstetric history: G${gravidity}P${parity}`,
      `- Gestational age: ${gestWeeks} weeks`,
      `- BMI group: ${bmiGroup}`
    ].join("\n");
  }
  function composePayloadSkeleton(values, activeMode) {
    const knownConditions = parseCsvList(values.knownConditions).filter(
      (x) => x.toLowerCase() !== "none"
    );
    const medications = parseCsvList(values.medications).filter(
      (x) => x.toLowerCase() !== "none"
    );
    const note = (values.freeText || "").trim();
    const age = values.age ?? 0;
    const gravidity = values.gravidity ?? 0;
    const parity = values.parity ?? 0;
    const gestWeeks = values.gestWeeks ?? 0;
    const bmiGroup = values.bmiGroup || "normal";
    const profileSummary = composeProfileSummary(
      age,
      gravidity,
      parity,
      gestWeeks,
      bmiGroup
    );
    const tempF = values.tempF ?? 98;
    const requestId = (values.requestId || "").trim() || `qa-${Date.now()}-${randomInt(100, 999)}`;
    const noteParts = [profileSummary];
    if (note) {
      noteParts.push(`Clinical note:
${note}`);
    } else if (activeMode === "voice") {
      noteParts.push(
        "Voice-first check-in. Prioritize transcript content from MedASR."
      );
    } else if (activeMode === "image") {
      noteParts.push(
        "Image-first check-in. Combine specialist image outputs with vitals."
      );
    }
    return {
      request_id: requestId,
      patient_context: {
        age_years: age,
        gestational_weeks: gestWeeks,
        known_conditions: knownConditions,
        medications
      },
      vitals: {
        systolic_bp: values.systolicBp ?? 0,
        diastolic_bp: values.diastolicBp ?? 0,
        fasting_glucose: values.fastingGlucose ?? 0,
        hr: values.restingHr ?? 0,
        spo2: values.spo2 ?? 0,
        temp_c: celsiusFromFahrenheit(tempF),
        hrv: values.hrv ?? 0
      },
      inputs: {
        headache: values.headache,
        vision_changes: values.visionChanges,
        decreased_fetal_movement: values.decreasedFetalMovement,
        free_text: noteParts.join("\n\n"),
        wound_image_b64: null,
        skin_image_b64: null,
        audio_b64: null
      },
      metadata: {
        source: PAYLOAD_SOURCE,
        composer_mode: COMPOSER_MODE,
        medgemma_output_style: OUTPUT_STYLE,
        ui_mode: activeMode,
        simulator: {
          age_years: age,
          bmi_group: bmiGroup,
          gravidity,
          parity,
          temp_input_unit: TEMP_INPUT_UNIT
        }
      }
    };
  }

  // src/domain/sse.ts
  function parseBlock(block, onEvent) {
    const lines = block.split("\n");
    let eventName = "message";
    const dataLines = [];
    for (const line of lines) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    const dataRaw = dataLines.join("\n");
    let payload = { raw: dataRaw };
    if (dataRaw) {
      try {
        payload = JSON.parse(dataRaw);
      } catch {
        payload = { raw: dataRaw };
      }
    }
    onEvent(eventName, payload);
  }
  function createSseEventParser(onEvent) {
    let buffer = "";
    return (chunk, flush = false) => {
      buffer += chunk.replace(/\r/g, "");
      while (true) {
        const splitAt = buffer.indexOf("\n\n");
        if (splitAt === -1) break;
        const block = buffer.slice(0, splitAt);
        buffer = buffer.slice(splitAt + 2);
        parseBlock(block, onEvent);
      }
      if (flush && buffer.trim()) {
        parseBlock(buffer, onEvent);
        buffer = "";
      }
    };
  }

  // src/adapters/audio.ts
  function encodeWavFromAudioBuffer(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const samplesPerChannel = audioBuffer.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = samplesPerChannel * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    function writeString(offset2, text) {
      for (let i = 0; i < text.length; i += 1) {
        view.setUint8(offset2 + i, text.charCodeAt(i));
      }
    }
    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);
    let offset = 44;
    for (let i = 0; i < samplesPerChannel; i += 1) {
      for (let channel = 0; channel < numChannels; channel += 1) {
        const sample = audioBuffer.getChannelData(channel)[i];
        const clamped = Math.max(-1, Math.min(1, sample));
        const int16 = clamped < 0 ? clamped * 32768 : clamped * 32767;
        view.setInt16(offset, int16, true);
        offset += 2;
      }
    }
    return buffer;
  }
  async function convertToWavIfPossible(blob) {
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtor) return null;
    try {
      const audioContext = new AudioCtor();
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      await audioContext.close();
      const wavBuffer = encodeWavFromAudioBuffer(audioBuffer);
      return new Blob([wavBuffer], { type: "audio/wav" });
    } catch {
      return null;
    }
  }

  // src/adapters/base64.ts
  function base64FromBlob(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const value = String(reader.result || "");
        const split = value.split(",");
        resolve(split.length > 1 ? split[1] : value);
      };
      reader.onerror = () => reject(new Error("Failed to read file for base64 conversion."));
      reader.readAsDataURL(blob);
    });
  }

  // src/adapters/media.ts
  async function attachMedia(payload, files) {
    let audioStatusMessage = null;
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
      audioStatusMessage = wavBlob ? "Uploaded audio converted to WAV for MedASR" : `Uploaded audio kept as ${files.uploadedAudio.type || "original format"}`;
      return { audioStatusMessage };
    }
    if (files.recordedAudioBlob) {
      payload.inputs.audio_b64 = await base64FromBlob(files.recordedAudioBlob);
    }
    return { audioStatusMessage };
  }

  // src/adapters/streaming.ts
  async function streamSseFromFetch(response, onChunk) {
    if (!response.body) {
      throw new Error("Streaming response body is missing.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      onChunk(decoder.decode(value, { stream: true }), false);
    }
    onChunk("", true);
  }

  // src/domain/formatters.ts
  function escapeHtml(value) {
    return String(value || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  var SECTION_KEY_MAP = {
    "RISK LEVEL": "risk_level",
    "CLINICAL REASONING": "clinical_reasoning",
    "POTENTIAL COMPLICATIONS": "potential_complications",
    "RECOMMENDED ACTIONS": "recommended_actions",
    "WARNING SIGNS": "warning_signs"
  };
  var SECTION_HEADING_REGEX = /^\s*(RISK LEVEL|CLINICAL REASONING|POTENTIAL COMPLICATIONS|RECOMMENDED ACTIONS|WARNING SIGNS)\s*:\s*(.*)$/i;
  function splitSectionedGuidance(rawText) {
    const text = String(rawText || "").replace(/\r/g, "");
    const sections = {
      risk_level: "",
      clinical_reasoning: [],
      potential_complications: [],
      recommended_actions: [],
      warning_signs: []
    };
    let currentKey = "";
    for (const line of text.split("\n")) {
      const match = line.match(SECTION_HEADING_REGEX);
      if (match) {
        const heading = match[1].toUpperCase();
        currentKey = SECTION_KEY_MAP[heading] || "";
        const tail = (match[2] || "").trim();
        if (tail && currentKey) {
          if (currentKey === "risk_level") {
            sections.risk_level = tail;
          } else {
            sections[currentKey].push(tail);
          }
        }
        continue;
      }
      if (!currentKey) continue;
      const cleaned = line.trim();
      if (!cleaned) continue;
      if (currentKey === "risk_level") {
        sections.risk_level = cleaned;
      } else {
        sections[currentKey].push(cleaned);
      }
    }
    return sections;
  }
  function toCleanList(lines) {
    const out = [];
    const arr = Array.isArray(lines) ? lines : [];
    for (const line of arr) {
      const cleaned = String(line || "").replace(/^\s*[-*•]+\s*/, "").replace(/\s+/g, " ").trim();
      if (!cleaned) continue;
      if (!out.includes(cleaned)) out.push(cleaned);
    }
    return out;
  }
  function toParagraph(lines) {
    return toCleanList(lines).join(" ");
  }
  function buildImageEvidenceLines(specialist) {
    const lines = [];
    if (specialist?.wound_scores) {
      const wound = specialist.wound_scores;
      const urgency = Number(wound.urgency || 0);
      const infection = Number(wound.infection_risk || 0);
      if (urgency >= 0.6 || infection >= 0.7) {
        lines.push(
          `Wound-image specialist flagged elevated risk (urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}).`
        );
      } else {
        lines.push(
          `Wound-image specialist reviewed the image with no high-risk threshold crossed (urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}).`
        );
      }
    }
    if (specialist?.skin_top3 && Array.isArray(specialist.skin_top3) && specialist.skin_top3.length) {
      const labels = specialist.skin_top3.slice(0, 3).map(
        (row) => `${String(row.condition || "unknown").replace(/_/g, " ")} (${Number(row.score || 0).toFixed(2)})`
      );
      lines.push(`Skin specialist top findings: ${labels.join(", ")}.`);
    }
    return lines;
  }
  function formatPatientGuidanceHtml(finalPayload) {
    const fp = finalPayload || {};
    const sectioned = splitSectionedGuidance(fp.patient_message);
    const reasoning = toParagraph(sectioned.clinical_reasoning);
    const complications = toParagraph(sectioned.potential_complications);
    const actions = toCleanList(sectioned.recommended_actions);
    const warnings = toCleanList(sectioned.warning_signs);
    const specialist = fp.specialist_outputs || {};
    const imageEvidenceLines = buildImageEvidenceLines(specialist);
    const risk = String(fp.risk_level || sectioned.risk_level || "-").toUpperCase().replace("YELLOW", "MID").replace("GREEN", "LOW").replace("RED", "HIGH");
    const hasSections = reasoning || complications || actions.length || warnings.length;
    if (!hasSections) {
      return `<p>${escapeHtml(fp.patient_message || "No patient message generated.")}</p>`;
    }
    const riskClass = risk === "HIGH" ? "red" : risk === "MID" ? "yellow" : "green";
    const actionsHtml = actions.length ? `<ul class="guidance-list">${actions.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>` : `<p class="guidance-empty">No immediate actions listed.</p>`;
    const warningsHtml = warnings.length ? `<ul class="guidance-list">${warnings.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>` : `<p class="guidance-empty">No urgent warning signs listed.</p>`;
    return `
    <div class="patient-guidance">
      <div class="guidance-head">
        <span class="guidance-title">For the patient</span>
        <span class="risk-pill ${riskClass}">${escapeHtml(risk)} RISK</span>
      </div>
      <div class="guidance-card">
        <h4>What this means now</h4>
        <p>${escapeHtml(
      reasoning || "Current readings do not show severe immediate danger, but continue close monitoring."
    )}</p>
      </div>
      <div class="guidance-card">
        <h4>Possible complications if this continues</h4>
        <p>${escapeHtml(
      complications || "Complication risk depends on future trends and symptom progression."
    )}</p>
      </div>
      ${imageEvidenceLines.length ? `<div class="guidance-card">
              <h4>What image analysis found</h4>
              <ul class="guidance-list">${imageEvidenceLines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul>
            </div>` : ""}
      <div class="guidance-card">
        <h4>What to do next</h4>
        ${actionsHtml}
      </div>
      <div class="guidance-card warning">
        <h4>Seek urgent care now if</h4>
        ${warningsHtml}
      </div>
    </div>
  `;
  }
  function formatVisitSummaryHtml(summaryText) {
    const text = String(summaryText || "").trim();
    if (!text) return "No visit summary generated.";
    const lines = text.split("\n").map((x) => x.trim()).filter(Boolean);
    const bullets = lines.filter((line) => line.startsWith("-")).map((line) => line.replace(/^-+\s*/, "").trim());
    if (!bullets.length) {
      return `<p>${escapeHtml(text)}</p>`;
    }
    return `<ul class="visit-list">${bullets.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul>`;
  }
  function formatSpecialistSummaryHtml(specialist) {
    const parts = [];
    if (specialist?.transcript) {
      const full = String(specialist.transcript).replace(/\s+/g, " ").trim();
      const clipped = full.length > 170 ? `${full.slice(0, 170)}...` : full;
      parts.push(`
      <div class="specialist-chip">
        <strong>What the app heard:</strong> ${escapeHtml(clipped)}
      </div>
    `);
    }
    if (specialist?.wound_scores) {
      const wound = specialist.wound_scores;
      const urgency = Number(wound.urgency || 0);
      const infection = Number(wound.infection_risk || 0);
      const erythema = Number(wound.erythema || 0);
      parts.push(
        `<div class="specialist-chip"><strong>Wound model:</strong> urgency ${urgency.toFixed(2)}, infection ${infection.toFixed(2)}, erythema ${erythema.toFixed(2)}.</div>`
      );
    }
    if (specialist?.skin_top3 || specialist?.skin_scores) {
      if (Array.isArray(specialist?.skin_top3) && specialist.skin_top3.length) {
        const labels = specialist.skin_top3.slice(0, 3).map(
          (row) => `${String(row.condition || "unknown").replace(/_/g, " ")} (${Number(row.score || 0).toFixed(2)})`
        );
        parts.push(
          `<div class="specialist-chip"><strong>Skin model:</strong> ${escapeHtml(labels.join(", "))}.</div>`
        );
      } else {
        parts.push(
          '<div class="specialist-chip">Skin specialist was used.</div>'
        );
      }
    }
    if (!parts.length) {
      return "No specialist outputs were used in this run.";
    }
    return `<div class="specialist-stack">${parts.join("")}</div>`;
  }
  function formatTimelineEntryText(eventName, payload) {
    const p = payload || {};
    const time = typeof p.timestamp === "string" ? p.timestamp : (/* @__PURE__ */ new Date()).toISOString();
    const label = EVENT_LABELS[eventName] || eventName;
    const details = [];
    if (p.model) details.push(`model=${p.model}`);
    if (p.intent) details.push(`intent=${p.intent}`);
    if (p.prompt_strategy) details.push(`strategy=${p.prompt_strategy}`);
    if (p.risk_level) details.push(`risk=${p.risk_level}`);
    if (p.latency_ms !== void 0) details.push(`latency=${p.latency_ms}ms`);
    if (p.gpu_warmup_ms !== void 0) details.push(`warmup=${p.gpu_warmup_ms}ms`);
    if (p.medgemma_inference_ms !== void 0) {
      details.push(`medgemma_infer=${p.medgemma_inference_ms}ms`);
    }
    if (p.composer_mode) details.push(`composer=${p.composer_mode}`);
    if (p.reason) details.push(`reason=${p.reason}`);
    if (p.error) details.push(`error=${p.error}`);
    return `${time} | ${label}${details.length ? ` | ${details.join(" | ")}` : ""}`;
  }
  function stringifyJson(data) {
    return JSON.stringify(data, null, 2);
  }

  // src/ui/dom.ts
  function el(id) {
    return document.getElementById(id);
  }
  function inputEl(id) {
    return document.getElementById(id);
  }
  function selectEl(id) {
    return document.getElementById(id);
  }
  function textareaEl(id) {
    return document.getElementById(id);
  }
  var state = {
    lastRun: null,
    activeRun: null,
    mediaRecorder: null,
    mediaStream: null,
    mediaChunks: [],
    recordedAudioBlob: null,
    finalReceived: false,
    activeMode: "text",
    resultCardIndex: 0,
    inferenceTimer: null,
    inferenceStartTime: null
  };
  function getNumericValue(id) {
    return parseNumericInput(inputEl(id).value);
  }
  function extractRequiredValues() {
    const values = {};
    for (const rule of REQUIRED_RULES) {
      values[rule.id] = getNumericValue(rule.id);
    }
    return values;
  }
  function extractOptionalValues() {
    const values = {};
    for (const rule of OPTIONAL_NUMERIC_RULES) {
      values[rule.id] = inputEl(rule.id).value;
    }
    return values;
  }
  function extractFormValues() {
    return {
      age: getNumericValue("age"),
      gestWeeks: getNumericValue("gestWeeks"),
      gravidity: getNumericValue("gravidity"),
      parity: getNumericValue("parity"),
      bmiGroup: selectEl("bmiGroup").value,
      knownConditions: inputEl("knownConditions").value,
      medications: inputEl("medications").value,
      systolicBp: getNumericValue("systolicBp"),
      diastolicBp: getNumericValue("diastolicBp"),
      fastingGlucose: getNumericValue("fastingGlucose"),
      tempF: getNumericValue("tempF"),
      restingHr: getNumericValue("restingHr"),
      spo2: getNumericValue("spo2"),
      hrv: getNumericValue("hrv"),
      headache: inputEl("headache").checked,
      visionChanges: inputEl("visionChanges").checked,
      decreasedFetalMovement: inputEl("decreasedFetalMovement").checked,
      freeText: textareaEl("freeText").value,
      requestId: inputEl("requestId").value,
      backendUrl: inputEl("backendUrl").value
    };
  }
  function updateStatus(message) {
    el("statusLine").textContent = message;
  }
  function updateValidation(message) {
    el("validationLine").textContent = message || "";
  }
  function setHealthLine(message, kind = "") {
    const node = el("healthLine");
    node.textContent = message;
    node.className = kind ? `health-line ${kind}` : "health-line";
  }
  function setPreview(fileInput, previewEl) {
    const file = fileInput.files?.[0];
    if (!file) {
      previewEl.style.display = "none";
      previewEl.removeAttribute("src");
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    previewEl.src = objectUrl;
    previewEl.style.display = "block";
  }
  function updateRequiredGate() {
    const requiredValues = extractRequiredValues();
    const { missing, errors } = evaluateRequiredInputs(requiredValues);
    const requiredHint = el("requiredHint");
    const badge = el("requiredBadge");
    const vitalsChip = el("openVitalsBtn");
    const needsInput = missing.length > 0 || errors.length > 0;
    badge.textContent = String(missing.length);
    if (missing.length > 0) {
      requiredHint.textContent = `Missing required inputs: ${missing.join(", ")}.`;
      requiredHint.className = "required-hint";
      vitalsChip.classList.remove("ok");
    } else if (errors.length > 0) {
      requiredHint.textContent = errors[0];
      requiredHint.className = "required-hint";
      vitalsChip.classList.remove("ok");
    } else {
      requiredHint.textContent = "All required vitals are present.";
      requiredHint.className = "required-hint ok";
    }
    if (needsInput) {
      vitalsChip.classList.remove("ok");
      vitalsChip.classList.add("needs-input");
    } else {
      vitalsChip.classList.remove("needs-input");
      vitalsChip.classList.add("ok");
    }
    if (!state.activeRun) {
      el("submitBtn").disabled = missing.length > 0 || errors.length > 0;
    }
  }
  function setActiveMode(mode) {
    state.activeMode = mode;
    const modeBtnIds = {
      text: "modeTextBtn",
      voice: "modeVoiceBtn",
      image: "modeImageBtn"
    };
    Object.entries(modeBtnIds).forEach(([m, id]) => {
      el(id).classList.toggle("active", m === mode);
    });
    const panels = [
      { id: "textPanel", mode: "text" },
      { id: "voicePanel", mode: "voice" },
      { id: "imagePanel", mode: "image" }
    ];
    for (const item of panels) {
      const node = el(item.id);
      if (node) {
        node.classList.toggle("active", item.mode === mode);
      }
    }
    updateModeIndicators();
  }
  function updateModeIndicators() {
    const hasNote = textareaEl("freeText").value.trim().length > 0;
    const hasAudio = inputEl("audioFile").files?.length || state.recordedAudioBlob;
    const hasImage = inputEl("woundImage").files?.length || inputEl("skinImage").files?.length;
    el("modeTextBtn").classList.toggle("has-data", hasNote);
    el("modeVoiceBtn").classList.toggle("has-data", !!hasAudio);
    el("modeImageBtn").classList.toggle("has-data", !!hasImage);
    const modeLabel = state.activeMode.charAt(0).toUpperCase() + state.activeMode.slice(1);
    const subLabel = el("submitBtn").querySelector("small");
    if (subLabel) {
      subLabel.textContent = `Run AI Triage (${modeLabel})`;
    }
  }
  function showInferenceLoader() {
    const loader = document.getElementById("inferenceLoader");
    if (!loader) return;
    loader.hidden = false;
    const cards = Array.from(document.querySelectorAll(".result-card"));
    for (const card of cards) card.style.display = "none";
    const nav = document.querySelector(".carousel-bottom-nav");
    if (nav) nav.style.display = "none";
    const pipelineSection = document.querySelector("#resultsWrapper > .panel");
    if (pipelineSection) pipelineSection.style.display = "none";
    state.inferenceStartTime = Date.now();
    const elapsedEl = document.getElementById("loaderElapsed");
    if (elapsedEl) elapsedEl.textContent = "Elapsed: 0s";
    if (state.inferenceTimer) clearInterval(state.inferenceTimer);
    state.inferenceTimer = setInterval(() => {
      if (!state.inferenceStartTime) return;
      const secs = Math.floor((Date.now() - state.inferenceStartTime) / 1e3);
      const mins = Math.floor(secs / 60);
      const remainSecs = secs % 60;
      if (elapsedEl) {
        elapsedEl.textContent = mins > 0 ? `Elapsed: ${mins}m ${remainSecs}s` : `Elapsed: ${secs}s`;
      }
    }, 1e3);
  }
  function hideInferenceLoader() {
    const loader = document.getElementById("inferenceLoader");
    if (loader) loader.hidden = true;
    const cards = Array.from(document.querySelectorAll(".result-card"));
    for (const card of cards) card.style.display = "";
    const nav = document.querySelector(".carousel-bottom-nav");
    if (nav) nav.style.display = "";
    const pipelineSection = document.querySelector("#resultsWrapper > .panel");
    if (pipelineSection) pipelineSection.style.display = "";
    if (state.inferenceTimer) {
      clearInterval(state.inferenceTimer);
      state.inferenceTimer = null;
    }
    state.inferenceStartTime = null;
  }
  function listResultCards() {
    return Array.from(document.querySelectorAll(".result-card"));
  }
  function showResultCard(index) {
    const cards = listResultCards();
    if (!cards.length) return;
    const safeIndex = Math.max(0, Math.min(index, cards.length - 1));
    state.resultCardIndex = safeIndex;
    for (let i = 0; i < cards.length; i += 1) {
      cards[i].classList.toggle("is-active", i === safeIndex);
    }
    const prev = el("prevCardBtn");
    const next = el("nextCardBtn");
    if (prev) prev.disabled = safeIndex <= 0;
    if (next) next.disabled = safeIndex >= cards.length - 1;
    el("cardPosition").textContent = `${safeIndex + 1} / ${cards.length}`;
  }
  function shiftResultCard(delta) {
    showResultCard(state.resultCardIndex + delta);
    const wrapper = el("resultsWrapper");
    if (wrapper) wrapper.scrollTop = 0;
  }
  function renderList(id, values, emptyText) {
    const node = el(id);
    node.innerHTML = "";
    const items = Array.isArray(values) ? values : [];
    if (!items.length) {
      const li = document.createElement("li");
      li.textContent = emptyText;
      node.appendChild(li);
      return;
    }
    for (const item of items) {
      const li = document.createElement("li");
      li.textContent = String(item);
      node.appendChild(li);
    }
  }
  function resetRunViews() {
    el("eventTimeline").innerHTML = "";
    textareaEl("finalJson").value = "";
    el("riskLevel").textContent = "-";
    const riskEl = document.getElementById("riskLevel");
    if (riskEl) riskEl.className = "risk-pill-hero";
    const pf = document.getElementById("policyFloor");
    if (pf) pf.textContent = "-";
    const dc = document.getElementById("diagComposer");
    if (dc) dc.textContent = "-";
    const dms = document.getElementById("diagModelShare");
    if (dms) dms.textContent = "-";
    const dcs = document.getElementById("diagColdStart");
    if (dcs) dcs.textContent = "-";
    const dwm = document.getElementById("diagWarmupMs");
    if (dwm) dwm.textContent = "-";
    const dmm = document.getElementById("diagMedgemmaMs");
    if (dmm) dmm.textContent = "-";
    el("diagTotalMs").textContent = "-";
    el("patientMessage").innerHTML = "No result yet.";
    el("visitSummary").innerHTML = "No result yet.";
    el("specialistSummary").innerHTML = "No specialist outputs yet.";
    renderList("reasonsList", [], "No reasons yet.");
    renderList("actionsList", [], "No action plan yet.");
    showResultCard(0);
  }
  function setDiagnosticsFromBreakdown(payload) {
    const modelShare = typeof payload.model === "string" ? payload.model : "-";
    const dms = document.getElementById("diagModelShare");
    if (dms) dms.textContent = modelShare;
    const dc = document.getElementById("diagComposer");
    if (dc)
      dc.textContent = typeof payload.composer_mode === "string" ? payload.composer_mode : "-";
    if (payload.gpu_warmup_ms !== void 0) {
      const dwm = document.getElementById("diagWarmupMs");
      if (dwm) dwm.textContent = String(payload.gpu_warmup_ms);
    }
    if (payload.medgemma_inference_ms !== void 0) {
      const dmm = document.getElementById("diagMedgemmaMs");
      if (dmm) dmm.textContent = String(payload.medgemma_inference_ms);
    }
    if (payload.latency_ms !== void 0) {
      const dtm = document.getElementById("diagTotalMs");
      if (dtm) dtm.textContent = String(payload.latency_ms);
    }
  }
  function setDiagnosticsFromFinal(finalPayload) {
    const diag = finalPayload.inference_diagnostics || {};
    const timing = diag.medgemma_timing_breakdown || {};
    const share = diag.latency_share_pct || {};
    const total = finalPayload.latency_ms?.total;
    const dc = document.getElementById("diagComposer");
    if (dc)
      dc.textContent = diag.composer_mode || dc.textContent || "-";
    if (share.medgemma !== void 0 || share.gemini !== void 0) {
      const m = share.medgemma !== void 0 ? Number(share.medgemma).toFixed(1) : "0.0";
      const g = share.gemini !== void 0 ? Number(share.gemini).toFixed(1) : "0.0";
      const dms = document.getElementById("diagModelShare");
      if (dms) dms.textContent = `medgemma:${m}% | gemini:${g}%`;
    }
    if (timing.cold_start !== void 0) {
      const dcs = document.getElementById("diagColdStart");
      if (dcs) dcs.textContent = timing.cold_start ? "YES" : "NO";
    }
    if (timing.gpu_warmup_ms !== void 0) {
      const dwm = document.getElementById("diagWarmupMs");
      if (dwm) dwm.textContent = String(timing.gpu_warmup_ms);
    }
    if (timing.medgemma_inference_ms !== void 0) {
      const dmm = document.getElementById("diagMedgemmaMs");
      if (dmm) dmm.textContent = String(timing.medgemma_inference_ms);
    }
    if (total !== void 0) {
      const dtm = document.getElementById("diagTotalMs");
      if (dtm) dtm.textContent = String(total);
    }
  }
  function setFinalResponse(finalPayload) {
    const riskLevel = String(finalPayload.risk_level || "-").toLowerCase();
    const riskEl = el("riskLevel");
    riskEl.textContent = riskLevel.toUpperCase();
    riskEl.className = "risk-pill-hero " + riskLevel;
    el("policyFloor").textContent = String(
      finalPayload.policy_floor || "-"
    ).toUpperCase();
    setDiagnosticsFromFinal(finalPayload);
    el("patientMessage").innerHTML = formatPatientGuidanceHtml(finalPayload);
    el("visitSummary").innerHTML = formatVisitSummaryHtml(
      finalPayload.visit_prep_summary
    );
    const specialist = finalPayload.specialist_outputs || {};
    el("specialistSummary").innerHTML = formatSpecialistSummaryHtml(specialist);
    renderList(
      "reasonsList",
      finalPayload.medgemma_reasons,
      "No clinical reasons returned."
    );
    renderList("actionsList", finalPayload.action_items, "No actions returned.");
    textareaEl("finalJson").value = stringifyJson(finalPayload);
    showResultCard(0);
  }
  function timelineEntry(eventName, payload) {
    const li = document.createElement("li");
    li.textContent = formatTimelineEntryText(eventName, payload);
    el("eventTimeline").appendChild(li);
    li.scrollIntoView({ block: "nearest" });
  }
  function buildRunLog(url, payload) {
    state.activeRun = {
      backend_url: url,
      started_at: (/* @__PURE__ */ new Date()).toISOString(),
      completed_at: null,
      duration_ms: null,
      request_payload: payload,
      events: [],
      final_response: null
    };
    state.lastRun = null;
    state.finalReceived = false;
    el("downloadLogBtn").disabled = true;
  }
  function completeRunLog() {
    if (!state.activeRun) return;
    state.activeRun.completed_at = (/* @__PURE__ */ new Date()).toISOString();
    const start = Date.parse(state.activeRun.started_at);
    const end = Date.parse(state.activeRun.completed_at);
    state.activeRun.duration_ms = Number.isNaN(start) || Number.isNaN(end) ? null : end - start;
    state.lastRun = state.activeRun;
    state.activeRun = null;
    el("downloadLogBtn").disabled = false;
  }
  async function checkBackendHealth() {
    updateValidation("");
    const baseUrl = inputEl("backendUrl").value.trim().replace(/\/$/, "");
    if (!baseUrl.startsWith("http://") && !baseUrl.startsWith("https://")) {
      setHealthLine("Invalid backend URL.", "err");
      return;
    }
    setHealthLine("Checking backend health...", "");
    try {
      const response = await fetch(`${baseUrl}/health?probe=1`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const core = data.core_gpu_reachable === true ? "core:up" : "core:down";
      const derm = data.derm_reachable === true ? "derm:up" : "derm:down";
      const asr = data.medasr_reachable === true ? "medasr:up" : "medasr:down";
      const composerDefault = data.default_response_composer_mode || data.response_composer_mode || "unknown";
      setHealthLine(
        `Healthy | ${core} | ${derm} | ${asr} | default_mode=${composerDefault} (request can override)`,
        "ok"
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setHealthLine(`Health check failed: ${msg}`, "err");
    }
  }
  async function submitRequest() {
    updateValidation("");
    const formValues = extractFormValues();
    const requiredValues = extractRequiredValues();
    const optionalValues = extractOptionalValues();
    const uploadedAudio = inputEl("audioFile").files?.[0] ?? null;
    const hasAudio = Boolean(uploadedAudio || state.recordedAudioBlob);
    const validationErrors = validateAllInputs(
      requiredValues,
      optionalValues,
      formValues.backendUrl.trim(),
      state.activeMode,
      hasAudio
    );
    if (validationErrors.length) {
      updateValidation(validationErrors.join(" "));
      updateRequiredGate();
      openInputValuesPanel();
      return;
    }
    const payload = composePayloadSkeleton(
      formValues,
      state.activeMode
    );
    const mediaResult = await attachMedia(payload, {
      woundImage: inputEl("woundImage").files?.[0] ?? null,
      skinImage: inputEl("skinImage").files?.[0] ?? null,
      uploadedAudio,
      recordedAudioBlob: state.recordedAudioBlob
    });
    if (mediaResult.audioStatusMessage) {
      el("recordStatus").textContent = mediaResult.audioStatusMessage;
    }
    if (state.activeMode === "voice" && !payload.inputs.audio_b64) {
      updateValidation(
        "Voice mode selected but no audio payload could be attached."
      );
      return;
    }
    const baseUrl = formValues.backendUrl.trim().replace(/\/$/, "");
    const endpoint = `${baseUrl}/v1/triage/stream`;
    buildRunLog(baseUrl, payload);
    resetRunViews();
    updateStatus("Submitting request and opening SSE stream...");
    const submitBtn = el("submitBtn");
    submitBtn.disabled = true;
    openResultsPanel();
    showInferenceLoader();
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }
      const parser = createSseEventParser(
        (eventName, eventPayload) => {
          if (state.activeRun) {
            state.activeRun.events.push({
              event: eventName,
              payload: eventPayload
            });
          }
          timelineEntry(eventName, eventPayload);
          if (eventName === "diagnostics.inference_breakdown") {
            setDiagnosticsFromBreakdown(eventPayload);
          }
          if (eventName === "triage.final") {
            state.finalReceived = true;
            hideInferenceLoader();
            if (state.activeRun)
              state.activeRun.final_response = eventPayload;
            setFinalResponse(eventPayload);
          }
          if (eventName === "triage.error") {
            const msg = typeof eventPayload.error === "string" ? eventPayload.error : "Unknown triage error";
            updateValidation(`Triage error: ${msg}`);
          }
        }
      );
      await streamSseFromFetch(response, parser);
      completeRunLog();
      if (state.finalReceived) {
        updateStatus("Stream complete. Final triage received.");
      } else {
        updateStatus(
          "Stream complete, but no triage.final event was found."
        );
      }
    } catch (err) {
      hideInferenceLoader();
      completeRunLog();
      updateStatus("Request failed.");
      const msg = err instanceof Error ? err.message : String(err);
      updateValidation(msg);
    } finally {
      updateRequiredGate();
    }
  }
  function resetToDefaults() {
    inputEl("requestId").value = DEFAULT_FORM_VALUES.requestId;
    inputEl("randomProfileMode").value = DEFAULT_FORM_VALUES.randomProfileMode;
    inputEl("age").value = DEFAULT_FORM_VALUES.age;
    inputEl("gestWeeks").value = DEFAULT_FORM_VALUES.gestWeeks;
    inputEl("gravidity").value = DEFAULT_FORM_VALUES.gravidity;
    inputEl("parity").value = DEFAULT_FORM_VALUES.parity;
    selectEl("bmiGroup").value = DEFAULT_FORM_VALUES.bmiGroup;
    inputEl("knownConditions").value = DEFAULT_FORM_VALUES.knownConditions;
    inputEl("medications").value = DEFAULT_FORM_VALUES.medications;
    inputEl("systolicBp").value = DEFAULT_FORM_VALUES.systolicBp;
    inputEl("diastolicBp").value = DEFAULT_FORM_VALUES.diastolicBp;
    inputEl("fastingGlucose").value = DEFAULT_FORM_VALUES.fastingGlucose;
    inputEl("tempF").value = DEFAULT_FORM_VALUES.tempF;
    inputEl("restingHr").value = DEFAULT_FORM_VALUES.restingHr;
    inputEl("spo2").value = DEFAULT_FORM_VALUES.spo2;
    inputEl("hrv").value = DEFAULT_FORM_VALUES.hrv;
    inputEl("headache").checked = DEFAULT_FORM_VALUES.headache;
    inputEl("visionChanges").checked = DEFAULT_FORM_VALUES.visionChanges;
    inputEl("decreasedFetalMovement").checked = DEFAULT_FORM_VALUES.decreasedFetalMovement;
    textareaEl("freeText").value = DEFAULT_FORM_VALUES.freeText;
    inputEl("woundImage").value = "";
    inputEl("skinImage").value = "";
    inputEl("audioFile").value = "";
    setPreview(
      inputEl("woundImage"),
      document.getElementById("woundPreview")
    );
    setPreview(
      inputEl("skinImage"),
      document.getElementById("skinPreview")
    );
    state.recordedAudioBlob = null;
    el("recordStatus").textContent = "Not recording";
    setHealthLine("Not checked.");
    updateValidation("");
    updateStatus("Ready.");
    resetRunViews();
    state.lastRun = null;
    state.activeRun = null;
    el("downloadLogBtn").disabled = true;
    setActiveMode("text");
    updateRequiredGate();
    updateModeIndicators();
    closeResultsPanel();
  }
  function updateAudioStatus() {
    const uploadedAudio = inputEl("audioFile").files?.[0];
    if (uploadedAudio) {
      el("recordStatus").textContent = `Audio file selected: ${uploadedAudio.name}`;
      return;
    }
    if (state.recordedAudioBlob) {
      el("recordStatus").textContent = "Recorded (stored for MedASR)";
      return;
    }
    el("recordStatus").textContent = "Not recording";
  }
  function fillFromRandomProfile() {
    const profile = generateRandomProfile();
    inputEl("age").value = String(profile.age);
    inputEl("gestWeeks").value = String(profile.gestWeeks);
    inputEl("gravidity").value = String(profile.gravidity);
    inputEl("parity").value = String(profile.parity);
    selectEl("bmiGroup").value = profile.bmiGroup;
    inputEl("systolicBp").value = String(profile.systolicBp);
    inputEl("diastolicBp").value = String(profile.diastolicBp);
    inputEl("fastingGlucose").value = String(profile.fastingGlucose);
    inputEl("tempF").value = String(profile.tempF);
    inputEl("restingHr").value = String(profile.restingHr);
    inputEl("spo2").value = String(profile.spo2);
    inputEl("hrv").value = String(profile.hrv);
    inputEl("headache").checked = profile.headache;
    inputEl("visionChanges").checked = profile.visionChanges;
    inputEl("decreasedFetalMovement").checked = profile.decreasedFetalMovement;
    inputEl("knownConditions").value = profile.knownConditions;
    inputEl("medications").value = profile.medications;
    textareaEl("freeText").value = profile.freeText;
    inputEl("randomProfileMode").value = profile.profileModeName;
    updateStatus(`Randomized using profile mode: ${profile.profileModeName}`);
    updateValidation("");
    updateRequiredGate();
    updateModeIndicators();
  }
  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      updateValidation("Browser microphone access is not available.");
      return;
    }
    try {
      state.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true
      });
      const mimeType = AUDIO_MIME_CANDIDATES.find(
        (m) => MediaRecorder.isTypeSupported(m)
      ) || "";
      state.mediaChunks = [];
      state.mediaRecorder = mimeType ? new MediaRecorder(state.mediaStream, { mimeType }) : new MediaRecorder(state.mediaStream);
      state.mediaRecorder.ondataavailable = (event) => {
        if (event.data?.size > 0) {
          state.mediaChunks.push(event.data);
        }
      };
      state.mediaRecorder.onstop = async () => {
        const blobType = mimeType || "audio/webm";
        const rawBlob = new Blob(state.mediaChunks, { type: blobType });
        const wavBlob = await convertToWavIfPossible(rawBlob);
        state.recordedAudioBlob = wavBlob || rawBlob;
        if (state.mediaStream) {
          state.mediaStream.getTracks().forEach((track) => track.stop());
        }
        state.mediaStream = null;
        state.mediaRecorder = null;
        state.mediaChunks = [];
        el("recordStart").disabled = false;
        el("recordStop").disabled = true;
        el("recordStatus").textContent = wavBlob ? "Recorded (stored as WAV for MedASR)" : "Recorded (original codec)";
        updateModeIndicators();
      };
      state.mediaRecorder.start();
      state.recordedAudioBlob = null;
      updateModeIndicators();
      el("recordStart").disabled = true;
      el("recordStop").disabled = false;
      el("recordStatus").textContent = "Recording...";
      updateStatus("Microphone recording started.");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      updateValidation(`Microphone error: ${msg}`);
    }
  }
  function stopRecording() {
    if (!state.mediaRecorder) return;
    state.mediaRecorder.stop();
    updateStatus("Recording stopped, processing audio...");
  }
  function downloadRunLog() {
    if (!state.lastRun) return;
    const fileName = `momnitrix-run-${Date.now()}.json`;
    const blob = new Blob([stringifyJson(state.lastRun)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }
  function openInputValuesPanel() {
    const panel = el("inputValuesPanel");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.hidden = false;
    panel.classList.add("is-open");
    backdrop.hidden = false;
    backdrop.classList.add("is-open");
    el("openVitalsBtn").setAttribute("aria-expanded", "true");
    document.body.classList.add("sheet-open");
    const firstField = document.getElementById("age");
    if (firstField) firstField.focus();
  }
  function openInteractionPanel() {
    const panel = document.getElementById("interactionSheet");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add("sheet-open");
    void panel.offsetWidth;
    panel.classList.add("is-open");
    backdrop.classList.add("is-open");
  }
  function closeInteractionPanel() {
    const panel = document.getElementById("interactionSheet");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.classList.remove("is-open");
    backdrop.classList.remove("is-open");
    document.body.classList.remove("sheet-open");
    setTimeout(() => {
      if (!panel.classList.contains("is-open")) {
        panel.hidden = true;
        backdrop.hidden = true;
      }
    }, 250);
  }
  function closeInputValuesPanel() {
    const panel = el("inputValuesPanel");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.classList.remove("is-open");
    backdrop.classList.remove("is-open");
    document.body.classList.remove("sheet-open");
    el("openVitalsBtn").setAttribute("aria-expanded", "false");
    el("openVitalsBtn").focus();
    setTimeout(() => {
      if (!panel.classList.contains("is-open")) {
        panel.hidden = true;
        backdrop.hidden = true;
      }
    }, 250);
  }
  function openResultsPanel() {
    const panel = el("resultsWrapper");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add("sheet-open");
    void panel.offsetWidth;
    panel.classList.add("is-open");
    backdrop.classList.add("is-open");
  }
  function closeResultsPanel() {
    const panel = el("resultsWrapper");
    const backdrop = el("inputValuesBackdrop");
    if (!panel || !backdrop) return;
    panel.classList.remove("is-open");
    backdrop.classList.remove("is-open");
    document.body.classList.remove("sheet-open");
    setTimeout(() => {
      if (!panel.classList.contains("is-open")) {
        panel.hidden = true;
        backdrop.hidden = true;
      }
    }, 250);
  }
  function importWatchPlaceholder() {
    updateStatus(
      "Watch import placeholder is active. Manual values remain enabled for prototype testing."
    );
  }
  function wireEvents() {
    el("modeTextBtn").addEventListener("click", () => {
      setActiveMode("text");
      openInteractionPanel();
    });
    el("modeVoiceBtn").addEventListener("click", () => {
      setActiveMode("voice");
      openInteractionPanel();
    });
    el("modeImageBtn").addEventListener("click", () => {
      setActiveMode("image");
      openInteractionPanel();
    });
    el("openVitalsBtn").addEventListener("click", openInputValuesPanel);
    el("closeVitalsBtn").addEventListener("click", closeInputValuesPanel);
    const closeResultsBtn = document.getElementById("closeResultsBtn");
    if (closeResultsBtn) {
      closeResultsBtn.addEventListener("click", closeResultsPanel);
    }
    const closeInteractionBtn = document.getElementById("closeInteractionBtn");
    if (closeInteractionBtn) {
      closeInteractionBtn.addEventListener("click", closeInteractionPanel);
    }
    el("inputValuesBackdrop").addEventListener("click", () => {
      closeInputValuesPanel();
      closeResultsPanel();
      closeInteractionPanel();
    });
    el("prevCardBtn").addEventListener("click", () => shiftResultCard(-1));
    el("nextCardBtn").addEventListener("click", () => shiftResultCard(1));
    el("importWatchBtn").addEventListener("click", importWatchPlaceholder);
    inputEl("woundImage").addEventListener("change", () => {
      setPreview(
        inputEl("woundImage"),
        document.getElementById("woundPreview")
      );
      updateModeIndicators();
    });
    inputEl("skinImage").addEventListener("change", () => {
      setPreview(
        inputEl("skinImage"),
        document.getElementById("skinPreview")
      );
      updateModeIndicators();
    });
    el("randomizeBtn").addEventListener("click", fillFromRandomProfile);
    el("submitBtn").addEventListener("click", () => void submitRequest());
    el("clearBtn").addEventListener("click", resetToDefaults);
    el("downloadLogBtn").addEventListener("click", downloadRunLog);
    el("recordStart").addEventListener("click", () => void startRecording());
    el("recordStop").addEventListener("click", stopRecording);
    el("checkHealthBtn").addEventListener(
      "click",
      () => void checkBackendHealth()
    );
    inputEl("audioFile").addEventListener("change", () => {
      if (inputEl("audioFile").files?.[0]) {
        state.recordedAudioBlob = null;
      }
      updateAudioStatus();
      updateModeIndicators();
    });
    el("clearTextBtn").addEventListener("click", () => {
      textareaEl("freeText").value = "";
      updateModeIndicators();
    });
    el("clearVoiceBtn").addEventListener("click", () => {
      inputEl("audioFile").value = "";
      state.recordedAudioBlob = null;
      updateAudioStatus();
      updateModeIndicators();
    });
    el("clearImageBtn").addEventListener("click", () => {
      inputEl("woundImage").value = "";
      inputEl("skinImage").value = "";
      setPreview(inputEl("woundImage"), el("woundPreview"));
      setPreview(inputEl("skinImage"), el("skinPreview"));
      updateModeIndicators();
    });
    el("freeText").addEventListener("input", updateModeIndicators);
    const watchedFields = [
      ...REQUIRED_RULES.map((x) => x.id),
      ...OPTIONAL_NUMERIC_RULES.map((x) => x.id),
      "backendUrl"
    ];
    for (const id of watchedFields) {
      el(id).addEventListener("input", updateRequiredGate);
    }
    updateModeIndicators();
    setActiveMode("text");
    document.addEventListener("keydown", (event) => {
      const inputPanel = document.getElementById("inputValuesPanel");
      const resultsPanel = document.getElementById("resultsWrapper");
      const interactionPanel = document.getElementById("interactionSheet");
      if (event.key !== "Escape") return;
      if (inputPanel && !inputPanel.hidden) closeInputValuesPanel();
      if (resultsPanel && !resultsPanel.hidden) closeResultsPanel();
      if (interactionPanel && !interactionPanel.hidden) closeInteractionPanel();
    });
  }

  // src/main.ts
  wireEvents();
  showResultCard(0);
  resetToDefaults();
})();
//# sourceMappingURL=app.js.map
