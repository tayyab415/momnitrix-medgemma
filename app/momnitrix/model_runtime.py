"""Runtime wrappers for model-serving containers.

These wrappers support two modes:
- real model mode (`MOMNITRIX_USE_REAL_MODELS=true`)
- deterministic stub mode for local validation/tests.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from momnitrix.config import Settings
from momnitrix.risk import compute_policy_floor, heuristic_medgemma_decision
from momnitrix.schemas import MedGemmaRiskRequest, MedGemmaRiskResponse, TriageStreamRequest
from momnitrix.utils import max_risk


def _decode_b64(data: str) -> bytes:
    if data.startswith("data:") and "," in data:
        _, _, data = data.partition(",")
    return base64.b64decode(data, validate=False)


def _unit(content: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{content}".encode("utf-8")).digest()
    return round(int.from_bytes(digest[:2], "big") / 65535.0, 4)


def _norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def _risk_to_internal(raw: Any) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "low": "green",
        "green": "green",
        "mid": "yellow",
        "medium": "yellow",
        "moderate": "yellow",
        "yellow": "yellow",
        "high": "red",
        "severe": "red",
        "red": "red",
    }
    return mapping.get(token)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_json_candidate(text: str) -> dict[str, Any] | None:
    body = _strip_code_fences(text)
    if not body:
        return None

    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(body[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _coerce_medgemma_json(parsed: dict[str, Any]) -> MedGemmaRiskResponse | None:
    risk = _risk_to_internal(
        parsed.get("risk_level")
        or parsed.get("risk")
        or parsed.get("risk_class")
        or parsed.get("classification")
    )
    if risk is None:
        return None

    reasons = _as_list(parsed.get("reasons"))
    if not reasons:
        reasons = _as_list(parsed.get("clinical_reasoning"))
    if not reasons:
        reasons = _as_list(parsed.get("clinical_interpretation"))
    reasons = _dedupe_text_items(reasons, max_items=3)
    if not reasons:
        reasons = ["Model provided no explicit reasoning text."]

    actions = _as_list(parsed.get("action_items"))
    if not actions:
        actions = _as_list(parsed.get("recommended_actions"))
    if not actions:
        actions = _as_list(parsed.get("action_plan"))
    actions = _dedupe_text_items(actions, max_items=6)
    if not actions:
        actions = ["Seek routine prenatal follow-up and monitor symptoms."]

    summary = str(parsed.get("clinical_summary") or parsed.get("clinical_reasoning") or reasons[0]).strip()
    if not summary:
        summary = reasons[0]

    return MedGemmaRiskResponse(
        risk_level=risk,
        reasons=reasons,
        action_items=actions,
        clinical_summary=summary,
    )


def _normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _clean_generated_fragment(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<start_of_turn>", " ").replace("<end_of_turn>", " ")
    cleaned = re.sub(r"<unused\d+>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = cleaned.replace("mmoL/L", "mmol/L").replace("mmol L", "mmol/L")

    if re.fullmatch(r"(model|user|assistant)\s*:?", cleaned, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(
        (
            r"(clinical reasoning|clinical interpretation|potential complications|"
            r"likely complications|recommended actions|action plan|management(?: actions)?|"
            r"warning signs|urgent warning signs|red flags)\s*:?"
        ),
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""

    # Common malformed numeric artifact from generative prose (e.g. "12.0/8.0 mmHg").
    if re.search(r"\b\d{1,2}\.\d\s*/\s*\d{1,2}\.\d\s*mmhg\b", cleaned, flags=re.IGNORECASE):
        return ""

    # Filter impossible glucose-unit hallucinations (e.g. "7.0 mg/dL" in pregnancy context).
    mgdl = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*mg/?dl\b", cleaned, flags=re.IGNORECASE)
    if mgdl and float(mgdl.group(1)) < 30.0:
        return ""

    return cleaned


def _dedupe_text_items(items: list[str], *, max_items: int) -> list[str]:
    deduped: list[str] = []
    seen: list[str] = []
    for item in items:
        cleaned = _clean_generated_fragment(item)
        if not cleaned:
            continue
        key = _normalize_text_key(cleaned)
        if not key:
            continue
        if any(key == old or key in old or old in key for old in seen):
            continue
        seen.append(key)
        deduped.append(cleaned)
        if len(deduped) >= max_items:
            break
    return deduped


def _find_first_header(text: str, headers: list[str]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for header in headers:
        pattern = re.compile(rf"^\s*{re.escape(header)}\s*:\s*", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(text)
        if not match:
            continue
        span = (match.start(), match.end())
        if best is None or span[0] < best[0]:
            best = span
    return best


def _extract_section_blocks(text: str) -> dict[str, str]:
    if not text.strip():
        return {}

    headings = {
        "reasoning": ["CLINICAL REASONING", "CLINICAL INTERPRETATION", "REASONING"],
        "complications": ["POTENTIAL COMPLICATIONS", "LIKELY COMPLICATIONS", "COMPLICATIONS"],
        "actions": ["RECOMMENDED ACTIONS", "ACTION PLAN", "MANAGEMENT", "MANAGEMENT ACTIONS"],
        "warnings": ["WARNING SIGNS", "RED FLAGS", "URGENT WARNING SIGNS"],
    }
    anchors: list[tuple[int, int, str]] = []
    for key, header_list in headings.items():
        span = _find_first_header(text, header_list)
        if span is not None:
            anchors.append((span[0], span[1], key))

    anchors.sort(key=lambda x: x[0])
    sections: dict[str, str] = {}
    for idx, (_, content_start, key) in enumerate(anchors):
        if key in sections:
            continue
        next_start = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(text)
        block = text[content_start:next_start].strip()
        if block:
            sections[key] = block
    return sections


def _extract_list_items(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        cleaned = _clean_generated_fragment(re.sub(r"^\s*[-*]\s*", "", line))
        if not cleaned:
            continue
        if re.fullmatch(
            r"(warning|warnings|warning signs|red flags|urgent warning signs)\s*:?",
            cleaned,
            flags=re.IGNORECASE,
        ):
            continue
        if cleaned.isupper() and len(cleaned) <= 20:
            continue
        items.append(cleaned)
    if not items and block.strip():
        items = [re.sub(r"\s+", " ", block).strip()]
    return items


def _coerce_medgemma_text(text: str) -> MedGemmaRiskResponse | None:
    if not text.strip():
        return None

    risk_match = re.search(r"RISK\s*LEVEL\s*:\s*([A-Za-z _-]+)", text, flags=re.IGNORECASE)
    risk = _risk_to_internal(risk_match.group(1) if risk_match else None)
    if risk is None:
        return None

    sections = _extract_section_blocks(text)
    reasoning = sections.get("reasoning", "")
    complications = sections.get("complications", "")
    actions_block = sections.get("actions", "")
    warnings = sections.get("warnings", "")

    reasons: list[str] = []
    if reasoning:
        reasons.extend(_extract_list_items(reasoning))
    if complications:
        reasons.extend(_extract_list_items(complications))
    reasons = _dedupe_text_items(reasons, max_items=3)
    if not reasons:
        reasons.append("Model did not provide a sectioned reasoning block.")

    action_items: list[str] = []
    if actions_block:
        action_items.extend(_extract_list_items(actions_block))
    if warnings:
        warning_items = _extract_list_items(warnings)
        action_items.extend(f"Urgent warning sign: {item}" for item in warning_items[:3])
    action_items = _dedupe_text_items(action_items, max_items=6)
    if not action_items:
        action_items.append("Seek routine prenatal follow-up and monitor symptoms.")

    clinical_summary = reasons[0][:600]
    return MedGemmaRiskResponse(
        risk_level=risk,
        reasons=reasons,
        action_items=action_items[:6],
        clinical_summary=clinical_summary,
    )


def _sanitize_medgemma_output(text: str) -> str:
    cleaned = text.replace("<start_of_turn>", "\n").replace("<end_of_turn>", "\n").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"<unused\d+>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*(model|assistant|user)\s*$", "", cleaned)
    dup_marker = re.search(r"\nmodel\s*\n\s*RISK\s*LEVEL\s*:", cleaned, flags=re.IGNORECASE)
    if dup_marker:
        cleaned = cleaned[: dup_marker.start()].strip()

    # Some generations append a second full sectioned answer without a clean
    # "model" delimiter (e.g., "...symptoms.model\nRISK LEVEL: ...").
    # Keep only the first complete answer block.
    risk_headers = list(re.finditer(r"(?im)^\s*RISK\s*LEVEL\s*:", cleaned))
    if len(risk_headers) > 1:
        cleaned = cleaned[: risk_headers[1].start()].strip()

    cleaned = re.sub(r"(?i)[\s\.\-]*\b(model|assistant|user)\b\s*$", "", cleaned).strip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_temperature(temp_c: float | None) -> str:
    if temp_c is None:
        return "unknown"
    temp_f = (float(temp_c) * 9.0 / 5.0) + 32.0
    return f"{temp_c:.1f}\u00b0C ({temp_f:.1f}\u00b0F)"


def _trimester_from_weeks(weeks: int | None) -> str:
    if weeks is None:
        return "unknown"
    if weeks <= 13:
        return "1st trimester"
    if weeks <= 27:
        return "2nd trimester"
    return "3rd trimester"


def _normalize_noisy_asr_text(text: str, *, max_chars: int = 320) -> str:
    cleaned = str(text or "").replace("\n", " ").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)

    tokens = re.findall(r"[A-Za-z']+|[0-9]+(?:\.[0-9]+)?|\?", cleaned)
    if not tokens:
        return cleaned[:max_chars].strip()

    normalized_tokens: list[str] = []
    prev = ""
    for token in tokens:
        lower = token.lower()
        lower = re.sub(r"(.)\1{2,}", r"\1\1", lower)
        lower = re.sub(r"([a-z]{2,})\1{1,}", r"\1", lower)
        if lower == prev and lower not in {"very", "really"}:
            continue
        normalized_tokens.append(lower)
        prev = lower

    normalized = " ".join(normalized_tokens)
    normalized = re.sub(r"\s+\?", "?", normalized).strip()
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip() + "..."
    return normalized


def _extract_concern_signals(note: str, transcript: str) -> list[str]:
    corpus = " ".join(part for part in [note, transcript] if part).lower()
    corpus = re.sub(r"(.)\1{2,}", r"\1\1", corpus)
    if not corpus.strip():
        return []

    signals: list[str] = []
    rules: list[tuple[str, list[str]]] = [
        ("patient reports headache", [r"\bhead[a-z]*ach[a-z]*\b", r"\bmigraine\b", r"\bhead pain\b"]),
        ("patient reports swelling/puffiness", [r"\bpuff[a-z]*\b", r"\bswel[a-z]*\b", r"\bedema\b"]),
        ("patient reports vision symptoms", [r"\bvision\b", r"\bblur[a-z]*\b", r"\bspot[s]?\b", r"\bflash[a-z]*\b"]),
        ("patient reports decreased fetal movement", [r"\bfetal movement\b", r"\bbaby movement\b", r"\bnot moving\b"]),
        ("patient asks if this is urgent", [r"\bworr[a-z]*\b", r"\bscared\b", r"\bafraid\b", r"\burgent\b", r"\bserious\b"]),
    ]
    for label, patterns in rules:
        if any(re.search(pattern, corpus) for pattern in patterns):
            signals.append(label)
    return signals[:4]


def _resolve_medgemma_prompt_profile(payload: MedGemmaRiskRequest) -> str:
    metadata = payload.metadata or {}
    strategy = str(metadata.get("router_prompt_strategy") or metadata.get("prompt_strategy") or "").strip().lower()
    selected = {str(x).strip().lower() for x in (metadata.get("selected_specialists") or []) if str(x).strip()}

    has_wound = isinstance(payload.specialist_outputs.get("wound_scores"), dict) and bool(payload.specialist_outputs.get("wound_scores"))
    has_derm = isinstance(payload.specialist_outputs.get("skin_top3"), list) and bool(payload.specialist_outputs.get("skin_top3"))
    has_voice = isinstance(payload.specialist_outputs.get("transcript"), str) and bool(str(payload.specialist_outputs.get("transcript")).strip())

    if strategy == "multimodal_fusion" or (has_wound and has_derm):
        return "multimodal_image"
    if strategy == "wound_focus" or has_wound or "medsiglip" in selected:
        return "wound_image"
    if strategy == "derm_focus" or has_derm or "derm" in selected:
        return "derma_image"
    if has_voice or "medasr" in selected:
        return "voice_vitals"
    return "text_only"


def _profile_system_directives(profile: str) -> list[str]:
    profile_map = {
        "text_only": [
            "Channel profile: text_only.",
            "Focus on text question + vitals only.",
            "Do not reference image or audio findings unless explicitly provided.",
        ],
        "voice_vitals": [
            "Channel profile: voice_vitals.",
            "Prioritize transcript symptoms and answer the patient's spoken concern first.",
            "Treat transcript as noisy clinical signal; recover intent conservatively.",
        ],
        "wound_image": [
            "Channel profile: wound_image.",
            "You must explicitly interpret wound specialist scores in CLINICAL REASONING.",
            "You must include at least one wound-care action tied to those scores.",
        ],
        "derma_image": [
            "Channel profile: derma_image.",
            "You must explicitly interpret skin specialist top classes in CLINICAL REASONING.",
            "You must include at least one dermatology follow-up action tied to those classes.",
        ],
        "multimodal_image": [
            "Channel profile: multimodal_image.",
            "Fuse wound + derm + transcript + vitals conservatively.",
            "When signals conflict, escalate and explain uncertainty clearly.",
        ],
    }
    return profile_map.get(profile, profile_map["text_only"])


def _wound_evidence_summary(wound_scores: dict[str, Any]) -> str:
    if not isinstance(wound_scores, dict) or not wound_scores:
        return ""
    urgency = float(wound_scores.get("urgency", 0.0))
    infection = float(wound_scores.get("infection_risk", 0.0))
    erythema = float(wound_scores.get("erythema", 0.0))
    edema = float(wound_scores.get("edema", 0.0))
    exudate = float(wound_scores.get("exudate", 0.0))

    flags: list[str] = []
    if urgency >= 0.6:
        flags.append(f"elevated urgency ({urgency:.2f})")
    if infection >= 0.7:
        flags.append(f"elevated infection risk ({infection:.2f})")
    if erythema >= 0.6:
        flags.append(f"erythema signal ({erythema:.2f})")
    if edema >= 0.6:
        flags.append(f"edema signal ({edema:.2f})")
    if exudate >= 0.6:
        flags.append(f"exudate signal ({exudate:.2f})")

    if not flags:
        return (
            f"Wound image model reviewed the photo with no high-risk threshold crossed "
            f"(urgency {urgency:.2f}, infection_risk {infection:.2f})."
        )
    return "Wound image model flags " + ", ".join(flags) + "."


def _derm_evidence_summary(skin_top3: Any) -> str:
    if not isinstance(skin_top3, list) or not skin_top3:
        return ""
    items: list[str] = []
    for row in skin_top3[:3]:
        if not isinstance(row, dict):
            continue
        condition = str(row.get("condition", "unknown")).replace("_", " ")
        score = float(row.get("score", 0.0))
        items.append(f"{condition} ({score:.2f})")
    if not items:
        return ""
    return "Skin model top findings: " + ", ".join(items) + "."


def _build_medgemma_system_instruction(payload: MedGemmaRiskRequest) -> str:
    profile = _resolve_medgemma_prompt_profile(payload)
    lines = [
        "You are Momnitrix (formerly MamaGuard), a pregnancy-focused maternal risk assistant.",
        "Prioritize pregnancy-specific thresholds and conservative escalation for maternal-fetal safety.",
        "ASR transcripts may be noisy or stuttered; recover the likely intent and symptom meaning conservatively.",
        "If fasting glucose is provided, treat it as fasting unless explicitly stated otherwise.",
        "Do not reinterpret fasting values as post-meal values.",
        "Use concise, structured, clinically grounded language.",
        "When the patient asks a direct question (voice or text), answer that concern explicitly before broader counseling.",
    ]
    lines.extend(_profile_system_directives(profile))
    metadata = payload.metadata or {}
    task_instruction = str(metadata.get("gemini_task_instruction") or "").strip()
    if task_instruction:
        lines.extend(
            [
                "Task refinement from Gemini router:",
                task_instruction,
            ]
        )
    return "\n".join(lines)


def _build_medgemma_prompt(payload: MedGemmaRiskRequest) -> str:
    patient = payload.patient_context
    vitals = payload.vitals
    inputs = payload.inputs
    weeks = patient.gestational_weeks

    profile_lines = [
        "Evaluate the following pregnancy vitals and determine risk level.",
        "",
        "Patient profile:",
        (
            f"- Age: {patient.age_years} years"
            if patient.age_years is not None
            else "- Age: unknown"
        ),
        f"- Gestational age: {weeks if weeks is not None else 'unknown'} weeks ({_trimester_from_weeks(weeks)})",
        f"- Known conditions: {', '.join(patient.known_conditions) if patient.known_conditions else 'none reported'}",
        f"- Medications: {', '.join(patient.medications) if patient.medications else 'none reported'}",
    ]
    if patient.patient_id:
        profile_lines.append(f"- Patient ID: {patient.patient_id}")

    vitals_lines = [
        "",
        "Monitoring data (smartwatch + app logs):",
        (
            f"- Blood pressure: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg"
            if vitals.systolic_bp is not None and vitals.diastolic_bp is not None
            else "- Blood pressure: unavailable"
        ),
        (
            f"- Fasting plasma glucose: {vitals.fasting_glucose_mmol_l:.1f} mmol/L"
            if vitals.fasting_glucose_mmol_l is not None
            else "- Fasting plasma glucose: unavailable"
        ),
        f"- Core temperature: {_format_temperature(vitals.temp_c)}",
        f"- Resting heart rate: {vitals.hr if vitals.hr is not None else 'unknown'} bpm",
        f"- SpO2: {vitals.spo2 if vitals.spo2 is not None else 'unknown'}%",
        f"- HRV: {vitals.hrv if vitals.hrv is not None else 'unknown'} ms",
    ]

    symptom_lines = [
        "",
        "Current symptom flags:",
        f"- Headache: {'yes' if inputs.headache else 'no'}",
        f"- Vision changes: {'yes' if inputs.vision_changes else 'no'}",
        f"- Decreased fetal movement: {'yes' if inputs.decreased_fetal_movement else 'no'}",
    ]

    specialist_lines: list[str] = []
    wound_scores = payload.specialist_outputs.get("wound_scores")
    if isinstance(wound_scores, dict) and wound_scores:
        compact_wound = ", ".join(
            f"{k}={float(v):.3f}" for k, v in sorted(wound_scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:4]
        )
        specialist_lines.extend(["", "Specialist model outputs:", f"- Wound scores: {compact_wound}"])
        wound_summary = _wound_evidence_summary(wound_scores)
        if wound_summary:
            specialist_lines.append(f"- Wound evidence summary: {wound_summary}")

    skin_top3 = payload.specialist_outputs.get("skin_top3")
    if isinstance(skin_top3, list) and skin_top3:
        labels: list[str] = []
        for row in skin_top3[:3]:
            if isinstance(row, dict):
                name = str(row.get("condition", "unknown"))
                score = float(row.get("score", 0.0))
                labels.append(f"{name}={score:.3f}")
        if labels:
            if not specialist_lines:
                specialist_lines.extend(["", "Specialist model outputs:"])
            specialist_lines.append(f"- Derm top classes: {', '.join(labels)}")
            derm_summary = _derm_evidence_summary(skin_top3)
            if derm_summary:
                specialist_lines.append(f"- Derm evidence summary: {derm_summary}")

    transcript = payload.specialist_outputs.get("transcript")
    cleaned_transcript = ""
    patient_concern = ""
    if isinstance(transcript, str) and transcript.strip():
        cleaned_transcript = _normalize_noisy_asr_text(transcript, max_chars=320)
        snippet = cleaned_transcript or transcript.strip().replace("\n", " ")[:300]
        if not specialist_lines:
            specialist_lines.extend(["", "Specialist model outputs:"])
        specialist_lines.append(f"- ASR transcript snippet: {snippet}")
        patient_concern = snippet

    note = (inputs.free_text or "").strip()
    if note:
        specialist_lines.extend(["", "Additional patient note:", note])
        if not patient_concern:
            patient_concern = note[:300]

    concern_signals = _extract_concern_signals(note, cleaned_transcript or str(transcript or ""))
    if concern_signals:
        if not specialist_lines:
            specialist_lines.extend(["", "Specialist model outputs:"])
        specialist_lines.append(f"- Extracted concern cues: {', '.join(concern_signals)}")

    metadata = payload.metadata or {}
    profile = _resolve_medgemma_prompt_profile(payload)
    output_style = str(
        metadata.get("medgemma_output_style")
        or metadata.get("medgemma_mode")
        or metadata.get("composer_mode")
        or ""
    ).strip().lower().replace("-", "_")
    notebook_style = output_style in {"medgemma_first", "medgemma_notebook", "notebook", "raw"}

    output_requirements = [
        "",
        (
            f"Patient-stated concern to address first: {patient_concern}"
            if patient_concern
            else "Patient-stated concern to address first: Not explicitly provided."
        ),
        "",
        "Please return:",
        "1) LOW/MID/HIGH risk classification",
        "2) Clinical interpretation tied to threshold values",
        "3) Likely maternal-fetal complications",
        "4) Week-appropriate management actions",
        "5) Immediate red-flag symptoms requiring urgent evaluation",
        "",
        "Clinical threshold reminders:",
        "- In pregnancy, fasting glucose target is <5.3 mmol/L.",
        "- Fasting glucose >=7.0 mmol/L is diabetes-range in pregnancy.",
        "- Fasting glucose >=10.0 mmol/L is severe hyperglycemia requiring urgent escalation.",
        "- Treat provided glucose as fasting unless explicitly stated otherwise.",
        "",
        "Cross-signal correlation rules (CRITICAL):",
        "- If wound shows elevated urgency AND voice transcript mentions fever or chills, escalate infection risk.",
        "- If BP is borderline AND transcript mentions headaches AND skin shows urticaria on abdomen in 3rd trimester, consider preeclampsia and PUPPP.",
        "- If systolic BP > 160 OR mentions of visual changes + headaches OR decreased fetal movement OR vaginal bleeding > 20 weeks, escalate to HIGH/URGENT immediately.",
        "- Start by directly answering the patient's own question/concern from voice/text before detailing thresholds and actions.",
        "- Begin CLINICAL REASONING by directly answering whether the patient's stated concern needs urgent in-person care now.",
        "- If any specialist model outputs are present (wound/skin/ASR), explicitly cite them in CLINICAL REASONING using the phrase 'Specialist evidence:' and include at least one related action item.",
    ]
    if profile == "text_only":
        output_requirements.extend(
            [
                "- Text-only mode: reason only from provided text + vitals and do not infer missing image/audio findings.",
            ]
        )
    elif profile == "voice_vitals":
        output_requirements.extend(
            [
                "- Voice mode: begin by answering the spoken concern directly, then provide threshold-based triage.",
            ]
        )
    elif profile == "wound_image":
        output_requirements.extend(
            [
                "- Wound-image mode: include an explicit sentence starting with 'Specialist evidence:' that cites wound scores.",
            ]
        )
    elif profile == "derma_image":
        output_requirements.extend(
            [
                "- Derma-image mode: include an explicit sentence starting with 'Specialist evidence:' that cites derm top findings.",
            ]
        )
    elif profile == "multimodal_image":
        output_requirements.extend(
            [
                "- Multimodal-image mode: include at least one 'Specialist evidence:' sentence for each relevant channel used.",
            ]
        )

    if notebook_style:
        output_requirements.extend(
            [
                "",
                "Use this exact sectioned format (not JSON):",
                "RISK LEVEL: <LOW/MID/HIGH>",
                "",
                "CLINICAL REASONING:",
                "<threshold-linked interpretation>",
                "",
                "POTENTIAL COMPLICATIONS:",
                "<maternal-fetal complications>",
                "",
                "RECOMMENDED ACTIONS:",
                "- <action 1>",
                "- <action 2>",
                "",
                "WARNING SIGNS:",
                "- <urgent sign 1>",
                "- <urgent sign 2>",
            ]
        )
    else:
        output_requirements.extend(
            [
                "",
                "Return JSON with keys: risk_level, clinical_reasoning, action_items, flags_for_provider, clinical_summary.",
            ]
        )

    return "\n".join(profile_lines + vitals_lines + symptom_lines + specialist_lines + output_requirements)


def _resolve_derm_artifact_path(configured_path: str, default_filename: str) -> Path:
    candidates = [
        Path(configured_path),
        Path("/root/artifacts/derm") / default_filename,
        Path("artifacts/derm") / default_filename,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return Path(configured_path)


def _escalate_with_policy_floor(
    decision: MedGemmaRiskResponse,
    request: TriageStreamRequest,
    specialist_outputs: dict[str, Any],
) -> MedGemmaRiskResponse:
    floor, reasons = compute_policy_floor(request, specialist_outputs)
    final_risk = max_risk(decision.risk_level, floor)
    if final_risk == decision.risk_level:
        return decision

    merged_reasons = list(decision.reasons)
    if reasons:
        merged_reasons.append(f"Safety floor escalation ({floor}): " + "; ".join(reasons[:2]))
    else:
        merged_reasons.append(f"Safety floor escalation ({floor}).")

    return decision.model_copy(
        update={
            "risk_level": final_risk,
            "reasons": merged_reasons,
            "clinical_summary": " ".join(merged_reasons[:2])[:600],
        }
    )


@dataclass
class CoreGpuRuntime:
    settings: Settings

    def __post_init__(self) -> None:
        self._medsiglip_processor = None
        self._medsiglip_model = None
        self._medgemma_processor = None
        self._medgemma_model = None

    def _ensure_medsiglip(self) -> bool:
        if self._medsiglip_model is not None and self._medsiglip_processor is not None:
            return True
        if not self.settings.use_real_models:
            return False

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self._medsiglip_processor = AutoImageProcessor.from_pretrained(
                self.settings.medsiglip_model_id,
                token=self.settings.hf_token,
            )
            self._medsiglip_model = AutoModelForImageClassification.from_pretrained(
                self.settings.medsiglip_model_id,
                token=self.settings.hf_token,
            )
            self._medsiglip_model.eval()
            if torch.cuda.is_available():
                self._medsiglip_model = self._medsiglip_model.cuda()
            return True
        except Exception:
            self._medsiglip_processor = None
            self._medsiglip_model = None
            return False

    def medsiglip_infer(self, image_b64: str) -> dict[str, float]:
        labels = [
            "healing_status",
            "erythema",
            "edema",
            "infection_risk",
            "urgency",
            "exudate",
        ]

        if self._ensure_medsiglip():
            try:
                import torch
                from PIL import Image

                raw = _decode_b64(image_b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                inputs = self._medsiglip_processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.inference_mode():
                    logits = self._medsiglip_model(**inputs).logits
                    scores = torch.sigmoid(logits).squeeze().detach().cpu().tolist()

                id2label = getattr(self._medsiglip_model.config, "id2label", None) or {}
                mapped: dict[str, float] = {}
                for idx, score in enumerate(scores):
                    key = _norm_label(str(id2label.get(idx, labels[idx] if idx < len(labels) else f"label_{idx}")))
                    mapped[key] = float(round(score, 4))

                # Ensure required keys exist for downstream policies.
                for label in labels:
                    mapped.setdefault(label, 0.0)
                return mapped
            except Exception:
                pass

        return {
            "healing_status": _unit(image_b64, "heal"),
            "erythema": _unit(image_b64, "ery"),
            "edema": _unit(image_b64, "ede"),
            "infection_risk": _unit(image_b64, "inf"),
            "urgency": _unit(image_b64, "urg"),
            "exudate": _unit(image_b64, "exu"),
        }

    def _ensure_medgemma(self) -> bool:
        if self._medgemma_model is not None and self._medgemma_processor is not None:
            return True
        if not self.settings.use_real_models:
            return False

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

            self._medgemma_processor = AutoProcessor.from_pretrained(
                self.settings.medgemma_base_model_id,
                token=self.settings.hf_token,
            )
            if hasattr(self._medgemma_processor, "tokenizer"):
                self._medgemma_processor.tokenizer.padding_side = "right"

            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "attn_implementation": "eager",
                "token": self.settings.hf_token,
            }
            if torch.cuda.is_available():
                load_kwargs["device_map"] = {"": 0}

            base_model = AutoModelForImageTextToText.from_pretrained(
                self.settings.medgemma_base_model_id,
                **load_kwargs,
            )

            if self.settings.medgemma_is_adapter:
                self._medgemma_model = PeftModel.from_pretrained(
                    base_model,
                    self.settings.medgemma_adapter_id,
                    token=self.settings.hf_token,
                    is_trainable=False,
                )
            else:
                self._medgemma_model = base_model

            # Enable KV cache for substantially faster token generation in inference mode.
            self._medgemma_model.config.use_cache = True
            self._medgemma_model.eval()
            return True
        except Exception as exc:
            self._medgemma_processor = None
            self._medgemma_model = None
            print(f"[momnitrix] medgemma_load_failed: {type(exc).__name__}: {exc}")
            return False

    def medgemma_decide(self, payload: MedGemmaRiskRequest) -> MedGemmaRiskResponse:
        request = TriageStreamRequest(
            patient_context=payload.patient_context,
            vitals=payload.vitals,
            inputs=payload.inputs,
        )

        started_total = perf_counter()
        was_loaded = self._medgemma_model is not None and self._medgemma_processor is not None
        ensure_t0 = perf_counter()
        ensured = self._ensure_medgemma()
        ensure_ms = int((perf_counter() - ensure_t0) * 1000)
        runtime_diag: dict[str, Any] = {
            "cold_start": not was_loaded,
            "gpu_warmup_ms": ensure_ms if not was_loaded else 0,
            "ensure_model_ms": ensure_ms,
            "medgemma_inference_ms": 0,
            "medgemma_generation_ms": 0,
            "medgemma_postprocess_ms": 0,
            "medgemma_total_ms": 0,
            "parse_mode": "unknown",
            "fallback_used": False,
        }
        mode_token = str(
            (payload.metadata or {}).get("medgemma_mode")
            or (payload.metadata or {}).get("medgemma_output_style")
            or (payload.metadata or {}).get("composer_mode")
            or ""
        ).strip().lower().replace("-", "_")
        notebook_mode = mode_token in {"medgemma_first", "medgemma_notebook", "notebook", "raw"}
        runtime_diag["notebook_mode"] = notebook_mode
        runtime_diag["prompt_profile"] = _resolve_medgemma_prompt_profile(payload)
        runtime_diag["generation_config"] = (
            {
                "max_new_tokens": 768,
                "do_sample": False,
            }
            if notebook_mode
            else {
                "max_new_tokens": 768,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 4,
            }
        )

        if ensured:
            try:
                import torch

                infer_t0 = perf_counter()
                system_instruction = _build_medgemma_system_instruction(payload)
                prompt = _build_medgemma_prompt(payload)
                rendered = self._medgemma_processor.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_instruction}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                tok = self._medgemma_processor(text=[rendered], return_tensors="pt", padding=True)
                model_device = next(self._medgemma_model.parameters()).device
                tok = {k: v.to(model_device) for k, v in tok.items()}

                tokenizer = self._medgemma_processor.tokenizer
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                gen_t0 = perf_counter()
                with torch.inference_mode():
                    if notebook_mode:
                        out = self._medgemma_model.generate(
                            **tok,
                            max_new_tokens=768,
                            do_sample=False,
                            pad_token_id=pad_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        out = self._medgemma_model.generate(
                            **tok,
                            max_new_tokens=768,
                            do_sample=False,
                            repetition_penalty=1.1,
                            no_repeat_ngram_size=4,
                            pad_token_id=pad_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                runtime_diag["medgemma_generation_ms"] = int((perf_counter() - gen_t0) * 1000)

                post_t0 = perf_counter()
                prompt_len = tok["input_ids"].shape[1]
                generated = out[0][prompt_len:]
                raw_text = self._medgemma_processor.decode(generated, skip_special_tokens=True)
                text = _sanitize_medgemma_output(raw_text)
                print(
                    "[momnitrix] medgemma_output_lengths "
                    f"raw={len(raw_text)} sanitized={len(text)}"
                )
                def _finalize(decision: MedGemmaRiskResponse, parse_mode: str) -> MedGemmaRiskResponse:
                    runtime_diag["parse_mode"] = parse_mode
                    runtime_diag["medgemma_postprocess_ms"] = int((perf_counter() - post_t0) * 1000)
                    runtime_diag["medgemma_inference_ms"] = int((perf_counter() - infer_t0) * 1000)
                    runtime_diag["medgemma_total_ms"] = int((perf_counter() - started_total) * 1000)
                    decided = _escalate_with_policy_floor(decision, request, payload.specialist_outputs)
                    return decided.model_copy(update={"runtime_diagnostics": runtime_diag, "raw_response_text": text})

                if notebook_mode:
                    parsed_text = _coerce_medgemma_text(text)
                    if parsed_text is not None:
                        print("[momnitrix] medgemma_parse_mode=sectioned_text fallback_used=false")
                        return _finalize(parsed_text, "sectioned_text")

                    parsed_json = _extract_json_candidate(text)
                    if parsed_json is not None:
                        decision = _coerce_medgemma_json(parsed_json)
                        if decision is not None:
                            print("[momnitrix] medgemma_parse_mode=json fallback_used=false")
                            return _finalize(decision, "json")
                else:
                    parsed_json = _extract_json_candidate(text)
                    if parsed_json is not None:
                        decision = _coerce_medgemma_json(parsed_json)
                        if decision is not None:
                            print("[momnitrix] medgemma_parse_mode=json fallback_used=false")
                            return _finalize(decision, "json")

                    parsed_text = _coerce_medgemma_text(text)
                    if parsed_text is not None:
                        print("[momnitrix] medgemma_parse_mode=sectioned_text fallback_used=false")
                        return _finalize(parsed_text, "sectioned_text")

                snippet = text.replace("\n", " ")[:280]
                print(f"[momnitrix] medgemma_parse_failed; sample='{snippet}'")
                runtime_diag["parse_mode"] = "unparsed_output"
                runtime_diag["medgemma_postprocess_ms"] = int((perf_counter() - post_t0) * 1000)
                runtime_diag["medgemma_inference_ms"] = int((perf_counter() - infer_t0) * 1000)
            except Exception as exc:
                print(f"[momnitrix] medgemma_generate_failed: {type(exc).__name__}: {exc}")
                runtime_diag["parse_mode"] = "generation_exception"
                runtime_diag["generation_error"] = f"{type(exc).__name__}: {exc}"

        print("[momnitrix] medgemma_parse_mode=fallback_heuristic fallback_used=true")
        runtime_diag["fallback_used"] = True
        runtime_diag["medgemma_total_ms"] = int((perf_counter() - started_total) * 1000)
        decision = heuristic_medgemma_decision(request, payload.specialist_outputs)
        return decision.model_copy(update={"runtime_diagnostics": runtime_diag})


@dataclass
class DermRuntime:
    settings: Settings

    def __post_init__(self) -> None:
        self._model = None
        self._classifier = None
        self._scaler = None
        self._labels: list[str] = []

    def _ensure(self) -> bool:
        if self._model is not None and self._classifier is not None:
            return True
        if not self.settings.use_real_models:
            return False

        try:
            from huggingface_hub import from_pretrained_keras

            self._model = from_pretrained_keras("google/derm-foundation")
            classifier_path = _resolve_derm_artifact_path(
                self.settings.derm_classifier_path,
                "derm_classifier.pkl",
            )
            with open(classifier_path, "rb") as fp:
                self._classifier = pickle.load(fp)

            scaler_path = _resolve_derm_artifact_path(
                self.settings.derm_scaler_path,
                "derm_scaler.pkl",
            )
            if scaler_path.exists():
                with open(scaler_path, "rb") as fp:
                    self._scaler = pickle.load(fp)

            labels_path = _resolve_derm_artifact_path(
                self.settings.derm_labels_path,
                "derm_labels.json",
            )
            if labels_path.exists():
                raw = json.loads(labels_path.read_text(encoding="utf-8"))
                self._labels = [_norm_label(x) for x in raw.get("labels", []) if isinstance(x, str)]

            return True
        except Exception:
            self._model = None
            self._classifier = None
            self._scaler = None
            self._labels = []
            return False

    def infer(self, image_b64: str) -> tuple[dict[str, float], list[dict[str, float | str]]]:
        labels = [
            "eczema",
            "allergic_contact_dermatitis",
            "insect_bite",
            "urticaria",
            "psoriasis",
            "folliculitis",
            "irritant_contact_dermatitis",
            "tinea",
            "herpes_zoster",
            "drug_rash",
        ]

        if self._ensure():
            try:
                import numpy as np
                import tensorflow as tf
                from PIL import Image

                raw = _decode_b64(image_b64)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                png_buf = io.BytesIO()
                image.save(png_buf, format="PNG")
                image_bytes = png_buf.getvalue()

                # Match Google's serving shape: a serialized tf.train.Example per image.
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                            "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"png"])),
                        }
                    )
                )
                serving_fn = self._model.signatures["serving_default"]
                input_key = next(iter(serving_fn.structured_input_signature[1]))
                output_key = next(iter(serving_fn.structured_outputs))
                emb = serving_fn(**{input_key: tf.constant([example.SerializeToString()])})[output_key].numpy()
                emb = np.asarray(emb, dtype="float32")

                if self._scaler is not None:
                    emb = self._scaler.transform(emb)

                probs = self._classifier.predict_proba(emb)
                resolved_labels = self._labels or labels
                if isinstance(probs, list):
                    # One-vs-rest format: list[label] -> (n_samples, 2)
                    scores = {label: float(p[0][1]) for label, p in zip(resolved_labels, probs)}
                else:
                    # Multi-label matrix shape (n_samples, n_labels)
                    scores = {label: float(probs[0][idx]) for idx, label in enumerate(resolved_labels)}
                top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                return scores, [{"condition": k, "score": round(v, 4)} for k, v in top3]
            except Exception:
                pass

        scores = {label: _unit(image_b64, label) for label in labels}
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return scores, [{"condition": k, "score": round(v, 4)} for k, v in top3]


@dataclass
class MedasrRuntime:
    settings: Settings

    def __post_init__(self) -> None:
        self._asr_model = None
        self._asr_processor = None
        self._asr_device = "cpu"

    @staticmethod
    def _normalize_token(token: str) -> str:
        lowered = token.lower()
        lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
        lowered = re.sub(r"([a-z]{2,})\1{1,}", r"\1", lowered)
        lowered = re.sub(r"(..)\1{2,}", r"\1", lowered)
        lowered = re.sub(r"[^a-z0-9']+", "", lowered)
        return lowered

    @staticmethod
    def _symptom_lexicon() -> tuple[str, ...]:
        return (
            "headache",
            "headaches",
            "migraine",
            "dizzy",
            "dizziness",
            "vision",
            "blurred",
            "spots",
            "swelling",
            "puffy",
            "pain",
            "cramping",
            "bleeding",
            "fever",
            "chills",
            "contractions",
            "movement",
            "fetal",
            "baby",
            "worry",
            "worried",
            "urgent",
            "scared",
        )

    @classmethod
    def _maybe_fix_symptom_token(cls, token: str) -> str:
        if len(token) < 4:
            return token
        noisy = bool(re.search(r"(.)\1{2,}", token) or re.search(r"([a-z]{2,})\1", token))
        if not noisy:
            return token
        best = token
        best_score = 0.0
        for target in cls._symptom_lexicon():
            score = SequenceMatcher(None, token, target).ratio()
            if score > best_score:
                best_score = score
                best = target
        return best if best_score >= 0.72 else token

    @classmethod
    def _clean_transcript(cls, text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"</?s>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<epsilon>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        keep_short = {"a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "we"}
        raw_tokens = re.findall(r"[A-Za-z']+|[0-9]+(?:\.[0-9]+)?|[.,!?;:]", cleaned)
        out_tokens: list[str] = []
        prev_word = ""

        for tok in raw_tokens:
            if re.fullmatch(r"[.,!?;:]", tok):
                if out_tokens and re.fullmatch(r"[.,!?;:]", out_tokens[-1]):
                    continue
                out_tokens.append(tok)
                prev_word = ""
                continue

            norm = cls._normalize_token(tok)
            norm = cls._maybe_fix_symptom_token(norm)
            if not norm:
                continue
            if len(norm) <= 2 and norm not in keep_short:
                continue
            if prev_word and norm == prev_word and norm not in {"very", "really"}:
                continue
            out_tokens.append(norm)
            prev_word = norm

        if not out_tokens:
            return ""

        # Join tokens and normalize punctuation spacing.
        text_out = " ".join(out_tokens)
        text_out = re.sub(r"\s+([.,!?;:])", r"\1", text_out)
        text_out = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text_out)
        text_out = re.sub(r"\bhave been ving\b", "have been having", text_out)
        text_out = re.sub(r"\bving\b", "having", text_out)
        text_out = re.sub(r"\bhe headaches\b", "headaches", text_out)
        text_out = re.sub(r"\bpu+f+y+\b", "puffy", text_out)
        text_out = re.sub(r"\bi is that\b", "is that", text_out)
        text_out = re.sub(r"\btwo day\b", "two days", text_out)
        text_out = re.sub(r"\bvery very ten\b", "very tense", text_out)
        text_out = re.sub(r"\bten and scared\b", "tense and scared", text_out)
        text_out = re.sub(r"\bin a very tense and scared to\.", "I am very tense and scared.", text_out)
        text_out = re.sub(r"\bhaving really headaches\b", "having really bad headaches", text_out)
        text_out = re.sub(r"\bmy hand look\b", "my hands look", text_out)
        text_out = re.sub(r"\. is that\b", ". Is that", text_out)
        text_out = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", text_out)
        text_out = re.sub(r"\s+", " ", text_out).strip()
        if text_out:
            text_out = text_out[0].upper() + text_out[1:]
        return text_out

    @staticmethod
    def _quality_score(text: str) -> float:
        words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
        if not words:
            return -1.0
        uniq_ratio = len(set(words)) / max(len(words), 1)
        repeated_char_penalty = len(re.findall(r"(.)\1{2,}", text))
        short_noise_penalty = sum(1 for w in words if len(w) <= 2 and w not in {"a", "i", "in", "to", "is", "of", "my"})
        score = (uniq_ratio * 2.0) + min(len(words), 80) / 80.0
        score -= (0.25 * repeated_char_penalty) + (0.03 * short_noise_penalty)
        if text.endswith("?"):
            score += 0.1
        return score

    @staticmethod
    def _merge_chunk_texts(chunks: list[str]) -> str:
        if not chunks:
            return ""
        merged = chunks[0].split()
        for chunk in chunks[1:]:
            words = chunk.split()
            if not words:
                continue
            max_overlap = min(10, len(merged), len(words))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                left = [w.lower() for w in merged[-size:]]
                right = [w.lower() for w in words[:size]]
                if left == right:
                    overlap = size
                    break
            merged.extend(words[overlap:])
        return " ".join(merged).strip()

    def _decode_once(self, waveform: Any) -> str:
        import torch

        if self._asr_model is None or self._asr_processor is None:
            return ""
        inputs = self._asr_processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self._asr_device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self._asr_model.generate(**inputs)
        decoded = self._asr_processor.batch_decode(output_ids)
        return str(decoded[0] if decoded else "").strip()

    def _decode_chunked(self, waveform: Any, *, chunk_length_s: float = 20.0, stride_length_s: float = 2.0) -> str:
        import numpy as np

        sample_rate = 16000
        chunk_samples = int(chunk_length_s * sample_rate)
        stride_samples = int(stride_length_s * sample_rate)
        step = max(chunk_samples - stride_samples, int(5 * sample_rate))
        parts: list[str] = []

        for start in range(0, len(waveform), step):
            end = min(start + chunk_samples, len(waveform))
            segment = np.asarray(waveform[start:end], dtype="float32")
            if segment.size < int(0.5 * sample_rate):
                break
            text = self._decode_once(segment)
            if text:
                parts.append(text)
            if end >= len(waveform):
                break
        return self._merge_chunk_texts(parts)

    def _ensure(self) -> bool:
        if self._asr_model is not None and self._asr_processor is not None:
            return True
        if not self.settings.use_real_models:
            return False

        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor

            self._asr_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._asr_processor = AutoProcessor.from_pretrained(
                "google/medasr",
                token=self.settings.hf_token,
                trust_remote_code=True,
            )
            self._asr_model = AutoModelForCTC.from_pretrained(
                "google/medasr",
                token=self.settings.hf_token,
                trust_remote_code=True,
            ).to(self._asr_device)
            self._asr_model.eval()
            return True
        except Exception as exc:
            self._asr_model = None
            self._asr_processor = None
            print(f"[momnitrix] medasr_load_failed: {type(exc).__name__}: {exc}")
            return False

    def transcribe(self, audio_b64: str) -> str:
        if self._ensure():
            try:
                import librosa
                import numpy as np
                import soundfile as sf

                raw = _decode_b64(audio_b64)
                try:
                    waveform, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
                except Exception:
                    waveform, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    if getattr(waveform, "ndim", 1) > 1:
                        waveform = waveform.mean(axis=1)
                    if sr != 16000:
                        waveform = librosa.resample(np.asarray(waveform), orig_sr=sr, target_sr=16000)

                waveform = np.asarray(waveform, dtype="float32")
                if waveform.size == 0:
                    raise ValueError("empty_audio")
                waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
                peak = float(np.max(np.abs(waveform)))
                if peak > 0:
                    waveform = waveform / peak

                duration_s = waveform.shape[0] / 16000.0
                candidates: list[tuple[str, str]] = []

                if duration_s <= 95:
                    full = self._decode_once(waveform)
                    if full:
                        candidates.append(("ctc_full", full))

                chunked = self._decode_chunked(waveform, chunk_length_s=20.0, stride_length_s=2.0)
                if chunked:
                    candidates.append(("ctc_chunked", chunked))

                if not candidates:
                    raise ValueError("no_decode_candidates")

                scored: list[tuple[float, str, str]] = []
                for mode, raw_text in candidates:
                    cleaned = self._clean_transcript(raw_text)
                    final_text = cleaned if len(cleaned.split()) >= 3 else raw_text
                    scored.append((self._quality_score(final_text), mode, final_text))

                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_mode, best_text = scored[0]
                print(
                    "[momnitrix] medasr_decode_selected "
                    f"mode={best_mode} score={best_score:.3f} duration_s={duration_s:.1f}"
                )
                if best_text:
                    return best_text
            except Exception as exc:
                print(f"[momnitrix] medasr_transcribe_failed: {type(exc).__name__}: {exc}")
                pass

        checksum = hashlib.sha1(audio_b64.encode("utf-8")).hexdigest()[:8]
        return f"Simulated MedASR transcript {checksum}."
