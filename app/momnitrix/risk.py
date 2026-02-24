"""Risk-policy and fallback MedGemma heuristics."""

from __future__ import annotations

import re
from typing import Any

from momnitrix.schemas import MedGemmaRiskResponse, TriageStreamRequest
from momnitrix.utils import max_risk

PREG_FASTING_GLUCOSE_TARGET_MMOL_L = 5.3
DIABETIC_FASTING_GLUCOSE_MMOL_L = 7.0
SEVERE_FASTING_GLUCOSE_MMOL_L = 10.0


def _trimester_label(gestational_weeks: int | None) -> str:
    if gestational_weeks is None:
        return "pregnancy"
    if gestational_weeks <= 13:
        return "first trimester"
    if gestational_weeks <= 27:
        return "second trimester"
    return "third trimester"


def _extract_fasting_glucose_mmol_l(request: TriageStreamRequest) -> float | None:
    if request.vitals.fasting_glucose_mmol_l is not None:
        return float(request.vitals.fasting_glucose_mmol_l)

    text = (request.inputs.free_text or "").strip()
    if not text:
        return None

    pattern = re.compile(
        r"fast(?:ing)?(?:\s+plasma)?\s+glucose[^0-9]{0,30}([0-9]+(?:\.[0-9]+)?)\s*(mmol/?l|mg/?dl)?",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None

    value = float(match.group(1))
    unit = (match.group(2) or "").lower().replace("/", "")
    if unit == "mgdl" or (not unit and value > 40):
        return round(value / 18.0, 2)
    return value


def _assess_fasting_glucose(
    request: TriageStreamRequest,
) -> tuple[str | None, str | None, str | None]:
    glucose = _extract_fasting_glucose_mmol_l(request)
    if glucose is None:
        return None, None, None

    trimester = _trimester_label(request.patient_context.gestational_weeks)
    target = PREG_FASTING_GLUCOSE_TARGET_MMOL_L
    if glucose >= SEVERE_FASTING_GLUCOSE_MMOL_L:
        return (
            "red",
            (
                f"Severe fasting glucose hyperglycemia ({glucose:.1f} mmol/L) in {trimester}; "
                f"pregnancy fasting target is <{target:.1f} mmol/L."
            ),
            "Seek urgent same-day obstetric evaluation for severe hyperglycemia.",
        )

    if glucose >= DIABETIC_FASTING_GLUCOSE_MMOL_L:
        return (
            "yellow",
            (
                f"Fasting glucose is in diabetes-range ({glucose:.1f} mmol/L) in {trimester}; "
                f"pregnancy fasting target is <{target:.1f} mmol/L."
            ),
            "Contact your OB/diabetes team within 24 hours to adjust glucose management.",
        )

    if glucose >= target:
        return (
            "yellow",
            (
                f"Fasting glucose ({glucose:.1f} mmol/L) is above pregnancy target "
                f"(<{target:.1f} mmol/L) in {trimester}."
            ),
            "Share fasting glucose logs with your prenatal team and repeat fasting checks.",
        )

    return None, None, None


def compute_policy_floor(request: TriageStreamRequest, specialist_outputs: dict[str, Any]) -> tuple[str, list[str]]:
    floor = "green"
    reasons: list[str] = []

    vitals = request.vitals
    inputs = request.inputs

    if vitals.systolic_bp is not None and vitals.diastolic_bp is not None:
        if vitals.systolic_bp >= 160 or vitals.diastolic_bp >= 110:
            floor = "red"
            reasons.append("Severely elevated blood pressure hard-stop.")
        elif vitals.systolic_bp >= 140 or vitals.diastolic_bp >= 90:
            floor = max_risk(floor, "yellow")
            reasons.append("Elevated blood pressure threshold crossed.")

    glucose_level, glucose_reason, _ = _assess_fasting_glucose(request)
    if glucose_level is not None and glucose_reason is not None:
        floor = max_risk(floor, glucose_level)
        reasons.append(glucose_reason)

    if inputs.headache and inputs.vision_changes:
        floor = "red"
        reasons.append("Headache + vision changes hard-stop.")

    if inputs.decreased_fetal_movement:
        floor = "red"
        reasons.append("Decreased fetal movement hard-stop.")

    wound = specialist_outputs.get("wound_scores") or {}
    urgency = float(wound.get("urgency", 0.0))
    infection = float(wound.get("infection_risk", 0.0))
    if urgency >= 0.6 or infection >= 0.7:
        floor = max_risk(floor, "yellow")
        reasons.append("Wound urgency/infection threshold crossed.")

    if vitals.temp_c is not None and vitals.temp_c >= 38.0:
        floor = max_risk(floor, "yellow")
        reasons.append("Fever threshold crossed.")

    skin_scores = specialist_outputs.get("skin_scores") or {}
    shingles = float(skin_scores.get("herpes_zoster", 0.0))
    drug_rash = float(skin_scores.get("drug_rash", 0.0))
    if shingles >= 0.6 or drug_rash >= 0.65:
        floor = max_risk(floor, "yellow")
        reasons.append("Dermatology high-risk class threshold crossed.")

    return floor, reasons


def heuristic_medgemma_decision(
    request: TriageStreamRequest,
    specialist_outputs: dict[str, Any],
) -> MedGemmaRiskResponse:
    risk = "green"
    reasons: list[str] = []
    actions: list[str] = []

    vitals = request.vitals
    inputs = request.inputs

    if vitals.systolic_bp is not None and vitals.diastolic_bp is not None:
        if vitals.systolic_bp >= 160 or vitals.diastolic_bp >= 110:
            risk = "red"
            reasons.append("Severe hypertension in pregnancy context.")
            actions.append("Call emergency services now.")
        elif vitals.systolic_bp >= 140 or vitals.diastolic_bp >= 90:
            risk = max_risk(risk, "yellow")
            reasons.append("Elevated blood pressure.")
            actions.append("Contact OB provider today.")

    glucose_level, glucose_reason, glucose_action = _assess_fasting_glucose(request)
    if glucose_level is not None and glucose_reason is not None:
        risk = max_risk(risk, glucose_level)
        reasons.append(glucose_reason)
    if glucose_action:
        actions.append(glucose_action)

    if inputs.headache and inputs.vision_changes:
        risk = "red"
        reasons.append("Headache with vision changes may indicate severe complications.")
        actions.append("Seek urgent same-day clinical assessment.")

    if inputs.decreased_fetal_movement:
        risk = "red"
        reasons.append("Reported decreased fetal movement.")
        actions.append("Contact labor triage immediately.")

    if vitals.temp_c is not None and vitals.temp_c >= 38.0:
        risk = max_risk(risk, "yellow")
        reasons.append("Fever may indicate infection.")
        actions.append("Monitor temperature and contact provider.")

    wound = specialist_outputs.get("wound_scores") or {}
    if wound:
        if float(wound.get("urgency", 0.0)) >= 0.6:
            risk = max_risk(risk, "yellow")
            reasons.append("Wound urgency score elevated.")
            actions.append("Share wound photo with care team today.")
        if float(wound.get("infection_risk", 0.0)) >= 0.7:
            risk = max_risk(risk, "yellow")
            reasons.append("Wound infection risk elevated.")
            actions.append("Watch for spreading redness, warmth, and drainage.")

    transcript = specialist_outputs.get("transcript")
    if transcript and any(k in transcript.lower() for k in ["bleeding", "dizzy", "faint", "pain worsening"]):
        risk = max_risk(risk, "yellow")
        reasons.append("Voice check-in includes concerning symptom language.")
        actions.append("Use urgent nurse line if symptoms are worsening.")

    if not reasons:
        reasons.append("No acute flags detected from submitted signals.")
    if not actions:
        actions.append("Continue routine monitoring and daily check-ins.")

    clinical_summary = " ".join(reasons[:3])
    return MedGemmaRiskResponse(
        risk_level=risk,
        reasons=reasons,
        action_items=actions,
        clinical_summary=clinical_summary,
    )
