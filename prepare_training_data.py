#!/usr/bin/env python3
"""Prepare maternal health instruction-tuning data for MedGemma/Gemma chat template.

Pipeline:
1) Load and validate CSV.
2) Add synthetic pregnancy context.
3) Build varied instruction prompts.
4) Generate rich responses (Gemini optional) with robust fallback templates.
5) Format in Gemma 3 chat template.
6) Build HuggingFace dataset + train/test split + JSONL exports.
7) Run quality validation and save summaries/examples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm


RISK_CANONICAL = {
    "low risk": "LOW",
    "mid risk": "MID",
    "high risk": "HIGH",
    "low": "LOW",
    "mid": "MID",
    "high": "HIGH",
}

DISALLOWED_PHRASES = ["i'm an ai", "consult your doctor", "disclaimer"]
REQUIRED_RESPONSE_SECTIONS = [
    "RISK LEVEL:",
    "POTENTIAL COMPLICATIONS",
    "RECOMMENDED ACTIONS",
    "WARNING SIGNS",
]


@dataclass
class SyntheticContext:
    gestational_week: int
    trimester: str
    gravida: int
    para: int
    bmi_category: str


@dataclass
class SampleRecord:
    row_index: int
    risk_level: str
    instruction: str
    response: str
    text: str


class GeminiEnricher:
    """Wrapper around google-genai SDK with retries and graceful fallback support."""

    def __init__(
        self,
        model_name: str,
        thinking_level: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
    ) -> None:
        self.enabled = False
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        self.client = None
        self.types = None
        self.api_error_type = Exception

        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import APIError

            self.client = genai.Client()
            self.types = types
            self.api_error_type = APIError
            self.enabled = True
        except ImportError as exc:
            print(f"[WARN] google-genai not installed or import failed: {exc}")
            print("[WARN] Falling back to template-based response generation.")
        except Exception as exc:
            print(f"[WARN] Gemini client initialization failed: {exc}")
            print("[WARN] Falling back to template-based response generation.")

    def generate(self, prompt: str, retries: int = 3, sleep_between_calls: float = 0.5) -> str | None:
        if not self.enabled or self.client is None or self.types is None:
            return None

        backoff_schedule = [2, 4, 8]
        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_output_tokens=self.max_output_tokens,
                        thinking_config=self.types.ThinkingConfig(
                            thinking_level=self.thinking_level,
                        ),
                    ),
                )
                if sleep_between_calls > 0:
                    time.sleep(sleep_between_calls)

                text = (response.text or "").strip()
                return text if text else None
            except self.api_error_type as exc:
                if attempt >= retries:
                    print(f"[ERROR] Gemini API error after retries: {exc}")
                    return None
                delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                print(f"[WARN] Gemini API error: {exc}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as exc:
                if attempt >= retries:
                    print(f"[ERROR] Gemini call failed after retries: {exc}")
                    return None
                delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                print(f"[WARN] Unexpected Gemini error: {exc}. Retrying in {delay}s...")
                time.sleep(delay)

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare maternal health training data.")
    parser.add_argument("--input", default="Maternal Health Risk Data Set.csv", help="Path to input CSV")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--use-gemini", action="store_true", help="Enable Gemini enrichment")
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview", help="Gemini model name")
    parser.add_argument(
        "--thinking-level",
        default="LOW",
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Gemini thinking level",
    )
    parser.add_argument(
        "--checkpoint",
        default="./enrichment_checkpoint.json",
        help="Path to checkpoint file for resume",
    )
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=800,
        help="Minimum acceptable Gemini response length in characters",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=2000,
        help="Maximum acceptable Gemini response length in characters",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalize_risk_label(raw_label: str) -> str:
    key = str(raw_label).strip().lower()
    if key not in RISK_CANONICAL:
        raise ValueError(f"Unknown RiskLevel value: {raw_label}")
    return RISK_CANONICAL[key]


def validate_and_describe_dataframe(df: pd.DataFrame) -> None:
    print("\n=== STEP 1: DATASET VALIDATION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nDtypes:")
    print(df.dtypes)

    print("\nClass distribution (RiskLevel):")
    print(df["RiskLevel"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isnull().sum())

    suspicious_hr_mask = df["HeartRate"] < 40
    fever_mask = df["BodyTemp"] > 101.0

    suspicious_count = int(suspicious_hr_mask.sum())
    fever_count = int(fever_mask.sum())

    if suspicious_count > 0:
        print(f"[WARN] Found {suspicious_count} rows with HeartRate < 40 bpm (kept in dataset)")
    else:
        print("[OK] No rows with HeartRate < 40 bpm")

    if fever_count > 0:
        print(f"[INFO] Found {fever_count} fever-case rows with BodyTemp > 101°F")
    else:
        print("[INFO] No fever-case rows with BodyTemp > 101°F")

    print("\nSummary statistics:")
    print(df.describe(include="all"))


def make_row_rng(seed: int, row_index: int) -> random.Random:
    return random.Random(seed * 1_000_003 + row_index)


def derive_trimester(gestational_week: int) -> str:
    if gestational_week <= 13:
        return "1st trimester"
    if gestational_week <= 27:
        return "2nd trimester"
    return "3rd trimester"


def sample_bmi_category(age: int, rng: random.Random) -> str:
    if age < 20:
        categories = ["normal", "overweight", "obese"]
        weights = [0.6, 0.3, 0.1]
    elif age < 35:
        categories = ["normal", "overweight", "obese"]
        weights = [0.45, 0.35, 0.20]
    else:
        categories = ["normal", "overweight", "obese"]
        weights = [0.30, 0.40, 0.30]

    return rng.choices(categories, weights=weights, k=1)[0]


def generate_synthetic_context(age: int, row_index: int, seed: int) -> SyntheticContext:
    rng = make_row_rng(seed, row_index)

    gest_week = rng.randint(8, 40)
    trimester = derive_trimester(gest_week)

    gravida = rng.choices([1, 2, 3, 4, 5], weights=[0.40, 0.30, 0.15, 0.10, 0.05], k=1)[0]
    para = rng.randint(0, max(0, gravida - 1))
    bmi_category = sample_bmi_category(age, rng)

    return SyntheticContext(
        gestational_week=gest_week,
        trimester=trimester,
        gravida=gravida,
        para=para,
        bmi_category=bmi_category,
    )


def build_instruction_prompt(row: pd.Series, context: SyntheticContext, row_index: int, seed: int) -> str:
    rng = make_row_rng(seed + 91, row_index)

    templates = [
        (
            "Assess maternal health risk for the following pregnant patient:\n\n"
            "Demographics:\n"
            "- Age: {age} years\n"
            "- Gravida {gravida}, Para {para}\n"
            "- BMI category: {bmi}\n"
            "- Gestational week: {gw} ({trimester})\n\n"
            "Current vitals (from wearable + manual entry):\n"
            "- Systolic blood pressure: {sbp} mmHg\n"
            "- Diastolic blood pressure: {dbp} mmHg\n"
            "- Blood sugar (fasting): {bs} mmol/L\n"
            "- Body temperature: {temp}°F\n"
            "- Resting heart rate: {hr} bpm\n\n"
            "Based on these readings, provide:\n"
            "1. Risk level classification (LOW, MID, or HIGH)\n"
            "2. Clinical reasoning citing specific values and thresholds\n"
            "3. Pregnancy complications these patterns may indicate\n"
            "4. Recommended actions appropriate for gestational week {gw}\n"
            "5. Warning signs the patient should watch for"
        ),
        (
            "Evaluate the following pregnancy vitals and determine risk level:\n\n"
            "Patient profile:\n"
            "- {age} years old, G{gravida}P{para}\n"
            "- Gestation: week {gw} ({trimester})\n"
            "- BMI group: {bmi}\n\n"
            "Monitoring data (smartwatch + app logs):\n"
            "- BP: {sbp}/{dbp} mmHg\n"
            "- Fasting glucose: {bs} mmol/L\n"
            "- Body temp: {temp}°F\n"
            "- Resting pulse: {hr} bpm\n\n"
            "Please return:\n"
            "1) LOW/MID/HIGH risk classification\n"
            "2) Clinical interpretation tied to threshold values\n"
            "3) Likely maternal-fetal complications\n"
            "4) Week-{gw} appropriate management actions\n"
            "5) Immediate red-flag symptoms"
        ),
        (
            "A {age}-year-old patient at {gw} weeks gestation presents with the following readings:\n\n"
            "Obstetric context:\n"
            "- Gravida {gravida}, Para {para}\n"
            "- Trimester: {trimester}\n"
            "- BMI category: {bmi}\n\n"
            "Vitals collected via wearable and manual input:\n"
            "- Systolic BP: {sbp} mmHg\n"
            "- Diastolic BP: {dbp} mmHg\n"
            "- Blood sugar: {bs} mmol/L\n"
            "- Temperature: {temp}°F\n"
            "- Heart rate: {hr} bpm\n\n"
            "Provide a structured maternal risk assessment including:\n"
            "1. Risk class (LOW/MID/HIGH)\n"
            "2. Reasoning using exact readings and thresholds\n"
            "3. Potential pregnancy complications\n"
            "4. Action plan suitable for week {gw}\n"
            "5. Warning signs requiring urgent care"
        ),
        (
            "Review these maternal monitoring readings and provide a risk assessment:\n\n"
            "Demographics and pregnancy context:\n"
            "- Age: {age}\n"
            "- G{gravida}P{para}\n"
            "- Gestational age: {gw} weeks ({trimester})\n"
            "- BMI status: {bmi}\n\n"
            "Latest clinical inputs:\n"
            "- Blood pressure: {sbp}/{dbp} mmHg\n"
            "- Fasting blood sugar: {bs} mmol/L\n"
            "- Body temperature: {temp}°F\n"
            "- Resting heart rate: {hr} bpm\n\n"
            "Include in your response:\n"
            "1. LOW/MID/HIGH classification\n"
            "2. Clear rationale linked to values\n"
            "3. Relevant obstetric complications\n"
            "4. Gestational-age-specific recommended actions\n"
            "5. Key warning signs to monitor"
        ),
    ]

    template = rng.choice(templates)
    return template.format(
        age=int(row["Age"]),
        gravida=context.gravida,
        para=context.para,
        bmi=context.bmi_category,
        gw=context.gestational_week,
        trimester=context.trimester,
        sbp=int(row["SystolicBP"]),
        dbp=int(row["DiastolicBP"]),
        bs=float(row["BS"]),
        temp=float(row["BodyTemp"]),
        hr=int(row["HeartRate"]),
    )


def build_gemini_prompt(row: pd.Series, context: SyntheticContext, risk_level: str) -> str:
    return (
        "You are a maternal-fetal medicine specialist creating training data for a medical AI system. "
        "Given these vital signs, write a clinical risk assessment response.\n\n"
        f"Patient: {int(row['Age'])}-year-old, G{context.gravida}P{context.para}, "
        f"gestational week {context.gestational_week}, BMI: {context.bmi_category}\n"
        "Vitals:\n"
        f"- BP: {int(row['SystolicBP'])}/{int(row['DiastolicBP'])} mmHg\n"
        f"- Blood Sugar: {float(row['BS'])} mmol/L\n"
        f"- Temperature: {float(row['BodyTemp'])}°F\n"
        f"- Heart Rate: {int(row['HeartRate'])} bpm\n\n"
        f"CORRECT RISK CLASSIFICATION: {risk_level} RISK\n\n"
        "Write a response that includes ALL of the following:\n\n"
        f"1. Start with \"RISK LEVEL: {risk_level}\" on the first line\n\n"
        "2. CLINICAL REASONING (2-3 sentences): Explain why this is the stated risk by referencing "
        "specific vital sign values against clinical thresholds. Use these thresholds:\n"
        "   - Normal BP in pregnancy: <130/85 mmHg\n"
        "   - Gestational hypertension: ≥140/90 mmHg\n"
        "   - Severe preeclampsia: ≥160/110 mmHg\n"
        "   - Normal fasting glucose: <5.1 mmol/L\n"
        "   - GDM threshold: ≥5.1 mmol/L fasting, ≥10.0 mmol/L 1-hr post-load\n"
        "   - Normal pregnancy HR: 60-100 bpm\n"
        "   - Fever: ≥100.4°F\n\n"
        "3. POTENTIAL COMPLICATIONS (1-2 sentences): Name likely obstetric complications linked to the pattern.\n\n"
        "4. RECOMMENDED ACTIONS (3-4 bullet points): Specific, actionable, and appropriate for current gestational week.\n\n"
        "5. WARNING SIGNS (2-3 items): Symptoms requiring urgent care.\n\n"
        "FORMATTING RULES:\n"
        "- Keep total response between 150-300 words\n"
        "- Use clinical but accessible patient-facing language\n"
        "- Do NOT include AI disclaimers or doctor-consult disclaimers\n"
        "- Do NOT use markdown headers with #\n"
        f"- The risk level MUST match {risk_level}\n"
        "- Vary sentence structure and wording so each output is unique"
    )


def trimester_context_sentence(gest_week: int) -> str:
    if gest_week <= 13:
        return "in the first trimester, early baseline stabilization and nausea/hydration management are key"
    if gest_week <= 27:
        return "in the second trimester, placental growth and glucose screening timing make close trend monitoring important"
    return "in the third trimester, complications can escalate faster and fetal surveillance becomes more critical"


def build_fallback_response(
    row: pd.Series,
    context: SyntheticContext,
    risk_level: str,
    row_index: int,
    seed: int,
) -> str:
    rng = make_row_rng(seed + 207, row_index)

    age = int(row["Age"])
    sbp = int(row["SystolicBP"])
    dbp = int(row["DiastolicBP"])
    bs = float(row["BS"])
    temp = float(row["BodyTemp"])
    hr = int(row["HeartRate"])

    reasons: list[str] = []
    complications: list[str] = []
    warning_signs: list[str] = []
    low_contradictions: list[str] = []

    bp_normal = 90 <= sbp < 130 and 60 <= dbp < 85
    glucose_above_normal = bs >= 5.1
    fever_present = temp >= 100.4
    hr_abnormal = hr < 60 or hr > 100
    hypotension_present = sbp < 90 or dbp < 60
    severe_hypotension = sbp < 85 or dbp < 55

    if risk_level == "HIGH":
        if sbp >= 140 or dbp >= 90:
            reasons.append("blood pressure is in a hypertensive range, increasing concern for gestational hypertension or preeclampsia")
            complications.extend(["preeclampsia", "gestational hypertension", "fetal growth restriction"])
            warning_signs.extend(["severe headache", "visual disturbance", "right upper abdominal pain"])
        if bs >= 11.1:
            reasons.append("blood sugar is significantly elevated and strongly suggests poorly controlled gestational diabetes")
            complications.extend(["gestational diabetes", "macrosomia", "preterm birth"])
            warning_signs.extend(["persistent excessive thirst", "reduced fetal movement"])
        elif 7.8 <= bs <= 11.0:
            reasons.append("glucose is above target and indicates impaired tolerance with likely gestational diabetes")
            complications.extend(["gestational diabetes", "polyhydramnios"])
            warning_signs.extend(["marked fatigue", "polyuria"])
        if temp >= 100.4:
            reasons.append("fever suggests possible infection with risk of chorioamnionitis")
            complications.extend(["chorioamnionitis", "preterm labor"])
            warning_signs.extend(["chills", "uterine tenderness", "foul-smelling discharge"])
        if age >= 35:
            reasons.append("advanced maternal age adds independent obstetric risk")
            complications.extend(["placental complications"])
        if age <= 17:
            reasons.append("adolescent pregnancy is associated with higher preterm and hypertensive risk")
            complications.extend(["preterm labor"])
        if hr < 50 or hr > 100:
            reasons.append("heart rate is outside expected pregnancy range and warrants cardiac evaluation")
            complications.extend(["maternal hemodynamic instability"])
            warning_signs.extend(["syncope", "chest pain", "palpitations"])
        if context.gestational_week >= 28:
            reasons.append("third-trimester findings are particularly concerning because deterioration can occur rapidly")

    elif risk_level == "MID":
        mild_flags = 0
        if 130 <= sbp <= 139 or 81 <= dbp <= 89:
            mild_flags += 1
            reasons.append("blood pressure is elevated but not yet in hypertensive diagnostic range")
            complications.extend(["progression to gestational hypertension"])
            warning_signs.extend(["new persistent headache", "visual blurring"])
        if 5.1 <= bs <= 11.0:
            mild_flags += 1
            if bs >= 7.8:
                reasons.append("glucose is moderately elevated and suggests impaired tolerance with likely gestational diabetes")
            else:
                reasons.append("fasting glucose is above normal and warrants formal glucose screening")
            complications.extend(["possible gestational diabetes"])
            warning_signs.extend(["increasing thirst", "increasing urination"])
        if 30 <= age <= 34:
            mild_flags += 1
            reasons.append("maternal age in the early-thirties range can modestly increase risk when combined with metabolic or blood pressure changes")
            complications.extend(["progression to gestational hypertension", "gestational diabetes risk"])
        if temp >= 100.4:
            mild_flags += 1
            reasons.append("fever adds infectious risk that can destabilize pregnancy")
            complications.extend(["maternal infection", "preterm contractions"])
            warning_signs.extend(["chills", "abdominal pain"])
        if hr < 60 or hr > 100:
            mild_flags += 1
            reasons.append("pulse trend is outside expected pregnancy-adjusted resting range")
            complications.extend(["maternal cardiovascular strain"])
            warning_signs.extend(["dizziness", "palpitations"])

        if mild_flags >= 2:
            reasons.append("the combination of multiple borderline abnormalities raises cumulative maternal-fetal risk")

    else:  # LOW
        if bp_normal:
            reasons.append(
                f"blood pressure of {sbp}/{dbp} mmHg is within expected pregnancy range (about 90-129 systolic and 60-84 diastolic)"
            )
        else:
            reasons.append(
                f"blood pressure of {sbp}/{dbp} mmHg is outside ideal pregnancy range and should be trended closely"
            )

        if glucose_above_normal:
            reasons.append(
                f"fasting glucose of {bs:.1f} mmol/L is above the normal fasting threshold of 5.1 and needs closer metabolic follow-up"
            )
            complications.extend(["possible gestational diabetes"])
            warning_signs.extend(["increasing thirst", "increasing urination"])
        else:
            reasons.append(
                f"fasting glucose of {bs:.1f} mmol/L is within normal fasting target for pregnancy"
            )

        if fever_present:
            low_contradictions.append("temperature is in fever range")
            complications.extend(["maternal infection risk"])
            warning_signs.extend(["chills", "abdominal pain"])
        if hypotension_present:
            if severe_hypotension:
                low_contradictions.append("blood pressure is in a marked hypotensive range")
            else:
                low_contradictions.append("blood pressure is in a hypotensive range")
            complications.extend(["maternal hemodynamic instability"])
            warning_signs.extend(["dizziness", "syncope"])
        if hr_abnormal:
            low_contradictions.append("heart rate is outside expected resting pregnancy range")
            complications.extend(["maternal cardiovascular strain"])
            warning_signs.extend(["palpitations", "chest pain"])
        if age >= 41:
            low_contradictions.append("advanced maternal age increases baseline obstetric risk")
            complications.extend(["placental complications", "hypertensive disorders"])
        elif age <= 16:
            low_contradictions.append("very young maternal age increases preterm and hypertensive risk")
            complications.extend(["preterm labor risk"])
        if bs > 7.8:
            low_contradictions.append("glucose level is substantially elevated for fasting pregnancy targets")
            complications.extend(["gestational diabetes"])

        if low_contradictions:
            reasons.append(
                "while this row is labeled low risk, specific findings warrant closer surveillance: "
                + "; ".join(dict.fromkeys(low_contradictions))
            )
        else:
            reasons.append("overall pattern supports low immediate maternal-fetal risk with routine surveillance")

        if not complications:
            complications.append("no immediate complications indicated")
        warning_signs.extend(["persistent severe headache", "vaginal bleeding", "reduced fetal movement"])

    if not reasons:
        reasons.append("the overall clinical pattern is most consistent with the assigned risk label")
    if not complications:
        complications = ["preeclampsia", "gestational diabetes", "preterm labor risk"] if risk_level != "LOW" else ["no immediate complications indicated"]
    if not warning_signs:
        warning_signs = ["severe headache", "visual changes", "decreased fetal movement"]

    reasoning_variants_general = [
        "Clinical Assessment: {reasons}. At gestational week {week}, {trimester_note}.",
        "Clinical Assessment: {reasons}. Given week {week} of pregnancy, {trimester_note}.",
        "Clinical Assessment: {reasons}. At {week} weeks gestation, {trimester_note}.",
    ]
    low_reasoning_variants = [
        "Clinical Assessment: {reasons}. At gestational week {week}, continue routine prenatal monitoring with attention to trend changes.",
        "Clinical Assessment: {reasons}. At {week} weeks gestation, findings support ongoing surveillance and prevention-focused care.",
        "Clinical Assessment: {reasons}. Given the current gestational stage (week {week}), prioritize consistency in monitoring and early escalation for changes.",
        "Clinical Assessment: {reasons}. At week {week}, maternal-fetal status appears stable overall, with follow-up intensity adjusted to any flagged values.",
    ]
    complication_variants = [
        "Potential Complications: {complications}.",
        "Potential Complications: These findings can be associated with {complications}.",
        "Potential Complications: Monitor for progression toward {complications}.",
    ]
    complication_variants_low = [
        "Potential Complications: {complications}.",
        "Potential Complications: Current data do not indicate immediate severe complications, but monitor for {complications}.",
    ]

    action_bank_high = [
        "Arrange urgent obstetric review within 24 hours with repeat BP and urine protein assessment",
        "Initiate frequent home BP monitoring twice daily and log readings in the app",
        "Order targeted labs (CBC, liver enzymes, creatinine) and fetal growth/wellbeing assessment",
        "Schedule or expedite glucose tolerance and diabetes management planning if hyperglycemia persists",
        "Advise hydration, rest, and strict symptom tracking with immediate escalation for worsening signs",
    ]
    action_bank_mid = [
        "Repeat blood pressure checks every 48-72 hours and trend results",
        "Schedule formal glucose screening during this gestational window",
        "Review nutrition and activity plan tailored for pregnancy",
        "Increase monitoring frequency in the app for temperature, pulse, and symptoms",
        "Book earlier follow-up prenatal visit to reassess trends",
    ]
    action_bank_low = [
        "Continue routine prenatal visits and standard trimester screening",
        "Maintain balanced nutrition and hydration with regular light activity",
        "Track BP, glucose, and heart rate trends weekly in the app",
        "Follow fetal movement awareness guidance as gestation advances",
        "Keep sleep and stress routines stable to support cardiovascular health",
    ]

    if risk_level == "HIGH":
        action_pool = action_bank_high
    elif risk_level == "MID":
        action_pool = action_bank_mid
    else:
        action_pool = action_bank_low

    actions = rng.sample(action_pool, k=min(4, len(action_pool)))
    warnings = rng.sample(list(dict.fromkeys(warning_signs)), k=min(3, len(set(warning_signs))))

    reasons_text = "; ".join(dict.fromkeys(reasons))
    complications_text = ", ".join(dict.fromkeys(complications))
    warning_text = ", ".join(warnings)

    if risk_level == "LOW":
        assessment = rng.choice(low_reasoning_variants).format(
            reasons=reasons_text,
            week=context.gestational_week,
        )
        complication_line = rng.choice(complication_variants_low).format(complications=complications_text)
    else:
        assessment = rng.choice(reasoning_variants_general).format(
            reasons=reasons_text,
            week=context.gestational_week,
            trimester_note=trimester_context_sentence(context.gestational_week),
        )
        complication_line = rng.choice(complication_variants).format(complications=complications_text)

    response = (
        f"RISK LEVEL: {risk_level}\n\n"
        f"{assessment}\n\n"
        f"{complication_line}\n\n"
        "Recommended Actions:\n"
        f"- {actions[0]}\n"
        f"- {actions[1]}\n"
        f"- {actions[2]}\n"
        f"- {actions[3]}\n\n"
        f"Warning Signs: Watch for {warning_text}. Seek immediate medical attention if these occur."
    )

    return response


def format_gemma_chat(user_instruction: str, model_response: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{user_instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{model_response}<end_of_turn>"
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "processed" not in data or not isinstance(data["processed"], dict):
            return {"processed": {}}
        return data
    except Exception as exc:
        print(f"[WARN] Failed to load checkpoint ({path}): {exc}. Starting fresh.")
        return {"processed": {}}


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def ensure_response_quality(response: str, risk_level: str) -> str:
    text = (response or "").strip()
    if not text:
        return f"RISK LEVEL: {risk_level}\n\nClinical Assessment: Response generation failed; fallback summary required."

    if not text.upper().startswith("RISK LEVEL:"):
        text = f"RISK LEVEL: {risk_level}\n\n{text}"

    first_line = text.splitlines()[0].upper()
    if risk_level not in first_line:
        lines = text.splitlines()
        lines[0] = f"RISK LEVEL: {risk_level}"
        text = "\n".join(lines)

    return text


def contains_disallowed_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in DISALLOWED_PHRASES)


def evaluate_response_quality(
    response: str,
    risk_level: str,
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str]:
    text = (response or "").strip()
    if not text:
        return False, "empty response"

    first_line = text.splitlines()[0].upper()
    if not first_line.startswith("RISK LEVEL:") or risk_level not in first_line:
        return False, "missing or incorrect first-line risk label"

    text_upper = text.upper()
    missing_sections = [s for s in REQUIRED_RESPONSE_SECTIONS if s not in text_upper]
    if missing_sections:
        return False, f"missing sections: {', '.join(missing_sections)}"

    if len(text) < min_chars:
        return False, f"too short ({len(text)} chars), minimum is {min_chars}"
    if len(text) > max_chars:
        return False, f"too long ({len(text)} chars), maximum is {max_chars}"

    bullet_count = sum(1 for line in text.splitlines() if line.strip().startswith("- "))
    if bullet_count < 3:
        return False, "insufficient bullet-point actions"

    if contains_disallowed_phrase(text):
        return False, "contains disallowed phrase"

    return True, "ok"


def export_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def average_word_count(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return float(statistics.mean(len(t.split()) for t in texts))


def risk_distribution(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {"LOW": counts.get("LOW", 0), "MID": counts.get("MID", 0), "HIGH": counts.get("HIGH", 0)}


def distribution_percent(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def validate_outputs(samples: list[SampleRecord], split_map: dict[int, str], output_dir: Path, seed: int) -> dict[str, Any]:
    print("\n=== STEP 7: QUALITY VALIDATION ===")

    bad_prefix: list[int] = []
    bad_length: list[int] = []
    disallowed_hits: list[dict[str, Any]] = []

    for s in samples:
        first_line = s.response.splitlines()[0] if s.response else ""
        if not first_line.upper().startswith("RISK LEVEL:") or s.risk_level not in first_line.upper():
            bad_prefix.append(s.row_index)

        if len(s.response) < 100 or len(s.response) > 2000:
            bad_length.append(s.row_index)

        lowered = s.response.lower()
        for phrase in DISALLOWED_PHRASES:
            if phrase in lowered:
                disallowed_hits.append({"row_index": s.row_index, "phrase": phrase})

    print(f"Prefix/risk-label violations: {len(bad_prefix)}")
    print(f"Length violations (outside 100-2000 chars): {len(bad_length)}")
    print(f"Disallowed phrase hits: {len(disallowed_hits)}")

    all_dist = risk_distribution([s.risk_level for s in samples])
    train_dist = risk_distribution([s.risk_level for s in samples if split_map[s.row_index] == "train"])
    test_dist = risk_distribution([s.risk_level for s in samples if split_map[s.row_index] == "test"])

    all_pct = distribution_percent(all_dist)
    train_pct = distribution_percent(train_dist)
    test_pct = distribution_percent(test_dist)

    mismatch_flags = []
    for risk in ["LOW", "MID", "HIGH"]:
        if abs(train_pct[risk] - all_pct[risk]) > 0.10:
            mismatch_flags.append(f"train {risk} distribution differs >10% from overall")
        if abs(test_pct[risk] - all_pct[risk]) > 0.10:
            mismatch_flags.append(f"test {risk} distribution differs >10% from overall")

    print("Class distribution overall:", all_dist)
    print("Class distribution train:", train_dist)
    print("Class distribution test:", test_dist)

    if mismatch_flags:
        print("[WARN] Distribution mismatch flags:")
        for item in mismatch_flags:
            print(f"- {item}")
    else:
        print("[OK] Train/test class distributions roughly match overall distribution")

    rng = random.Random(seed + 999)
    sample_examples_by_risk: dict[str, list[SampleRecord]] = {"LOW": [], "MID": [], "HIGH": []}
    for risk in ["LOW", "MID", "HIGH"]:
        risk_samples = [s for s in samples if s.risk_level == risk]
        take_n = min(3, len(risk_samples))
        if take_n > 0:
            sample_examples_by_risk[risk] = rng.sample(risk_samples, take_n)

    examples_txt_path = output_dir / "sample_examples.txt"
    with examples_txt_path.open("w", encoding="utf-8") as f:
        for risk in ["LOW", "MID", "HIGH"]:
            f.write(f"=== {risk} RISK EXAMPLES ===\n\n")
            for i, ex in enumerate(sample_examples_by_risk[risk], start=1):
                f.write(f"Example {i} | row_index={ex.row_index}\n")
                f.write("--- PROMPT ---\n")
                f.write(ex.instruction + "\n")
                f.write("--- RESPONSE ---\n")
                f.write(ex.response + "\n\n")
            f.write("\n")

    print(f"Saved manual review examples to: {examples_txt_path}")

    return {
        "bad_prefix_count": len(bad_prefix),
        "bad_prefix_rows": bad_prefix[:100],
        "bad_length_count": len(bad_length),
        "bad_length_rows": bad_length[:100],
        "disallowed_phrase_hits": disallowed_hits[:200],
        "distribution_overall": all_dist,
        "distribution_train": train_dist,
        "distribution_test": test_dist,
        "distribution_flags": mismatch_flags,
        "sample_examples_file": str(examples_txt_path),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    random.seed(args.seed)

    required_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

    df = pd.read_csv(input_path)
    df = df.reset_index(drop=True)
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    validate_and_describe_dataframe(df)

    checkpoint = load_checkpoint(checkpoint_path)
    processed_map: dict[str, Any] = checkpoint.get("processed", {})
    print(f"\nLoaded checkpoint entries: {len(processed_map)}")

    gemini = None
    if args.use_gemini:
        gemini = GeminiEnricher(
            model_name=args.gemini_model,
            thinking_level=args.thinking_level,
            temperature=0.8,
            top_p=0.95,
            max_output_tokens=8192,
        )
        if not gemini.enabled:
            print("[WARN] Gemini requested but unavailable. Using fallback templates for all rows.")
    else:
        print("[INFO] Running in template mode (Gemini disabled).")

    samples: list[SampleRecord] = []

    fallback_count = 0
    gemini_count = 0

    print("\n=== STEP 2-5: CONTEXT + PROMPTS + ENRICHMENT + CHAT FORMATTING ===")

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        ck = processed_map.get(str(row_index))
        if ck is not None:
            if args.use_gemini:
                cached_response = str(ck.get("response", ""))
                cached_risk = str(ck.get("risk_level", ""))
                ok_cached, _ = evaluate_response_quality(
                    response=cached_response,
                    risk_level=cached_risk,
                    min_chars=args.min_response_chars,
                    max_chars=args.max_response_chars,
                )
                if not ok_cached:
                    processed_map.pop(str(row_index), None)
                else:
                    samples.append(
                        SampleRecord(
                            row_index=row_index,
                            risk_level=str(ck["risk_level"]),
                            instruction=str(ck["instruction"]),
                            response=str(ck["response"]),
                            text=str(ck["text"]),
                        )
                    )
                    continue
            else:
                # Even in template mode, validate cached entries against quality gates
                cached_response = str(ck.get("response", ""))
                cached_risk = str(ck.get("risk_level", ""))
                ok_cached, _ = evaluate_response_quality(
                    response=cached_response,
                    risk_level=cached_risk,
                    min_chars=args.min_response_chars,
                    max_chars=args.max_response_chars,
                )
                if ok_cached:
                    samples.append(
                        SampleRecord(
                            row_index=row_index,
                            risk_level=str(ck["risk_level"]),
                            instruction=str(ck["instruction"]),
                            response=str(ck["response"]),
                            text=str(ck["text"]),
                        )
                    )
                    continue
                else:
                    # Discard stale/truncated checkpoint entry; regenerate below
                    processed_map.pop(str(row_index), None)

        risk_level = normalize_risk_label(str(row["RiskLevel"]))
        context = generate_synthetic_context(age=int(row["Age"]), row_index=row_index, seed=args.seed)

        instruction = build_instruction_prompt(row=row, context=context, row_index=row_index, seed=args.seed)

        response_text: str | None = None
        if gemini is not None and gemini.enabled:
            base_prompt = build_gemini_prompt(row=row, context=context, risk_level=risk_level)
            gem_prompt = base_prompt
            for semantic_attempt in range(3):
                candidate = gemini.generate(gem_prompt, retries=3, sleep_between_calls=0.5)
                if candidate:
                    candidate = ensure_response_quality(candidate, risk_level)
                    ok_response, reason = evaluate_response_quality(
                        response=candidate,
                        risk_level=risk_level,
                        min_chars=args.min_response_chars,
                        max_chars=args.max_response_chars,
                    )
                    if ok_response:
                        response_text = candidate
                        gemini_count += 1
                        break

                    if semantic_attempt < 2:
                        gem_prompt = (
                            base_prompt
                            + "\n\nYour previous draft was invalid: "
                            + reason
                            + ". Rewrite fully and ensure all required sections are complete. "
                            + f"Return between {args.min_response_chars} and {args.max_response_chars} characters."
                        )

        if not response_text:
            response_text = build_fallback_response(
                row=row,
                context=context,
                risk_level=risk_level,
                row_index=row_index,
                seed=args.seed,
            )
            fallback_count += 1

        response_text = ensure_response_quality(response_text, risk_level)
        chat_text = format_gemma_chat(instruction, response_text)

        rec = SampleRecord(
            row_index=row_index,
            risk_level=risk_level,
            instruction=instruction,
            response=response_text,
            text=chat_text,
        )
        samples.append(rec)

        processed_map[str(row_index)] = {
            "risk_level": risk_level,
            "instruction": instruction,
            "response": response_text,
            "text": chat_text,
        }

        if (row_index + 1) % 50 == 0:
            print(
                f"[PROGRESS] processed={row_index + 1}/{len(df)} | "
                f"gemini={gemini_count} | fallback={fallback_count}"
            )

        if (row_index + 1) % 100 == 0:
            checkpoint["processed"] = processed_map
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"[CHECKPOINT] Saved intermediate checkpoint at row {row_index + 1}")

    checkpoint["processed"] = processed_map
    save_checkpoint(checkpoint_path, checkpoint)
    print(f"[CHECKPOINT] Final checkpoint saved: {checkpoint_path}")

    # Keep ordering stable by original row index.
    samples.sort(key=lambda s: s.row_index)

    print("\n=== STEP 6: DATASET BUILD + EXPORT ===")
    all_texts = [s.text for s in samples]
    ds = Dataset.from_dict({"text": all_texts})
    split_ds = ds.train_test_split(test_size=0.10, seed=args.seed)
    ds_dict = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})

    hf_out_dir = output_dir / "maternal_health_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(hf_out_dir))

    train_indices = set(split_ds["train"].indices) if hasattr(split_ds["train"], "indices") else set()

    # Build split mapping from row order; when indices aren't directly exposed, derive from texts.
    split_map: dict[int, str] = {}
    if train_indices:
        for i, s in enumerate(samples):
            split_map[s.row_index] = "train" if i in train_indices else "test"
    else:
        train_texts = set(split_ds["train"]["text"])
        for s in samples:
            split_map[s.row_index] = "train" if s.text in train_texts else "test"

    train_rows = [{"text": s.text, "risk_level": s.risk_level, "row_index": s.row_index} for s in samples if split_map[s.row_index] == "train"]
    test_rows = [{"text": s.text, "risk_level": s.risk_level, "row_index": s.row_index} for s in samples if split_map[s.row_index] == "test"]

    export_jsonl(output_dir / "maternal_health_train.jsonl", train_rows)
    export_jsonl(output_dir / "maternal_health_eval.jsonl", test_rows)

    all_counts = risk_distribution([s.risk_level for s in samples])
    train_counts = risk_distribution([row["risk_level"] for row in train_rows])
    eval_counts = risk_distribution([row["risk_level"] for row in test_rows])

    avg_len_all = average_word_count([s.text for s in samples])
    avg_len_train = average_word_count([r["text"] for r in train_rows])
    avg_len_eval = average_word_count([r["text"] for r in test_rows])

    print(f"Total samples: {len(samples)}")
    print(f"Train size: {len(train_rows)}")
    print(f"Eval size: {len(test_rows)}")
    print(f"Average length (words): all={avg_len_all:.2f}, train={avg_len_train:.2f}, eval={avg_len_eval:.2f}")
    print("Class distribution (all):", all_counts)
    print("Class distribution (train):", train_counts)
    print("Class distribution (eval):", eval_counts)

    validation_report = validate_outputs(samples=samples, split_map=split_map, output_dir=output_dir, seed=args.seed)

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_samples": len(samples),
        "train_samples": len(train_rows),
        "eval_samples": len(test_rows),
        "avg_word_length": {
            "all": avg_len_all,
            "train": avg_len_train,
            "eval": avg_len_eval,
        },
        "class_distribution": {
            "all": all_counts,
            "train": train_counts,
            "eval": eval_counts,
        },
        "generation": {
            "mode": "gemini" if args.use_gemini else "template",
            "gemini_success_count": gemini_count,
            "fallback_count": fallback_count,
            "model": args.gemini_model if args.use_gemini else None,
            "thinking_level": args.thinking_level if args.use_gemini else None,
        },
        "validation": validation_report,
        "artifacts": {
            "hf_dataset_dir": str(hf_out_dir),
            "train_jsonl": str(output_dir / "maternal_health_train.jsonl"),
            "eval_jsonl": str(output_dir / "maternal_health_eval.jsonl"),
            "summary_json": str(output_dir / "data_summary.json"),
            "sample_examples": str(output_dir / "sample_examples.txt"),
            "checkpoint": str(checkpoint_path),
        },
    }

    summary_path = output_dir / "data_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"Saved HuggingFace dataset to: {hf_out_dir}")
    print(f"Saved train JSONL to: {output_dir / 'maternal_health_train.jsonl'}")
    print(f"Saved eval JSONL to: {output_dir / 'maternal_health_eval.jsonl'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
