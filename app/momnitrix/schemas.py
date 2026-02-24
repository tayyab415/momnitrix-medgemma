"""Pydantic schemas for Momnitrix endpoints and internal contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


RiskLevel = Literal["green", "yellow", "red"]


class PatientContext(BaseModel):
    patient_id: str | None = None
    age_years: int | None = Field(default=None, ge=10, le=70)
    gestational_weeks: int | None = Field(default=None, ge=1, le=45)
    known_conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


class Vitals(BaseModel):
    systolic_bp: int | None = Field(default=None, ge=50, le=260)
    diastolic_bp: int | None = Field(default=None, ge=30, le=160)
    fasting_glucose_mmol_l: float | None = Field(
        default=None,
        ge=1.0,
        le=40.0,
        validation_alias=AliasChoices(
            "fasting_glucose_mmol_l",
            "fasting_glucose",
            "fasting_plasma_glucose",
            "fasting_glucose_mmol",
        ),
    )
    hr: int | None = Field(default=None, ge=20, le=260)
    spo2: int | None = Field(default=None, ge=40, le=100)
    temp_c: float | None = Field(default=None, ge=30.0, le=45.0)
    hrv: float | None = Field(default=None, ge=0.0, le=300.0)


class Inputs(BaseModel):
    wound_image_b64: str | None = None
    skin_image_b64: str | None = None
    audio_b64: str | None = None
    free_text: str | None = None

    headache: bool = False
    vision_changes: bool = False
    decreased_fetal_movement: bool = False


class TriageStreamRequest(BaseModel):
    request_id: str | None = None
    patient_context: PatientContext = Field(default_factory=PatientContext)
    vitals: Vitals = Field(default_factory=Vitals)
    inputs: Inputs = Field(default_factory=Inputs)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MedSigLIPInferRequest(BaseModel):
    image_b64: str


class MedSigLIPInferResponse(BaseModel):
    label_scores: dict[str, float]


class DermInferRequest(BaseModel):
    image_b64: str


class DermInferResponse(BaseModel):
    condition_scores: dict[str, float]
    top3: list[dict[str, float | str]]


class MedASRTranscribeRequest(BaseModel):
    audio_b64: str


class MedASRTranscribeResponse(BaseModel):
    transcript: str


class MedGemmaRiskRequest(BaseModel):
    patient_context: PatientContext
    vitals: Vitals
    inputs: Inputs
    specialist_outputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MedGemmaRiskResponse(BaseModel):
    risk_level: RiskLevel
    reasons: list[str]
    action_items: list[str]
    clinical_summary: str
    raw_response_text: str | None = None
    runtime_diagnostics: dict[str, Any] = Field(default_factory=dict)


class FinalTriageResponse(BaseModel):
    request_id: str
    trace_id: str
    timestamp: datetime

    risk_level: RiskLevel
    policy_floor: RiskLevel

    patient_message: str
    visit_prep_summary: str

    action_items: list[str]
    medgemma_reasons: list[str]

    specialist_outputs: dict[str, Any]
    latency_ms: dict[str, int]
    artifact_refs: dict[str, str]
    inference_diagnostics: dict[str, Any] = Field(default_factory=dict)
