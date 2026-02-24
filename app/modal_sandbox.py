"""Minimal Modal sandbox for MamaGuard orchestration prototyping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import modal
from fastapi import FastAPI
from pydantic import BaseModel, Field


APP_NAME = "mamaguard-sandbox-dev"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.6",
    "pydantic==2.10.5",
)


class SandboxRequest(BaseModel):
    """Simple payload to emulate multi-signal triage logic."""

    gestational_weeks: int | None = Field(default=None, ge=1, le=45)
    systolic_bp: int | None = Field(default=None, ge=50, le=260)
    diastolic_bp: int | None = Field(default=None, ge=30, le=160)
    headache: bool = False
    vision_changes: bool = False
    decreased_fetal_movement: bool = False
    fever_c: float | None = Field(default=None, ge=30.0, le=45.0)
    wound_urgency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    infection_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    free_text: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def triage_logic(payload: SandboxRequest) -> dict[str, Any]:
    risk = "green"
    reasons: list[str] = []
    actions: list[str] = []

    if payload.systolic_bp is not None and payload.diastolic_bp is not None:
        if payload.systolic_bp >= 160 or payload.diastolic_bp >= 110:
            risk = "red"
            reasons.append("Severely elevated blood pressure.")
            actions.append("Call emergency services now.")
        elif payload.systolic_bp >= 140 or payload.diastolic_bp >= 90:
            risk = "yellow"
            reasons.append("Elevated blood pressure for pregnancy context.")
            actions.append("Contact OB provider today.")

    if payload.headache and payload.vision_changes:
        if risk != "red":
            risk = "red"
        reasons.append("Headache with vision changes can indicate severe risk.")
        actions.append("Seek urgent same-day clinical evaluation.")

    if payload.decreased_fetal_movement:
        if risk != "red":
            risk = "red"
        reasons.append("Reported decreased fetal movement.")
        actions.append("Contact labor triage immediately.")

    if payload.fever_c is not None and payload.fever_c >= 38.0:
        if risk == "green":
            risk = "yellow"
        reasons.append("Fever may indicate infection.")
        actions.append("Monitor temperature and contact provider.")

    if payload.wound_urgency_score is not None and payload.wound_urgency_score >= 0.6:
        if risk == "green":
            risk = "yellow"
        reasons.append("Wound urgency score is elevated.")
        actions.append("Send wound photo to care team today.")

    if payload.infection_risk_score is not None and payload.infection_risk_score >= 0.7:
        if risk == "green":
            risk = "yellow"
        reasons.append("Infection risk score is elevated.")
        actions.append("Watch for spreading redness, warmth, and drainage.")

    if not reasons:
        reasons.append("No acute flags detected from submitted signals.")
    if not actions:
        actions.append("Continue routine monitoring and daily check-ins.")

    return {
        "risk_level": risk,
        "reasons": reasons,
        "actions": actions,
    }


@app.function(image=image, cpu=1, timeout=60)
def ping() -> dict[str, str]:
    return {"status": "ok", "service": APP_NAME, "timestamp": utc_now()}


@app.function(image=image, cpu=1, timeout=60)
@modal.asgi_app()
def api():
    web = FastAPI(title="MamaGuard Modal Sandbox", version="0.1.0")

    @web.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": APP_NAME, "timestamp": utc_now()}

    @web.post("/triage/sandbox")
    async def triage(payload: SandboxRequest) -> dict[str, Any]:
        result = triage_logic(payload)
        return {
            "timestamp": utc_now(),
            "input": payload.model_dump(),
            "triage": result,
        }

    return web
