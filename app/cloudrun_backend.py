"""Cloud Run entrypoint for Momnitrix orchestration API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from momnitrix.config import get_settings
from momnitrix.gemini import GeminiOrchestrator
from momnitrix.gateway import ModelGateway
from momnitrix.orchestration import TriageOrchestrator
from momnitrix.schemas import TriageStreamRequest
from momnitrix.sse import format_sse
from momnitrix.storage import ArtifactStore


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


settings = get_settings()
store = ArtifactStore(settings)
orchestrator = TriageOrchestrator(
    gateway=ModelGateway(settings),
    gemini=GeminiOrchestrator(settings),
    store=store,
    composer_mode=settings.response_composer_mode,
)

app = FastAPI(title="Momnitrix API (Cloud Run)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(probe: bool = False) -> dict[str, Any]:
    service_urls = {
        "core_gpu": settings.core_gpu_base_url,
        "derm": settings.derm_base_url,
        "medasr": settings.medasr_base_url,
    }
    service_status: dict[str, dict[str, Any]] = {
        name: {
            "configured": bool(url),
            "reachable": None,
            "status_code": None,
            "error": None,
            "url": url,
        }
        for name, url in service_urls.items()
    }

    if probe:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            async def _probe_one(name: str, url: str) -> None:
                try:
                    response = await client.get(f"{url.rstrip('/')}/health")
                    service_status[name]["reachable"] = response.is_success
                    service_status[name]["status_code"] = response.status_code
                    if not response.is_success:
                        service_status[name]["error"] = response.text[:180]
                except Exception as exc:
                    service_status[name]["reachable"] = False
                    service_status[name]["error"] = str(exc)

            await asyncio.gather(
                *[
                    _probe_one(name, url)
                    for name, url in service_urls.items()
                    if url
                ]
            )

    return {
        "status": "ok",
        "service": "momnitrix-api-cloudrun",
        "timestamp": utc_now().isoformat(),
        "gemini_model": settings.gemini_model,
        "response_composer_mode": settings.response_composer_mode,
        "gemini_key_configured": bool(settings.gemini_api_key),
        "core_gpu_configured": bool(settings.core_gpu_base_url),
        "derm_configured": bool(settings.derm_base_url),
        "medasr_configured": bool(settings.medasr_base_url),
        "core_gpu_reachable": service_status["core_gpu"]["reachable"],
        "derm_reachable": service_status["derm"]["reachable"],
        "medasr_reachable": service_status["medasr"]["reachable"],
        "probe_performed": probe,
        "services": service_status,
        "s3_configured": bool(settings.s3_bucket),
    }


@app.post("/v1/triage/stream")
async def triage_stream(payload: dict[str, Any] = Body(...)):
    try:
        request = TriageStreamRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    missing_required: list[str] = []
    if request.patient_context.age_years is None:
        missing_required.append("patient_context.age_years")
    if request.vitals.systolic_bp is None:
        missing_required.append("vitals.systolic_bp")
    if request.vitals.diastolic_bp is None:
        missing_required.append("vitals.diastolic_bp")
    if request.vitals.fasting_glucose_mmol_l is None:
        missing_required.append("vitals.fasting_glucose_mmol_l")
    if request.vitals.temp_c is None:
        missing_required.append("vitals.temp_c")
    if request.vitals.hr is None:
        missing_required.append("vitals.hr")
    if missing_required:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_required_inputs",
                "message": (
                    "Required inputs: age, systolic BP, diastolic BP, "
                    "fasting glucose, body temperature, and heart rate."
                ),
                "missing_fields": missing_required,
            },
        )

    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    done = asyncio.Event()

    async def emit(event_name: str, event_payload: dict[str, Any]) -> None:
        envelope = {
            "event": event_name,
            "timestamp": utc_now().isoformat(),
            **event_payload,
        }
        await store.append_event(event_name, envelope)
        await queue.put((event_name, envelope))

    async def runner() -> None:
        try:
            await orchestrator.run(request, emit)
        except Exception as exc:
            err = {"error": str(exc), "request_id": request.request_id}
            await emit("triage.error", err)
        finally:
            done.set()

    asyncio.create_task(runner())

    async def event_gen():
        while True:
            if done.is_set() and queue.empty():
                break
            try:
                event_name, envelope = await asyncio.wait_for(queue.get(), timeout=0.75)
                yield format_sse(event_name, envelope)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/v1/triage/{request_id}")
async def triage_result(request_id: str):
    payload = store.read_final_response(request_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="request_id not found")
    return payload

