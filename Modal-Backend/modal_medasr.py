"""Modal service for Momnitrix MedASR transcription."""

from __future__ import annotations

import re
from typing import Any

import modal
from fastapi import FastAPI

from momnitrix.config import get_settings
from momnitrix.model_runtime import MedasrRuntime
from momnitrix.schemas import MedASRTranscribeRequest, MedASRTranscribeResponse
from momnitrix.utils import utc_now


APP_NAME = "momnitrix-medasr"
RUNTIME_BUILD = "medasr-clean-v4"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install(
    "fastapi==0.115.6",
    "pydantic==2.10.5",
    "torch==2.5.1",
    "git+https://github.com/huggingface/transformers.git",
    "librosa==0.10.2.post1",
    "soundfile==0.12.1",
    "numpy==1.26.4",
).add_local_python_source("momnitrix")


@app.function(
    image=image,
    cpu=4,
    timeout=300,
    min_containers=0,
    max_containers=4,
    secrets=[
        modal.Secret.from_name("medgemma-hf"),
        modal.Secret.from_name("momnitrix-config"),
    ],
)
@modal.asgi_app()
def api():
    web = FastAPI(title="Momnitrix MedASR", version="1.0.0")
    runtime = MedasrRuntime(get_settings())

    @web.get("/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": APP_NAME,
            "runtime_build": RUNTIME_BUILD,
            "timestamp": utc_now().isoformat(),
            "real_models": settings.use_real_models,
        }

    @web.post("/internal/medasr/transcribe", response_model=MedASRTranscribeResponse)
    async def medasr_transcribe(payload: MedASRTranscribeRequest) -> MedASRTranscribeResponse:
        text = runtime.transcribe(payload.audio_b64)
        if text:
            text = re.sub(r"</?s>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<epsilon>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
        return MedASRTranscribeResponse(transcript=text)

    return web
