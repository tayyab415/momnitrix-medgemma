"""Modal service for Momnitrix core GPU inference.

Hosts fine-tuned MedGemma 1.5 4B and fine-tuned MedSigLIP in one GPU container.
"""

from __future__ import annotations

import asyncio
from typing import Any

import modal
from fastapi import FastAPI

from momnitrix.config import get_settings
from momnitrix.model_runtime import CoreGpuRuntime
from momnitrix.schemas import (
    MedGemmaRiskRequest,
    MedGemmaRiskResponse,
    MedSigLIPInferRequest,
    MedSigLIPInferResponse,
)
from momnitrix.utils import utc_now


APP_NAME = "momnitrix-core-gpu"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.6",
    "pydantic==2.10.5",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers>=4.50.0,<5.0.0",
    "peft>=0.17.0,<0.18.0",
    "accelerate==1.2.1",
    "pillow==11.1.0",
).add_local_python_source("momnitrix")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=900,
    min_containers=0,
    max_containers=1,
    secrets=[
        modal.Secret.from_name("medgemma-hf"),
        modal.Secret.from_name("momnitrix-config"),
        # Optional override for MedSigLIP model without altering shared config.
        modal.Secret.from_name("medsiglip-v2"),
    ],
)
@modal.asgi_app()
def api():
    web = FastAPI(title="Momnitrix Core GPU", version="1.0.0")
    runtime = CoreGpuRuntime(get_settings())

    # Serialize MedGemma generations to avoid GPU contention spikes.
    medgemma_lock = asyncio.Semaphore(1)

    @web.get("/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": APP_NAME,
            "timestamp": utc_now().isoformat(),
            "medgemma_base_model_id": settings.medgemma_base_model_id,
            "medgemma_adapter_id": settings.medgemma_adapter_id,
            "medsiglip_model_id": settings.medsiglip_model_id,
            "real_models": settings.use_real_models,
        }

    @web.post("/internal/medsiglip/infer", response_model=MedSigLIPInferResponse)
    async def medsiglip_infer(payload: MedSigLIPInferRequest) -> MedSigLIPInferResponse:
        scores = runtime.medsiglip_infer(payload.image_b64)
        return MedSigLIPInferResponse(label_scores=scores)

    @web.post("/internal/medgemma/risk_decide", response_model=MedGemmaRiskResponse)
    async def medgemma_risk(payload: MedGemmaRiskRequest) -> MedGemmaRiskResponse:
        async with medgemma_lock:
            return runtime.medgemma_decide(payload)

    return web
