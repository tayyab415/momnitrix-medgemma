"""Modal service for Momnitrix Derm Foundation inference."""

from __future__ import annotations

from typing import Any

import modal
from fastapi import FastAPI

from momnitrix.config import get_settings
from momnitrix.model_runtime import DermRuntime
from momnitrix.schemas import DermInferRequest, DermInferResponse
from momnitrix.utils import utc_now


APP_NAME = "momnitrix-derm-tf"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.6",
    "pydantic==2.10.5",
    "tensorflow==2.18.0",
    "keras==3.8.0",
    "scikit-learn==1.5.2",
    "numpy==1.26.4",
    "pillow==11.1.0",
    "huggingface_hub==0.27.1",
).add_local_python_source("momnitrix").add_local_dir("artifacts/derm", remote_path="/root/artifacts/derm")


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    min_containers=0,
    max_containers=3,
    secrets=[
        modal.Secret.from_name("medgemma"),
        modal.Secret.from_name("momnitrix-config"),
    ],
)
@modal.asgi_app()
def api():
    web = FastAPI(title="Momnitrix Derm TF", version="1.0.0")
    runtime = DermRuntime(get_settings())

    @web.get("/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": APP_NAME,
            "timestamp": utc_now().isoformat(),
            "classifier_path": settings.derm_classifier_path,
            "scaler_path": settings.derm_scaler_path,
            "labels_path": settings.derm_labels_path,
            "real_models": settings.use_real_models,
        }

    @web.post("/internal/derm/infer", response_model=DermInferResponse)
    async def derm_infer(payload: DermInferRequest) -> DermInferResponse:
        condition_scores, top3 = runtime.infer(payload.image_b64)
        return DermInferResponse(condition_scores=condition_scores, top3=top3)

    return web
