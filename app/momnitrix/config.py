"""Runtime settings for Momnitrix services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _composer_mode(value: str | None) -> str:
    if not value:
        return "gemini_full"
    token = value.strip().lower().replace("-", "_")
    mapping = {
        "gemini_full": "gemini_full",
        "full": "gemini_full",
        "default": "gemini_full",
        "medgemma_first": "medgemma_first",
        "medgemma": "medgemma_first",
        "medgemma_only": "medgemma_first",
        "gemini_off": "medgemma_first",
        "off": "medgemma_first",
    }
    return mapping.get(token, "gemini_full")


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("MOMNITRIX_APP_NAME", "momnitrix-api"))

    # Default orchestration model (overridable via env).
    gemini_model: str = field(
        default_factory=lambda: os.getenv("MOMNITRIX_GEMINI_MODEL", "gemini-3-flash-preview")
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: _first_env(
            "GEMINI_API_KEY",
            "Gemini_API_Key",
            "GEMINI_API_Key",
            "gemini_api_key",
        )
    )
    response_composer_mode: str = field(
        default_factory=lambda: _composer_mode(os.getenv("MOMNITRIX_RESPONSE_COMPOSER_MODE"))
    )
    hf_token: str | None = field(
        default_factory=lambda: _first_env(
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HF_API_TOKEN",
        )
    )

    core_gpu_base_url: str | None = field(default_factory=lambda: os.getenv("MOMNITRIX_CORE_GPU_BASE_URL"))
    derm_base_url: str | None = field(default_factory=lambda: os.getenv("MOMNITRIX_DERM_BASE_URL"))
    medasr_base_url: str | None = field(default_factory=lambda: os.getenv("MOMNITRIX_MEDASR_BASE_URL"))

    request_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("MOMNITRIX_REQUEST_TIMEOUT_SEC", "120"))
    )
    medgemma_request_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("MOMNITRIX_MEDGEMMA_REQUEST_TIMEOUT_SEC", "300"))
    )

    # Persistence
    s3_bucket: str | None = field(default_factory=lambda: os.getenv("MOMNITRIX_S3_BUCKET"))
    s3_region: str = field(default_factory=lambda: os.getenv("MOMNITRIX_S3_REGION", "us-east-1"))
    s3_prefix: str = field(default_factory=lambda: os.getenv("MOMNITRIX_S3_PREFIX", "momnitrix"))
    local_storage_dir: str = field(
        default_factory=lambda: os.getenv("MOMNITRIX_LOCAL_STORAGE_DIR", ".momnitrix_local_store")
    )

    # Model runtime toggles.
    use_real_models: bool = field(
        default_factory=lambda: _as_bool(os.getenv("MOMNITRIX_USE_REAL_MODELS"), default=False)
    )

    # Fine-tuned HF artifacts (user-provided model IDs).
    medgemma_base_model_id: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_MEDGEMMA_BASE_MODEL_ID",
            "google/medgemma-1.5-4b-it",
        )
    )
    # Backward compatibility: MOMNITRIX_MEDGEMMA_MODEL_ID is treated as adapter id.
    medgemma_adapter_id: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_MEDGEMMA_ADAPTER_ID",
            os.getenv("MOMNITRIX_MEDGEMMA_MODEL_ID", "tyb343/mamaguard-vitals-lora-p100"),
        )
    )
    medgemma_is_adapter: bool = field(
        default_factory=lambda: _as_bool(os.getenv("MOMNITRIX_MEDGEMMA_IS_ADAPTER"), default=True)
    )

    medsiglip_model_id: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_MEDSIGLIP_MODEL_ID",
            "tyb343/medsiglip-448-surgwound-v2",
        )
    )
    derm_classifier_path: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_DERM_CLASSIFIER_PATH",
            "/root/artifacts/derm/derm_classifier.pkl",
        )
    )
    derm_scaler_path: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_DERM_SCALER_PATH",
            "/root/artifacts/derm/derm_scaler.pkl",
        )
    )
    derm_labels_path: str = field(
        default_factory=lambda: os.getenv(
            "MOMNITRIX_DERM_LABELS_PATH",
            "/root/artifacts/derm/derm_labels.json",
        )
    )


def get_settings() -> Settings:
    return Settings()
