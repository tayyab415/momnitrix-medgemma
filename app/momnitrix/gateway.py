"""Gateway for calling Modal model endpoints with local fallbacks."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

from momnitrix.config import Settings
from momnitrix.risk import heuristic_medgemma_decision
from momnitrix.schemas import MedGemmaRiskResponse, TriageStreamRequest


class ModelGateway:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def _post_json(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(
            timeout=timeout_sec or self._settings.request_timeout_sec,
            follow_redirects=True,
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _hash_unit_interval(content: str, salt: str) -> float:
        digest = hashlib.sha1(f"{salt}:{content}".encode("utf-8")).digest()
        return round(int.from_bytes(digest[:2], "big") / 65535.0, 4)

    async def medsiglip_infer(self, image_b64: str) -> dict[str, float]:
        if self._settings.core_gpu_base_url:
            payload = await self._post_json(
                self._settings.core_gpu_base_url,
                "/internal/medsiglip/infer",
                {"image_b64": image_b64},
            )
            return {k: float(v) for k, v in payload["label_scores"].items()}

        return {
            "healing_status": self._hash_unit_interval(image_b64, "heal"),
            "erythema": self._hash_unit_interval(image_b64, "ery"),
            "edema": self._hash_unit_interval(image_b64, "ede"),
            "infection_risk": self._hash_unit_interval(image_b64, "inf"),
            "urgency": self._hash_unit_interval(image_b64, "urg"),
            "exudate": self._hash_unit_interval(image_b64, "exu"),
        }

    async def derm_infer(self, image_b64: str) -> tuple[dict[str, float], list[dict[str, float | str]]]:
        if self._settings.derm_base_url:
            payload = await self._post_json(
                self._settings.derm_base_url,
                "/internal/derm/infer",
                {"image_b64": image_b64},
            )
            scores = {k: float(v) for k, v in payload["condition_scores"].items()}
            return scores, payload.get("top3", [])

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
        scores = {label: self._hash_unit_interval(image_b64, label) for label in labels}
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_payload = [{"condition": k, "score": float(v)} for k, v in top3]
        return scores, top3_payload

    async def medasr_transcribe(self, audio_b64: str) -> str:
        if self._settings.medasr_base_url:
            payload = await self._post_json(
                self._settings.medasr_base_url,
                "/internal/medasr/transcribe",
                {"audio_b64": audio_b64},
            )
            return str(payload.get("transcript", "")).strip()

        checksum = hashlib.sha1(audio_b64.encode("utf-8")).hexdigest()[:8]
        return f"Simulated voice check-in transcript ({checksum})."

    async def medgemma_decide(
        self,
        request: TriageStreamRequest,
        specialist_outputs: dict[str, Any],
    ) -> MedGemmaRiskResponse:
        decision, _meta = await self.medgemma_decide_with_meta(request, specialist_outputs)
        return decision

    async def medgemma_decide_with_meta(
        self,
        request: TriageStreamRequest,
        specialist_outputs: dict[str, Any],
    ) -> tuple[MedGemmaRiskResponse, dict[str, Any]]:
        if self._settings.core_gpu_base_url:
            try:
                payload = await self._post_json(
                    self._settings.core_gpu_base_url,
                    "/internal/medgemma/risk_decide",
                    {
                        "patient_context": request.patient_context.model_dump(),
                        "vitals": request.vitals.model_dump(),
                        "inputs": request.inputs.model_dump(),
                        "specialist_outputs": specialist_outputs,
                        "metadata": request.metadata or {},
                    },
                    timeout_sec=self._settings.medgemma_request_timeout_sec,
                )
                decision = MedGemmaRiskResponse.model_validate(payload)
                return (
                    decision,
                    {
                        "engine": "medgemma.core_gpu",
                        "fallback_used": False,
                        "upstream": self._settings.core_gpu_base_url,
                        "runtime_diagnostics": dict(decision.runtime_diagnostics or {}),
                    },
                )
            except Exception as exc:
                # Controlled fallback keeps the pipeline available during demo.
                print(f"[momnitrix] gateway_medgemma_fallback: {type(exc).__name__}: {exc}")
                fallback = heuristic_medgemma_decision(request, specialist_outputs)
                return (
                    fallback,
                    {
                        "engine": "heuristic.gateway_fallback",
                        "fallback_used": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "upstream": self._settings.core_gpu_base_url,
                        "runtime_diagnostics": dict(fallback.runtime_diagnostics or {}),
                    },
                )

        fallback = heuristic_medgemma_decision(request, specialist_outputs)
        return (
            fallback,
            {
                "engine": "heuristic.local",
                "fallback_used": True,
                "upstream": None,
                "runtime_diagnostics": dict(fallback.runtime_diagnostics or {}),
            },
        )
