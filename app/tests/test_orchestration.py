import asyncio
from pathlib import Path

from momnitrix.config import Settings
from momnitrix.gemini import GeminiOrchestrator
from momnitrix.orchestration import TriageOrchestrator
from momnitrix.schemas import Inputs, MedGemmaRiskResponse, TriageStreamRequest, Vitals
from momnitrix.storage import ArtifactStore


class AlwaysGreenGateway:
    async def medsiglip_infer(self, image_b64: str):
        return {
            "healing_status": 0.2,
            "erythema": 0.1,
            "edema": 0.05,
            "infection_risk": 0.1,
            "urgency": 0.1,
            "exudate": 0.05,
        }

    async def derm_infer(self, image_b64: str):
        return {"eczema": 0.3, "drug_rash": 0.2}, [{"condition": "eczema", "score": 0.3}]

    async def medasr_transcribe(self, audio_b64: str):
        return "I have persistent headache and blurry vision"

    async def medgemma_decide(self, request, specialist_outputs):
        return MedGemmaRiskResponse(
            risk_level="green",
            reasons=["No immediate concerns in model output."],
            action_items=["Continue monitoring."],
            clinical_summary="Stable.",
        )

    async def medgemma_decide_with_meta(self, request, specialist_outputs):
        return await self.medgemma_decide(request, specialist_outputs), {
            "engine": "stub.medgemma",
            "fallback_used": False,
            "runtime_diagnostics": {},
        }


def _settings(tmp_path: Path, *, composer_mode: str = "gemini_full") -> Settings:
    return Settings(
        local_storage_dir=str(tmp_path),
        core_gpu_base_url=None,
        derm_base_url=None,
        medasr_base_url=None,
        s3_bucket=None,
        gemini_api_key=None,
        use_real_models=False,
        response_composer_mode=composer_mode,
    )


def test_orchestrator_escalates_to_policy_floor(tmp_path: Path):
    settings = _settings(tmp_path)
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=AlwaysGreenGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
    )

    request = TriageStreamRequest(
        request_id="req-policy-floor",
        patient_context={"age_years": 29},
        vitals=Vitals(systolic_bp=168, diastolic_bp=112, fasting_glucose=5.0, temp_c=36.8, hr=82),
        inputs=Inputs(headache=True, vision_changes=True),
    )

    events: list[str] = []

    async def emit(event_name, payload):
        events.append(event_name)

    final = asyncio.run(orchestrator.run(request, emit))

    assert final.risk_level == "red"
    assert final.policy_floor == "red"
    assert "triage.final" in events


def test_orchestrator_full_multimodal_path(tmp_path: Path):
    settings = _settings(tmp_path)
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=AlwaysGreenGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
    )

    request = TriageStreamRequest(
        request_id="req-multimodal",
        patient_context={"age_years": 30},
        vitals=Vitals(systolic_bp=142, diastolic_bp=91, temp_c=37.9, fasting_glucose=5.4, hr=92),
        inputs=Inputs(
            wound_image_b64="ZmFrZS13b3VuZA==",
            skin_image_b64="ZmFrZS1za2lu",
            audio_b64="ZmFrZS1hdWRpbw==",
            free_text="I feel worried today",
        ),
    )

    emitted: list[str] = []

    async def emit(event_name, payload):
        emitted.append(event_name)

    final = asyncio.run(orchestrator.run(request, emit))

    assert final.request_id == "req-multimodal"
    assert "wound_scores" in final.specialist_outputs
    assert "skin_scores" in final.specialist_outputs
    assert "transcript" in final.specialist_outputs
    assert "router.decision" in emitted
    assert "router.prompt_plan" in emitted
    assert "gemini.delta" in emitted
    assert "diagnostics.inference_breakdown" in emitted
    assert store.read_final_response("req-multimodal") is not None


def test_orchestrator_prioritizes_glucose_guardrails(tmp_path: Path):
    settings = _settings(tmp_path)
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=AlwaysGreenGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
    )

    request = TriageStreamRequest(
        request_id="req-glucose-guardrail",
        patient_context={"gestational_weeks": 19, "age_years": 25},
        vitals={"systolic_bp": 120, "diastolic_bp": 80, "fasting_glucose": 7.7, "temp_c": 36.7, "hr": 66},
        inputs={},
    )

    async def emit(event_name, payload):
        _ = (event_name, payload)

    final = asyncio.run(orchestrator.run(request, emit))

    assert final.risk_level == "yellow"
    assert final.policy_floor == "yellow"
    assert final.medgemma_reasons
    assert any("glucose" in reason.lower() for reason in final.medgemma_reasons[:3])
    assert any("ob" in x.lower() or "glucose" in x.lower() for x in final.action_items)
    timing = final.inference_diagnostics.get("medgemma_timing_breakdown", {})
    assert "gpu_warmup_ms" in timing
    assert "medgemma_inference_ms" in timing


def test_orchestrator_medgemma_first_mode_skips_gemini(tmp_path: Path):
    settings = _settings(tmp_path, composer_mode="medgemma_first")
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=AlwaysGreenGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
        composer_mode=settings.response_composer_mode,
    )

    request = TriageStreamRequest(
        request_id="req-medgemma-first",
        patient_context={"gestational_weeks": 34, "age_years": 31},
        vitals={"systolic_bp": 120, "diastolic_bp": 80, "fasting_glucose": 5.0, "temp_c": 36.8, "hr": 78},
        inputs={},
        metadata={"composer_mode": "medgemma_first"},
    )

    emitted: list[str] = []

    async def emit(event_name, payload):
        _ = payload
        emitted.append(event_name)

    final = asyncio.run(orchestrator.run(request, emit))

    assert "gemini.skipped" in emitted
    assert "router.decision" in emitted
    assert "router.prompt_plan" in emitted
    assert "gemini.delta" not in emitted
    assert "medgemma.delta" in emitted
    assert "diagnostics.inference_breakdown" in emitted
    assert final.latency_ms.get("gemini") == 0
    assert "Current risk level is" in final.patient_message
    assert final.inference_diagnostics.get("field_authorship", {}).get("patient_message") == "medgemma"
    assert "medgemma_timing_breakdown" in final.inference_diagnostics


def test_orchestrator_medgemma_first_prefers_raw_response_text(tmp_path: Path):
    class RawGateway(AlwaysGreenGateway):
        async def medgemma_decide(self, request, specialist_outputs):
            _ = (request, specialist_outputs)
            return MedGemmaRiskResponse(
                risk_level="yellow",
                reasons=["Fasting glucose above pregnancy target."],
                action_items=["Contact OB within 24 hours."],
                clinical_summary="Elevated fasting glucose at 31 weeks.",
                raw_response_text=(
                    "RISK LEVEL: MID\n\n"
                    "CLINICAL REASONING: Fasting glucose is above pregnancy target.\n\n"
                    "POTENTIAL COMPLICATIONS: Gestational diabetes risk.\n\n"
                    "RECOMMENDED ACTIONS:\n- Contact OB.\n\n"
                    "WARNING SIGNS:\n- Severe headache."
                ),
            )

    settings = _settings(tmp_path, composer_mode="medgemma_first")
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=RawGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
        composer_mode=settings.response_composer_mode,
    )
    request = TriageStreamRequest(
        request_id="req-medgemma-raw",
        patient_context={"gestational_weeks": 31, "age_years": 27},
        vitals={"systolic_bp": 132, "diastolic_bp": 84, "fasting_glucose": 7.7, "temp_c": 36.9, "hr": 88},
        inputs={},
        metadata={"composer_mode": "medgemma_first"},
    )

    async def emit(event_name, payload):
        _ = (event_name, payload)

    final = asyncio.run(orchestrator.run(request, emit))
    assert final.patient_message.startswith("RISK LEVEL: MID")


def test_orchestrator_sanitizes_malformed_medgemma_items(tmp_path: Path):
    class NoisyGateway(AlwaysGreenGateway):
        async def medgemma_decide(self, request, specialist_outputs):
            _ = (request, specialist_outputs)
            return MedGemmaRiskResponse(
                risk_level="green",
                reasons=[
                    "No acute flags detected.",
                    "\"risk_level\": \"LOW\",",
                    "Stable vitals in latest reading.",
                ],
                action_items=[
                    "Continue routine monitoring.",
                    "Urgent warning sign: Call provider for severe headache.```json",
                    "\"risk_level\": \"LOW\",",
                    "{",
                    "}",
                ],
                clinical_summary="Stable trends and no immediate complications.",
            )

    settings = _settings(tmp_path)
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=NoisyGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
    )

    request = TriageStreamRequest(
        request_id="req-sanitize-items",
        patient_context={"age_years": 28},
        vitals={"systolic_bp": 118, "diastolic_bp": 76, "fasting_glucose": 4.8, "temp_c": 36.7, "hr": 74},
        inputs={},
    )

    async def emit(event_name, payload):
        _ = (event_name, payload)

    final = asyncio.run(orchestrator.run(request, emit))
    combined = " ".join(final.action_items + final.medgemma_reasons).lower()
    assert "```" not in combined
    assert "\"risk_level\"" not in combined


def test_orchestrator_rejects_missing_required_inputs(tmp_path: Path):
    settings = _settings(tmp_path)
    store = ArtifactStore(settings)
    orchestrator = TriageOrchestrator(
        gateway=AlwaysGreenGateway(),
        gemini=GeminiOrchestrator(settings),
        store=store,
    )

    request = TriageStreamRequest(
        request_id="req-missing-required",
        patient_context={"gestational_weeks": 20},
        vitals={"systolic_bp": 120, "diastolic_bp": 80},
        inputs={},
    )

    emitted: list[str] = []

    async def emit(event_name, payload):
        _ = payload
        emitted.append(event_name)

    try:
        asyncio.run(orchestrator.run(request, emit))
        assert False, "Expected ValueError for missing required inputs"
    except ValueError as exc:
        assert "missing required clinical inputs" in str(exc).lower()
    assert "request.rejected" in emitted
