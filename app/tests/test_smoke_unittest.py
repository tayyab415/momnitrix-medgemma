import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from momnitrix.config import Settings
from momnitrix.gemini import GeminiOrchestrator
from momnitrix.orchestration import TriageOrchestrator
from momnitrix.risk import compute_policy_floor
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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        local_storage_dir=str(tmp_path),
        core_gpu_base_url=None,
        derm_base_url=None,
        medasr_base_url=None,
        s3_bucket=None,
        gemini_api_key=None,
        use_real_models=False,
    )


class MomnitrixSmokeTests(unittest.TestCase):
    def test_policy_floor_red(self):
        request = TriageStreamRequest(vitals=Vitals(systolic_bp=162, diastolic_bp=111), inputs=Inputs())
        floor, reasons = compute_policy_floor(request, specialist_outputs={})

        self.assertEqual(floor, "red")
        self.assertTrue(reasons)

    def test_policy_floor_red_for_glucose(self):
        request = TriageStreamRequest(
            patient_context={"gestational_weeks": 31},
            vitals={"fasting_glucose": 15.0},
            inputs={},
        )
        floor, reasons = compute_policy_floor(request, specialist_outputs={})

        self.assertEqual(floor, "red")
        self.assertTrue(any("glucose" in reason.lower() for reason in reasons))

    def test_orchestration_runs_and_persists(self):
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))
            store = ArtifactStore(settings)
            orchestrator = TriageOrchestrator(
                gateway=AlwaysGreenGateway(),
                gemini=GeminiOrchestrator(settings),
                store=store,
            )

            request = TriageStreamRequest(
                request_id="unittest-req",
                patient_context={"age_years": 30},
                vitals=Vitals(systolic_bp=168, diastolic_bp=112, fasting_glucose=5.4, temp_c=36.9, hr=88),
                inputs=Inputs(
                    headache=True,
                    vision_changes=True,
                    wound_image_b64="ZmFrZS13b3VuZA==",
                    skin_image_b64="ZmFrZS1za2lu",
                    audio_b64="ZmFrZS1hdWRpbw==",
                ),
            )

            emitted: list[str] = []

            async def emit(event_name, payload):
                emitted.append(event_name)

            final = asyncio.run(orchestrator.run(request, emit))

            self.assertEqual(final.risk_level, "red")
            self.assertIn("triage.final", emitted)
            self.assertIn("gemini.delta", emitted)
            self.assertIsNotNone(store.read_final_response("unittest-req"))


if __name__ == "__main__":
    unittest.main()
