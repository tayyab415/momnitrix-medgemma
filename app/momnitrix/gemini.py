"""Gemini 3 Flash orchestration client."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from momnitrix.config import Settings
from momnitrix.schemas import MedGemmaRiskResponse, TriageStreamRequest


class GeminiOrchestrator:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def model_name(self) -> str:
        return self._settings.gemini_model

    def _build_prompt(
        self,
        request: TriageStreamRequest,
        decision: MedGemmaRiskResponse,
        specialist_outputs: dict[str, Any],
        final_risk: str,
    ) -> str:
        payload = {
            "risk_level": final_risk,
            "medgemma_decision": decision.model_dump(),
            "patient_context": request.patient_context.model_dump(),
            "vitals": request.vitals.model_dump(),
            "specialist_outputs": specialist_outputs,
            "style_rules": [
                "Use empathetic plain language for a pregnant patient.",
                "Keep to 130-180 words.",
                "Do not claim diagnosis certainty.",
                "End with clear next steps.",
            ],
        }
        return (
            "You are Momnitrix patient communication orchestrator. "
            "Create JSON with keys `patient_message` and `visit_prep_summary`.\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _clip_text(cls, text: str, *, max_chars: int) -> str:
        cleaned = cls._normalize_text(text)
        if len(cleaned) <= max_chars:
            return cleaned
        sentence_cut = max(cleaned.rfind(". ", 0, max_chars), cleaned.rfind("; ", 0, max_chars))
        if sentence_cut >= int(max_chars * 0.5):
            return cleaned[: sentence_cut + 1].rstrip()
        space_cut = cleaned.rfind(" ", 0, max_chars)
        if space_cut > 40:
            return cleaned[:space_cut].rstrip() + "..."
        return cleaned[:max_chars].rstrip() + "..."

    @classmethod
    def _compact_list(cls, items: list[str], *, limit: int, max_chars: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = cls._normalize_text(str(item))
            if not cleaned:
                continue
            key = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(cls._clip_text(cleaned, max_chars=max_chars).rstrip(" ,.;"))
            if len(out) >= limit:
                break
        return out

    async def _call_gemini(self, prompt: str) -> dict[str, str] | None:
        if not self._settings.gemini_api_key:
            print("[momnitrix] gemini_api_key_missing; using fallback response")
            return None

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        async def _call_model(model_name: str) -> dict[str, Any]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_sec) as client:
                response = await client.post(url, params={"key": self._settings.gemini_api_key}, json=body)
                response.raise_for_status()
                return response.json()

        model_name = self._settings.gemini_model
        try:
            data = await _call_model(model_name)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403, 404} and model_name != "gemini-3-flash":
                print(
                    f"[momnitrix] gemini_model_retry: {model_name} returned {status}; retrying gemini-3-flash"
                )
                data = await _call_model("gemini-3-flash")
            else:
                raise

        candidates = data.get("candidates") or []
        if not candidates:
            return None

        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
        if not text:
            return None

        parsed = self._extract_json(text)
        if parsed:
            msg = str(parsed.get("patient_message", "")).strip()
            summary = str(parsed.get("visit_prep_summary", "")).strip()
            if msg and summary:
                return {
                    "patient_message": msg,
                    "visit_prep_summary": summary,
                }

        # Accept plain-text fallback from Gemini instead of dropping to template fallback.
        plain = text.strip()
        if plain:
            summary = self._extract_visit_summary(plain)
            return {
                "patient_message": plain,
                "visit_prep_summary": summary,
            }

        return None

    async def _call_gemini_json(self, prompt: str, *, temperature: float = 0.1) -> dict[str, Any] | None:
        if not self._settings.gemini_api_key:
            return None

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }

        async def _call_model(model_name: str) -> dict[str, Any]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_sec) as client:
                response = await client.post(url, params={"key": self._settings.gemini_api_key}, json=body)
                response.raise_for_status()
                return response.json()

        model_name = self._settings.gemini_model
        try:
            data = await _call_model(model_name)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403, 404} and model_name != "gemini-3-flash":
                data = await _call_model("gemini-3-flash")
            else:
                raise

        candidates = data.get("candidates") or []
        if not candidates:
            return None

        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
        if not text:
            return None
        parsed = self._extract_json(text)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _build_task_instruction_prompt(
        self,
        request: TriageStreamRequest,
        route_context: dict[str, Any],
    ) -> str:
        payload = {
            "ui_mode": route_context.get("ui_mode"),
            "has_wound_image": bool(request.inputs.wound_image_b64),
            "has_skin_image": bool(request.inputs.skin_image_b64),
            "has_audio": bool(request.inputs.audio_b64),
            "has_text": bool((request.inputs.free_text or "").strip()),
            "patient_context": request.patient_context.model_dump(),
            "vitals": request.vitals.model_dump(),
            "safety_rules": [
                "Treat fasting glucose as fasting unless explicitly stated otherwise.",
                "Use pregnancy-aware thresholds and conservative escalation.",
                "Avoid diagnosis certainty claims; provide risk-aware guidance.",
            ],
            "output_contract": {
                "intent": "short snake_case label",
                "prompt_strategy": "one of: maternal_diagnosis, general_medical_qa, wound_focus, derm_focus, multimodal_fusion",
                "medgemma_task_instruction": "2-5 concise lines instructing MedGemma response framing",
            },
        }
        return (
            "You are Momnitrix orchestration planner. "
            "Return ONLY JSON with keys intent, prompt_strategy, medgemma_task_instruction.\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _fallback_task_instruction(
        request: TriageStreamRequest,
        route_context: dict[str, Any],
    ) -> dict[str, str]:
        has_image = bool(request.inputs.wound_image_b64 or request.inputs.skin_image_b64)
        has_audio = bool(request.inputs.audio_b64)
        has_text = bool((request.inputs.free_text or "").strip())
        ui_mode = str(route_context.get("ui_mode") or "").strip().lower()

        intent = "maternal_diagnosis"
        prompt_strategy = "maternal_diagnosis"

        if has_image and request.inputs.wound_image_b64 and not request.inputs.skin_image_b64:
            intent = "wound_assessment"
            prompt_strategy = "wound_focus"
        elif has_image and request.inputs.skin_image_b64 and not request.inputs.wound_image_b64:
            intent = "derm_assessment"
            prompt_strategy = "derm_focus"
        elif has_image:
            intent = "multimodal_image_assessment"
            prompt_strategy = "multimodal_fusion"
        elif ui_mode == "text" and has_text and not has_audio:
            intent = "maternal_question_or_triage"
            prompt_strategy = "general_medical_qa"
        elif has_audio:
            intent = "voice_symptom_triage"
            prompt_strategy = "maternal_diagnosis"

        instruction_lines = [
            "Use a maternal-risk framing tied to exact provided values and thresholds.",
            "If fasting glucose is present, treat it as fasting and cite pregnancy threshold targets.",
            "Prioritize concise actions and urgent warning signs when risk is elevated.",
        ]
        if prompt_strategy == "general_medical_qa":
            instruction_lines.insert(0, "Answer as a practical pregnancy-safe medical Q/A summary.")
        if prompt_strategy == "wound_focus":
            instruction_lines.insert(0, "Integrate wound specialist outputs and infection risk context.")
        if prompt_strategy == "derm_focus":
            instruction_lines.insert(0, "Integrate dermatology specialist outputs and pregnancy-safe differentials.")
        if prompt_strategy == "multimodal_fusion":
            instruction_lines.insert(0, "Fuse image, voice, and vitals signals; resolve conflicts conservatively.")

        return {
            "intent": intent,
            "prompt_strategy": prompt_strategy,
            "medgemma_task_instruction": "\n".join(instruction_lines),
            "planner_source": "fallback",
        }

    async def compose_task_instruction(
        self,
        request: TriageStreamRequest,
        route_context: dict[str, Any],
    ) -> dict[str, str]:
        fallback = self._fallback_task_instruction(request, route_context)
        if not self._settings.gemini_api_key:
            return fallback

        prompt = self._build_task_instruction_prompt(request, route_context)
        try:
            parsed = await self._call_gemini_json(prompt, temperature=0.1)
            if not isinstance(parsed, dict):
                return fallback
            task_instruction = str(parsed.get("medgemma_task_instruction") or "").strip()
            prompt_strategy = str(parsed.get("prompt_strategy") or "").strip()
            intent = str(parsed.get("intent") or "").strip()
            if not task_instruction or not prompt_strategy:
                return fallback
            return {
                "intent": intent or fallback["intent"],
                "prompt_strategy": prompt_strategy[:80],
                "medgemma_task_instruction": task_instruction[:1200],
                "planner_source": "gemini",
            }
        except Exception as exc:
            print(f"[momnitrix] gemini_task_instruction_fallback: {type(exc).__name__}: {exc}")
            return fallback

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _extract_visit_summary(text: str) -> str:
        marker = "visit prep summary"
        lower = text.lower()
        idx = lower.find(marker)
        if idx != -1:
            snippet = text[idx:]
            return snippet[:1000].strip()
        return "Visit Prep Summary:\n- Generated from Gemini plain-text response."

    @staticmethod
    def _fallback_response(
        request: TriageStreamRequest,
        decision: MedGemmaRiskResponse,
        specialist_outputs: dict[str, Any],
        final_risk: str,
    ) -> dict[str, str]:
        gestation = request.patient_context.gestational_weeks
        week_text = f"at {gestation} weeks" if gestation else "during pregnancy"
        transcript = specialist_outputs.get("transcript")
        transcript_line = " Voice check-in was reviewed." if transcript else ""
        top_reasons = GeminiOrchestrator._compact_list(decision.reasons, limit=2, max_chars=180)
        top_reasons_brief = GeminiOrchestrator._compact_list(decision.reasons, limit=1, max_chars=160)
        top_actions = GeminiOrchestrator._compact_list(decision.action_items, limit=3, max_chars=180)
        reason_text = ", ".join(top_reasons_brief) if top_reasons_brief else "No acute flags detected."
        summary_reason_text = ", ".join(top_reasons) if top_reasons else reason_text
        first_action = top_actions[0] if top_actions else "Continue routine monitoring and daily check-ins."

        patient_message = (
            f"Momnitrix assessed your latest check-in {week_text}. "
            f"Current risk level is {final_risk.upper()}. "
            f"Main reasons: {reason_text}. "
            f"{first_action}.{transcript_line}"
        )
        visit_summary = (
            "Visit Prep Summary:\n"
            f"- Risk: {final_risk}\n"
            f"- Key findings: {summary_reason_text}\n"
            f"- Recommended actions: {', '.join(top_actions) if top_actions else first_action}"
        )
        return {
            "patient_message": patient_message,
            "visit_prep_summary": visit_summary,
        }

    async def compose(
        self,
        request: TriageStreamRequest,
        decision: MedGemmaRiskResponse,
        specialist_outputs: dict[str, Any],
        final_risk: str,
    ) -> dict[str, str]:
        prompt = self._build_prompt(request, decision, specialist_outputs, final_risk)

        try:
            remote = await self._call_gemini(prompt)
            if remote:
                return remote
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            print(f"[momnitrix] gemini_fallback: HTTPStatusError status={status}")
        except Exception as exc:
            print(f"[momnitrix] gemini_fallback: {type(exc).__name__}: {exc}")

        return self._fallback_response(request, decision, specialist_outputs, final_risk)
