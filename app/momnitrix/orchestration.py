"""End-to-end triage orchestration logic."""

from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

from momnitrix.gemini import GeminiOrchestrator
from momnitrix.gateway import ModelGateway
from momnitrix.risk import compute_policy_floor, heuristic_medgemma_decision
from momnitrix.schemas import FinalTriageResponse, TriageStreamRequest
from momnitrix.storage import ArtifactStore
from momnitrix.utils import chunk_text, max_risk, utc_now

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def _normalize_composer_mode(raw: Any) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
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


def _normalize_ui_mode(raw: Any) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
    mapping = {
        "text": "text",
        "voice": "voice",
        "audio": "voice",
        "image": "image",
        "camera": "image",
        "auto": "auto",
    }
    return mapping.get(token, "auto")


def _required_inputs_complete(request: TriageStreamRequest) -> bool:
    return not _missing_required_inputs(request)


def _missing_required_inputs(request: TriageStreamRequest) -> list[str]:
    v = request.vitals
    p = request.patient_context
    missing: list[str] = []
    if p.age_years is None:
        missing.append("patient_context.age_years")
    if v.systolic_bp is None:
        missing.append("vitals.systolic_bp")
    if v.diastolic_bp is None:
        missing.append("vitals.diastolic_bp")
    if v.fasting_glucose_mmol_l is None:
        missing.append("vitals.fasting_glucose_mmol_l")
    if v.temp_c is None:
        missing.append("vitals.temp_c")
    if v.hr is None:
        missing.append("vitals.hr")
    return missing


def _build_router_decision(request: TriageStreamRequest) -> dict[str, Any]:
    metadata = request.metadata or {}
    note = str(request.inputs.free_text or "").lower()
    has_wound = bool(request.inputs.wound_image_b64)
    has_skin = bool(request.inputs.skin_image_b64)
    has_audio = bool(request.inputs.audio_b64)
    has_text = bool((request.inputs.free_text or "").strip())
    ui_mode = _normalize_ui_mode(metadata.get("ui_mode"))
    if ui_mode == "auto":
        if has_audio:
            ui_mode = "voice"
        elif has_wound or has_skin:
            ui_mode = "image"
        else:
            ui_mode = "text"

    selected: list[str] = []
    reason_parts: list[str] = []
    intent = "maternal_diagnosis"
    prompt_strategy = "maternal_diagnosis"

    if has_audio:
        selected.append("medasr")
        reason_parts.append("Audio input present -> MedASR enabled.")
        intent = "voice_symptom_triage"

    image_words_wound = {"wound", "incision", "stitch", "c_section", "infection", "discharge"}
    image_words_derm = {"rash", "itch", "derm", "skin", "eczema", "hives", "urticaria"}
    wound_hint = any(word in note for word in image_words_wound)
    derm_hint = any(word in note for word in image_words_derm)

    if has_wound and has_skin:
        selected.extend(["medsiglip", "derm"])
        intent = "multimodal_image_assessment"
        prompt_strategy = "multimodal_fusion"
        reason_parts.append("Both wound and skin images present -> MedSigLIP + Derm enabled.")
    elif has_wound:
        selected.append("medsiglip")
        intent = "wound_assessment"
        prompt_strategy = "wound_focus"
        reason_parts.append("Wound image present -> MedSigLIP enabled.")
        if derm_hint:
            selected.append("derm")
            prompt_strategy = "multimodal_fusion"
            reason_parts.append("Derm keywords detected -> Derm added.")
    elif has_skin:
        selected.append("derm")
        intent = "derm_assessment"
        prompt_strategy = "derm_focus"
        reason_parts.append("Skin image present -> Derm enabled.")
        if wound_hint:
            selected.append("medsiglip")
            prompt_strategy = "multimodal_fusion"
            reason_parts.append("Wound keywords detected -> MedSigLIP added.")

    if ui_mode == "text" and has_text and not selected:
        intent = "maternal_question_or_triage"
        prompt_strategy = "general_medical_qa"
        reason_parts.append("Text-first interaction -> Q/A aware prompt strategy.")

    # De-duplicate while preserving order.
    selected = list(dict.fromkeys(selected))
    return {
        "router_mode": str(metadata.get("router_preference") or "hybrid_policy"),
        "ui_mode": ui_mode,
        "intent": intent,
        "prompt_strategy": prompt_strategy,
        "selected_specialists": selected,
        "reason": " ".join(reason_parts) if reason_parts else "Default maternal triage routing.",
    }


def _clean_generated_line(value: Any) -> str:
    text = str(value or "")
    text = text.replace("```json", " ").replace("```", " ")
    text = " ".join(text.split()).strip("` ")
    if not text:
        return ""
    if text in {"{", "}", "[", "]"}:
        return ""
    if re.search(r'(?i)"?risk_level"?\s*:', text):
        return ""
    if re.search(r'(?i)^json\s*$', text):
        return ""
    return text


def _clean_list_items(items: list[str], *, limit: int, fallback: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        line = _clean_generated_line(item)
        if not line:
            continue
        key = "".join(ch for ch in line.lower() if ch.isalnum())
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
        if len(cleaned) >= limit:
            break
    if cleaned:
        return cleaned
    return [fallback]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _merge_unique_lines(primary: list[str], secondary: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in (primary, secondary):
        for line in source:
            cleaned = " ".join(str(line).split()).strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith("safety floor escalation"):
                continue
            key = "".join(ch for ch in cleaned.lower() if ch.isalnum())
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def _extract_patient_concern(
    request: TriageStreamRequest,
    specialist_outputs: dict[str, Any],
    *,
    max_chars: int = 220,
) -> str:
    def _concern_from_text(text: str) -> str:
        has_headache = bool(re.search(r"\bhead[a-z]*ach[a-z]*\b|\bhead pain\b", text, flags=re.IGNORECASE))
        has_swelling = bool(re.search(r"\bpuff[a-z]*\b|\bswel[a-z]*\b|\bedema\b", text, flags=re.IGNORECASE))
        has_vision = bool(re.search(r"\bvision\b|\bblur[a-z]*\b|\bspot[s]?\b|\bflash[a-z]*\b", text, flags=re.IGNORECASE))
        asks_urgency = bool(re.search(r"\bworr[a-z]*\b|\bscared\b|\bafraid\b|\burgent\b|\bserious\b", text, flags=re.IGNORECASE))
        has_fetal_movement = bool(re.search(r"\bfetal movement\b|\bbaby movement\b|\bnot moving\b", text, flags=re.IGNORECASE))

        findings: list[str] = []
        if has_headache:
            findings.append("persistent headache")
        if has_swelling:
            findings.append("swelling/puffiness")
        if has_vision:
            findings.append("vision symptoms")
        if has_fetal_movement:
            findings.append("possible decreased fetal movement")
        if findings:
            concern = "Voice check-in reports " + ", ".join(findings)
            if asks_urgency:
                concern += "; patient asks if this is urgent."
            return concern
        return ""

    transcript = str(specialist_outputs.get("transcript") or "").strip()
    if transcript:
        cleaned = re.sub(r"\s+", " ", transcript).strip()
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
        deduped_tokens: list[str] = []
        prev = ""
        for token in cleaned.split(" "):
            lower = token.lower()
            if lower == prev and lower not in {"very", "really"}:
                continue
            deduped_tokens.append(token)
            prev = lower
        cleaned = " ".join(deduped_tokens).strip()
        derived = _concern_from_text(cleaned)
        if derived:
            return derived
        if len(cleaned) > max_chars:
            return cleaned[:max_chars].rstrip() + "..."
        return cleaned

    note = str(request.inputs.free_text or "").strip()
    if not note:
        return ""
    if "Clinical note:" in note:
        note = note.split("Clinical note:", 1)[1].strip()

    lines = [line.strip(" -") for line in note.splitlines() if line.strip()]
    filtered = [
        line
        for line in lines
        if not re.match(
            r"(?i)^(patient profile|age:|obstetric history:|gestational age:|bmi group:)",
            line,
        )
    ]
    concern = re.sub(r"\s+", " ", " ".join(filtered)).strip()
    derived_note = _concern_from_text(concern)
    if derived_note:
        return derived_note
    if len(concern) > max_chars:
        return concern[:max_chars].rstrip() + "..."
    return concern


def _specialist_summary_lines(specialist_outputs: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    wound = specialist_outputs.get("wound_scores")
    if isinstance(wound, dict) and wound:
        urgency = float(wound.get("urgency", 0.0))
        infection = float(wound.get("infection_risk", 0.0))
        if urgency >= 0.6 or infection >= 0.7:
            lines.append(
                "Image specialist (wound) flagged elevated risk "
                f"(urgency {urgency:.2f}, infection {infection:.2f})."
            )
        else:
            lines.append(
                "Image specialist (wound) reviewed photo with no high-risk threshold crossed "
                f"(urgency {urgency:.2f}, infection {infection:.2f})."
            )

    skin_top3 = specialist_outputs.get("skin_top3")
    if isinstance(skin_top3, list) and skin_top3:
        labels: list[str] = []
        for row in skin_top3[:3]:
            if not isinstance(row, dict):
                continue
            condition = str(row.get("condition", "unknown")).replace("_", " ")
            score = float(row.get("score", 0.0))
            labels.append(f"{condition} ({score:.2f})")
        if labels:
            lines.append("Skin specialist top classes: " + ", ".join(labels) + ".")
    return lines


class TriageOrchestrator:
    def __init__(
        self,
        gateway: ModelGateway,
        gemini: GeminiOrchestrator,
        store: ArtifactStore,
        *,
        composer_mode: str = "gemini_full",
    ):
        self._gateway = gateway
        self._gemini = gemini
        self._store = store
        self._composer_mode = _normalize_composer_mode(composer_mode)

    def _resolve_composer_mode(self, request: TriageStreamRequest) -> str:
        metadata = request.metadata or {}
        request_override = (
            metadata.get("composer_mode")
            or metadata.get("response_composer_mode")
            or metadata.get("orchestration_mode")
        )
        if request_override:
            return _normalize_composer_mode(request_override)
        return self._composer_mode

    @staticmethod
    def _compose_medgemma_first_payload(
        request: TriageStreamRequest,
        *,
        specialist_outputs: dict[str, Any],
        final_risk: str,
        reasons: list[str],
        actions: list[str],
        clinical_summary: str,
        raw_response_text: str | None = None,
    ) -> dict[str, str]:
        concern = _extract_patient_concern(request, specialist_outputs)
        concern_line = f"- Patient concern raised: {concern}\n" if concern else ""
        specialist_lines = _specialist_summary_lines(specialist_outputs)
        specialist_block = "".join(f"- Specialist evidence: {line}\n" for line in specialist_lines)
        if raw_response_text and raw_response_text.strip():
            clean_raw = raw_response_text.strip()
            if specialist_lines and "specialist evidence:" not in clean_raw.lower():
                clean_raw = f"Specialist evidence: {specialist_lines[0]}\n\n{clean_raw}"
            return {
                "patient_message": clean_raw,
                "visit_prep_summary": (
                    "Visit Prep Summary:\n"
                    f"- Risk: {final_risk}\n"
                    f"{concern_line}"
                    f"{specialist_block}"
                    f"- Clinical summary: {(_clean_generated_line(clinical_summary) or reasons[0])}\n"
                    f"- Recommended actions: {'; '.join(actions[:3]) if actions else 'Continue routine prenatal follow-up.'}"
                ),
            }

        gestation = request.patient_context.gestational_weeks
        week_text = f"{gestation} weeks" if gestation else "this stage of pregnancy"
        risk_text = final_risk.upper()
        primary_reason = reasons[0]
        reason_text = primary_reason
        action_text = "; ".join(actions[:3])
        summary_text = primary_reason or (_clean_generated_line(clinical_summary) or reasons[0])

        patient_message = (
            f"Momnitrix reviewed your latest check-in at {week_text}. "
            f"Current risk level is {risk_text}. "
            f"Key findings: {reason_text}. "
            f"Next steps: {action_text}."
        )
        visit_prep_summary = (
            "Visit Prep Summary:\n"
            f"- Risk: {final_risk}\n"
            f"{concern_line}"
            f"{specialist_block}"
            f"- Clinical summary: {summary_text}\n"
            f"- Key reasons: {reason_text}\n"
            f"- Recommended actions: {action_text}"
        )
        return {
            "patient_message": patient_message,
            "visit_prep_summary": visit_prep_summary,
        }

    async def _store_artifacts(
        self,
        request_id: str,
        request: TriageStreamRequest,
        emit: EmitFn,
    ) -> dict[str, str]:
        refs: dict[str, str] = {}
        if request.inputs.wound_image_b64:
            refs["wound_image"] = await self._store.store_blob_b64(
                request_id,
                "raw_wound.jpg",
                request.inputs.wound_image_b64,
                content_type="image/jpeg",
            )
        if request.inputs.skin_image_b64:
            refs["skin_image"] = await self._store.store_blob_b64(
                request_id,
                "raw_skin.jpg",
                request.inputs.skin_image_b64,
                content_type="image/jpeg",
            )
        if request.inputs.audio_b64:
            refs["audio_clip"] = await self._store.store_blob_b64(
                request_id,
                "raw_audio.wav",
                request.inputs.audio_b64,
                content_type="audio/wav",
            )

        if refs:
            await emit("artifact.uploaded", {"request_id": request_id, "artifact_refs": refs})

        return refs

    async def run(
        self,
        request: TriageStreamRequest,
        emit: EmitFn,
    ) -> FinalTriageResponse:
        request_id = request.request_id or str(uuid4())
        trace_id = str(uuid4())
        started = perf_counter()

        await emit(
            "request.accepted",
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "timestamp": utc_now().isoformat(),
            },
        )

        missing_required = _missing_required_inputs(request)
        if missing_required:
            await emit(
                "request.rejected",
                {
                    "request_id": request_id,
                    "reason": "missing_required_inputs",
                    "missing_fields": missing_required,
                },
            )
            raise ValueError(
                "Missing required clinical inputs: " + ", ".join(missing_required)
            )

        route_context = _build_router_decision(request)
        await emit(
            "router.decision",
            {
                "request_id": request_id,
                **route_context,
            },
        )

        task_plan = await self._gemini.compose_task_instruction(request, route_context)
        resolved_intent = str(task_plan.get("intent") or route_context.get("intent") or "maternal_diagnosis")
        resolved_strategy = str(
            task_plan.get("prompt_strategy") or route_context.get("prompt_strategy") or "maternal_diagnosis"
        )
        medgemma_task_instruction = str(task_plan.get("medgemma_task_instruction") or "").strip()
        planner_source = str(task_plan.get("planner_source") or "fallback")

        metadata = dict(request.metadata or {})
        metadata.update(
            {
                "ui_mode": route_context.get("ui_mode"),
                "router_intent": resolved_intent,
                "router_prompt_strategy": resolved_strategy,
                "router_mode": route_context.get("router_mode"),
                "selected_specialists": list(route_context.get("selected_specialists") or []),
                "gemini_task_instruction": medgemma_task_instruction,
                "gemini_task_planner_source": planner_source,
            }
        )
        request = request.model_copy(update={"metadata": metadata})

        await emit(
            "router.prompt_plan",
            {
                "request_id": request_id,
                "intent": resolved_intent,
                "prompt_strategy": resolved_strategy,
                "planner_source": planner_source,
                "selected_specialists": metadata.get("selected_specialists", []),
                "instruction_preview": medgemma_task_instruction[:280],
            },
        )

        artifact_refs = await self._store_artifacts(request_id, request, emit)

        specialist_outputs: dict[str, Any] = {}
        latency_ms: dict[str, int] = {}
        selected_specialists = {
            str(x).strip().lower()
            for x in (request.metadata or {}).get("selected_specialists", [])
            if str(x).strip()
        }

        tasks: list[asyncio.Task[tuple[str, Any, str | None]]] = []

        async def _run_named(name: str, coro: Awaitable[Any]) -> tuple[str, Any, str | None]:
            t0 = perf_counter()
            try:
                result = await coro
                latency_ms[name] = int((perf_counter() - t0) * 1000)
                return name, result, None
            except Exception as exc:
                latency_ms[name] = int((perf_counter() - t0) * 1000)
                return name, None, str(exc)

        if request.inputs.wound_image_b64 and (
            "medsiglip" in selected_specialists or not selected_specialists
        ):
            await emit("model.started", {"request_id": request_id, "model": "medsiglip"})
            tasks.append(
                asyncio.create_task(
                    _run_named("medsiglip", self._gateway.medsiglip_infer(request.inputs.wound_image_b64))
                )
            )

        if request.inputs.skin_image_b64 and ("derm" in selected_specialists or not selected_specialists):
            await emit("model.started", {"request_id": request_id, "model": "derm"})
            tasks.append(
                asyncio.create_task(_run_named("derm", self._gateway.derm_infer(request.inputs.skin_image_b64)))
            )

        if request.inputs.audio_b64 and ("medasr" in selected_specialists or not selected_specialists):
            await emit("model.started", {"request_id": request_id, "model": "medasr"})
            tasks.append(
                asyncio.create_task(
                    _run_named("medasr", self._gateway.medasr_transcribe(request.inputs.audio_b64))
                )
            )

        for task in asyncio.as_completed(tasks):
            name, result, error = await task
            if error:
                await emit(
                    "model.failed",
                    {
                        "request_id": request_id,
                        "model": name,
                        "error": error,
                        "latency_ms": latency_ms.get(name, 0),
                    },
                )
                continue

            if name == "medsiglip":
                specialist_outputs["wound_scores"] = result
            elif name == "derm":
                scores, top3 = result
                specialist_outputs["skin_scores"] = scores
                specialist_outputs["skin_top3"] = top3
            elif name == "medasr":
                transcript = str(result or "").strip()
                if transcript and transcript.lower() not in {"none", "null"}:
                    specialist_outputs["transcript"] = transcript

            await emit(
                "model.completed",
                {
                    "request_id": request_id,
                    "model": name,
                    "latency_ms": latency_ms.get(name, 0),
                },
            )

        await emit("medgemma.started", {"request_id": request_id, "model": "medgemma"})
        medgemma_t0 = perf_counter()
        medgemma_meta: dict[str, Any] = {"engine": "unknown", "fallback_used": None}
        if hasattr(self._gateway, "medgemma_decide_with_meta"):
            medgemma_decision, medgemma_meta = await self._gateway.medgemma_decide_with_meta(
                request, specialist_outputs
            )
        else:
            medgemma_decision = await self._gateway.medgemma_decide(request, specialist_outputs)
        medgemma_runtime = medgemma_meta.get("runtime_diagnostics") or {}
        prompt_profile = str(medgemma_runtime.get("prompt_profile") or "unknown")
        warmup_ms = _as_int(medgemma_runtime.get("gpu_warmup_ms"), default=0)
        latency_ms["medgemma"] = int((perf_counter() - medgemma_t0) * 1000)
        infer_ms = _as_int(
            medgemma_runtime.get("medgemma_inference_ms"),
            default=max(latency_ms["medgemma"] - warmup_ms, 0),
        )
        cold_start = bool(medgemma_runtime.get("cold_start", warmup_ms > 0))
        await emit(
            "medgemma.completed",
            {
                "request_id": request_id,
                "model": str(medgemma_meta.get("engine") or "medgemma"),
                "risk_level": medgemma_decision.risk_level,
                "latency_ms": latency_ms["medgemma"],
                "fallback_used": bool(medgemma_meta.get("fallback_used")),
                "prompt_profile": prompt_profile,
                "cold_start": cold_start,
                "gpu_warmup_ms": warmup_ms,
                "medgemma_inference_ms": infer_ms,
            },
        )

        guardrail_decision = heuristic_medgemma_decision(request, specialist_outputs)
        policy_floor, floor_reasons = compute_policy_floor(request, specialist_outputs)
        medgemma_decision.reasons = _merge_unique_lines(
            floor_reasons + guardrail_decision.reasons,
            medgemma_decision.reasons,
            limit=5,
        )
        medgemma_decision.action_items = _merge_unique_lines(
            guardrail_decision.action_items,
            medgemma_decision.action_items,
            limit=6,
        )

        final_risk = max_risk(max_risk(medgemma_decision.risk_level, policy_floor), guardrail_decision.risk_level)
        if final_risk != medgemma_decision.risk_level:
            medgemma_decision.reasons = _merge_unique_lines(
                [f"Risk escalated by safety guardrail ({final_risk})."],
                medgemma_decision.reasons,
                limit=5,
            )

        medgemma_decision.reasons = _clean_list_items(
            medgemma_decision.reasons,
            limit=5,
            fallback="No acute flags detected from submitted signals.",
        )
        medgemma_decision.action_items = _clean_list_items(
            medgemma_decision.action_items,
            limit=6,
            fallback="Continue routine prenatal monitoring and follow-up.",
        )

        composer_mode = self._resolve_composer_mode(request)
        if composer_mode == "gemini_full":
            await emit("gemini.started", {"request_id": request_id, "model": self._gemini.model_name})
            gemini_t0 = perf_counter()
            composed = await self._gemini.compose(request, medgemma_decision, specialist_outputs, final_risk)
            latency_ms["gemini"] = int((perf_counter() - gemini_t0) * 1000)

            for chunk in chunk_text(composed["patient_message"], size=120):
                await emit(
                    "gemini.delta",
                    {
                        "request_id": request_id,
                        "text": chunk,
                    },
                )

            await emit(
                "gemini.completed",
                {
                    "request_id": request_id,
                    "model": self._gemini.model_name,
                    "latency_ms": latency_ms["gemini"],
                },
            )
        else:
            latency_ms["gemini"] = 0
            composed = self._compose_medgemma_first_payload(
                request,
                specialist_outputs=specialist_outputs,
                final_risk=final_risk,
                reasons=medgemma_decision.reasons,
                actions=medgemma_decision.action_items,
                clinical_summary=medgemma_decision.clinical_summary,
                raw_response_text=medgemma_decision.raw_response_text,
            )
            await emit(
                "gemini.skipped",
                {
                    "request_id": request_id,
                    "composer_mode": composer_mode,
                    "reason": "MedGemma-first mode enabled.",
                },
            )
            for chunk in chunk_text(composed["patient_message"], size=120):
                await emit(
                    "medgemma.delta",
                    {
                        "request_id": request_id,
                        "text": chunk,
                    },
                )

        latency_ms["total"] = int((perf_counter() - started) * 1000)
        llm_time = max(latency_ms["medgemma"] + latency_ms["gemini"], 1)
        medgemma_share = round((latency_ms["medgemma"] / llm_time) * 100.0, 1)
        gemini_share = round((latency_ms["gemini"] / llm_time) * 100.0, 1)
        authorship = {
            "risk_level": "medgemma+policy_guardrail",
            "action_items": "medgemma+policy_guardrail",
            "patient_message": "gemini" if composer_mode == "gemini_full" else "medgemma",
            "visit_prep_summary": "gemini" if composer_mode == "gemini_full" else "medgemma",
        }
        inference_diagnostics = {
            "composer_mode": composer_mode,
            "medgemma_engine": str(medgemma_meta.get("engine") or "unknown"),
            "medgemma_fallback_used": bool(medgemma_meta.get("fallback_used")),
            "medgemma_prompt_profile": prompt_profile,
            "medgemma_timing_breakdown": {
                "cold_start": cold_start,
                "gpu_warmup_ms": warmup_ms,
                "medgemma_inference_ms": infer_ms,
            },
            "latency_split_ms": {
                "medgemma": latency_ms["medgemma"],
                "gemini": latency_ms["gemini"],
                "llm_total": latency_ms["medgemma"] + latency_ms["gemini"],
            },
            "latency_share_pct": {
                "medgemma": medgemma_share,
                "gemini": gemini_share,
            },
            "field_authorship": authorship,
            "router": {
                "intent": resolved_intent,
                "prompt_strategy": resolved_strategy,
                "planner_source": planner_source,
                "ui_mode": str((request.metadata or {}).get("ui_mode") or "auto"),
                "selected_specialists": list((request.metadata or {}).get("selected_specialists") or []),
            },
        }
        await emit(
            "diagnostics.inference_breakdown",
            {
                "request_id": request_id,
                "model": f"medgemma:{medgemma_share:.1f}%|gemini:{gemini_share:.1f}%",
                "risk_level": final_risk,
                "latency_ms": latency_ms["total"],
                "gpu_warmup_ms": warmup_ms,
                "medgemma_inference_ms": infer_ms,
                "intent": resolved_intent,
                "prompt_strategy": resolved_strategy,
                "planner_source": planner_source,
                **inference_diagnostics,
            },
        )

        final = FinalTriageResponse(
            request_id=request_id,
            trace_id=trace_id,
            timestamp=utc_now(),
            risk_level=final_risk,
            policy_floor=policy_floor,
            patient_message=composed["patient_message"],
            visit_prep_summary=composed["visit_prep_summary"],
            action_items=medgemma_decision.action_items,
            medgemma_reasons=medgemma_decision.reasons,
            specialist_outputs=specialist_outputs,
            latency_ms=latency_ms,
            artifact_refs=artifact_refs,
            inference_diagnostics=inference_diagnostics,
        )

        await self._store.store_json(request_id, "final_response_compact.json", final.model_dump(mode="json"))
        await self._store.write_final_response(request_id, final.model_dump(mode="json"))

        await emit("triage.final", final.model_dump(mode="json"))

        return final
