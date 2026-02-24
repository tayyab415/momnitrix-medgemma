#!/usr/bin/env python3
"""Benchmark Momnitrix response paths on shared maternal scenarios.

Compares:
- orchestrator (gemini_full composer mode)
- orchestrator (medgemma_first composer mode)
- optional direct MedGemma endpoints (LoRA/base) if provided
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


RISK_RANK = {"green": 0, "yellow": 1, "red": 2}


@dataclass
class PathResult:
    name: str
    ok: bool
    risk_level: str | None
    policy_floor: str | None
    latency_ms: int | None
    error: str | None


def _expected_min_risk(fasting_glucose: float | None) -> str:
    if fasting_glucose is None:
        return "green"
    if fasting_glucose >= 10.0:
        return "red"
    if fasting_glucose >= 7.0:
        return "yellow"
    return "green"


def _risk_at_least(actual: str | None, expected_min: str) -> bool:
    if actual not in RISK_RANK:
        return False
    return RISK_RANK[actual] >= RISK_RANK[expected_min]


def _default_cases() -> list[dict[str, Any]]:
    return [
        {
            "request_id": "bench-low-1",
            "patient_context": {"gestational_weeks": 28},
            "vitals": {"systolic_bp": 118, "diastolic_bp": 76, "fasting_glucose": 4.9, "hr": 82, "temp_c": 36.8},
            "inputs": {"free_text": "Routine monitoring update."},
        },
        {
            "request_id": "bench-yellow-glucose",
            "patient_context": {"gestational_weeks": 31},
            "vitals": {"systolic_bp": 126, "diastolic_bp": 82, "fasting_glucose": 7.7, "hr": 86, "temp_c": 36.9},
            "inputs": {"free_text": "No major complaints, glucose has been elevated."},
        },
        {
            "request_id": "bench-red-glucose",
            "patient_context": {"gestational_weeks": 33},
            "vitals": {"systolic_bp": 132, "diastolic_bp": 84, "fasting_glucose": 12.4, "hr": 90, "temp_c": 37.0},
            "inputs": {"free_text": "Persistent high fasting sugars this week."},
        },
        {
            "request_id": "bench-red-bp",
            "patient_context": {"gestational_weeks": 35},
            "vitals": {"systolic_bp": 166, "diastolic_bp": 112, "fasting_glucose": 5.2, "hr": 97, "temp_c": 37.1},
            "inputs": {"headache": True, "vision_changes": True, "free_text": "Severe headache and blurred vision."},
        },
    ]


def _request_from_row(row: dict[str, str], idx: int) -> dict[str, Any]:
    def _to_int(*keys: str) -> int | None:
        for key in keys:
            v = row.get(key)
            if v is None or str(v).strip() == "":
                continue
            try:
                return int(float(str(v).strip()))
            except ValueError:
                continue
        return None

    def _to_float(*keys: str) -> float | None:
        for key in keys:
            v = row.get(key)
            if v is None or str(v).strip() == "":
                continue
            try:
                return float(str(v).strip())
            except ValueError:
                continue
        return None

    gest_weeks = _to_int("gestational_weeks", "Gestational Age", "gestation_week", "week")
    systolic = _to_int("systolic_bp", "SystolicBP", "systolic")
    diastolic = _to_int("diastolic_bp", "DiastolicBP", "diastolic")
    glucose = _to_float(
        "fasting_glucose",
        "fasting_glucose_mmol_l",
        "Fasting Blood Sugar (mmol/L)",
        "blood_sugar",
        "glucose",
    )
    hr = _to_int("hr", "heart_rate", "Heart Rate")
    temp_c = _to_float("temp_c", "temperature_c")

    free_text = (
        f"CSV case {idx + 1}. "
        f"Age={row.get('Age') or row.get('age') or 'unknown'}, "
        f"G={row.get('G') or row.get('gravidity') or row.get('Pregnancies') or 'unknown'}, "
        f"P={row.get('P') or row.get('parity') or 'unknown'}."
    )

    return {
        "request_id": f"bench-csv-{idx + 1}",
        "patient_context": {"gestational_weeks": gest_weeks},
        "vitals": {
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "fasting_glucose": glucose,
            "hr": hr,
            "temp_c": temp_c,
        },
        "inputs": {"free_text": free_text},
    }


def _load_cases(cases_json: Path | None, maternal_csv: Path | None, *, limit: int) -> list[dict[str, Any]]:
    if cases_json:
        raw = json.loads(cases_json.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            cases = raw
        elif isinstance(raw, dict):
            cases = list(raw.values())
        else:
            raise ValueError("cases JSON must be a list or object")
        return cases[:limit]

    if maternal_csv:
        out: list[dict[str, Any]] = []
        with maternal_csv.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            for idx, row in enumerate(reader):
                out.append(_request_from_row(row, idx))
                if len(out) >= limit:
                    break
        return out

    return _default_cases()[:limit]


async def _stream_triage(
    client: httpx.AsyncClient,
    api_base_url: str,
    payload: dict[str, Any],
    *,
    composer_mode: str,
) -> PathResult:
    req = dict(payload)
    metadata = dict(req.get("metadata") or {})
    metadata["composer_mode"] = composer_mode
    req["metadata"] = metadata

    url = f"{api_base_url.rstrip('/')}/v1/triage/stream"
    started = time.perf_counter()
    event_name = "message"
    data_lines: list[str] = []
    final_payload: dict[str, Any] | None = None

    try:
        async with client.stream("POST", url, json=req, timeout=420.0) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip("\r")
                if line == "":
                    if data_lines:
                        data_raw = "\n".join(data_lines)
                        data_lines = []
                        payload_obj: dict[str, Any] | None = None
                        try:
                            payload_obj = json.loads(data_raw)
                        except json.JSONDecodeError:
                            payload_obj = None
                        if event_name == "triage.final" and isinstance(payload_obj, dict):
                            final_payload = payload_obj
                            break
                    event_name = "message"
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if not final_payload:
            return PathResult(
                name=f"orchestrator:{composer_mode}",
                ok=False,
                risk_level=None,
                policy_floor=None,
                latency_ms=elapsed_ms,
                error="missing triage.final",
            )

        model_latency = (final_payload.get("latency_ms") or {}).get("total")
        return PathResult(
            name=f"orchestrator:{composer_mode}",
            ok=True,
            risk_level=final_payload.get("risk_level"),
            policy_floor=final_payload.get("policy_floor"),
            latency_ms=int(model_latency) if isinstance(model_latency, int) else elapsed_ms,
            error=None,
        )
    except Exception as exc:
        return PathResult(
            name=f"orchestrator:{composer_mode}",
            ok=False,
            risk_level=None,
            policy_floor=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


async def _direct_medgemma(
    client: httpx.AsyncClient,
    core_url: str,
    payload: dict[str, Any],
    *,
    label: str,
) -> PathResult:
    url = f"{core_url.rstrip('/')}/internal/medgemma/risk_decide"
    body = {
        "patient_context": payload.get("patient_context", {}),
        "vitals": payload.get("vitals", {}),
        "inputs": payload.get("inputs", {}),
        "specialist_outputs": {},
    }
    started = time.perf_counter()
    try:
        response = await client.post(url, json=body, timeout=420.0)
        response.raise_for_status()
        data = response.json()
        return PathResult(
            name=f"direct:{label}",
            ok=True,
            risk_level=data.get("risk_level"),
            policy_floor=data.get("risk_level"),
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=None,
        )
    except Exception as exc:
        return PathResult(
            name=f"direct:{label}",
            ok=False,
            risk_level=None,
            policy_floor=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        expected = row["expected_min_risk"]
        for result in row["results"]:
            name = result["name"]
            rec = by_path.setdefault(
                name,
                {
                    "runs": 0,
                    "ok_runs": 0,
                    "latency_ms_values": [],
                    "risk_floor_passes": 0,
                    "errors": [],
                },
            )
            rec["runs"] += 1
            if result["ok"]:
                rec["ok_runs"] += 1
                if isinstance(result["latency_ms"], int):
                    rec["latency_ms_values"].append(result["latency_ms"])
                if _risk_at_least(result["risk_level"], expected):
                    rec["risk_floor_passes"] += 1
            elif result["error"]:
                rec["errors"].append(result["error"])

    summary_paths: dict[str, Any] = {}
    for name, rec in by_path.items():
        latencies = rec["latency_ms_values"]
        summary_paths[name] = {
            "runs": rec["runs"],
            "ok_runs": rec["ok_runs"],
            "success_rate": round(rec["ok_runs"] / rec["runs"], 4) if rec["runs"] else 0.0,
            "risk_floor_pass_rate": round(rec["risk_floor_passes"] / rec["runs"], 4) if rec["runs"] else 0.0,
            "latency_ms_avg": int(sum(latencies) / len(latencies)) if latencies else None,
            "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1] if len(latencies) >= 2 else (
                latencies[0] if latencies else None
            ),
            "sample_errors": rec["errors"][:5],
        }
    return summary_paths


def _markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Momnitrix Path Benchmark Report",
        "",
        f"Total cases: {len(rows)}",
        "",
        "## Aggregate",
        "",
        "| Path | Runs | Success | Floor-pass | Avg latency (ms) | P95 latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, rec in summary.items():
        lines.append(
            f"| `{name}` | {rec['runs']} | {rec['success_rate']:.2%} | {rec['risk_floor_pass_rate']:.2%} | "
            f"{rec['latency_ms_avg'] if rec['latency_ms_avg'] is not None else '-'} | "
            f"{rec['latency_ms_p95'] if rec['latency_ms_p95'] is not None else '-'} |"
        )
    lines.extend(["", "## Per Case", ""])
    for row in rows:
        lines.append(
            f"- `{row['request_id']}` glucose={row['fasting_glucose']} mmol/L expected_min={row['expected_min_risk']}"
        )
        for result in row["results"]:
            lines.append(
                f"  - {result['name']}: ok={result['ok']} risk={result['risk_level']} "
                f"latency_ms={result['latency_ms']} error={result['error'] or '-'}"
            )
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> int:
    cases = _load_cases(args.cases_json, args.maternal_csv, limit=args.limit)
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for idx, case in enumerate(cases):
            request_id = str(case.get("request_id") or f"bench-{idx + 1}")
            vitals = case.get("vitals") or {}
            fasting_glucose = vitals.get("fasting_glucose")
            if fasting_glucose is None:
                fasting_glucose = vitals.get("fasting_glucose_mmol_l")
            expected_min = _expected_min_risk(float(fasting_glucose) if fasting_glucose is not None else None)

            path_results: list[PathResult] = []
            path_results.append(
                await _stream_triage(client, args.api_base_url, case, composer_mode="gemini_full")
            )
            path_results.append(
                await _stream_triage(client, args.api_base_url, case, composer_mode="medgemma_first")
            )

            if args.lora_core_url:
                path_results.append(
                    await _direct_medgemma(client, args.lora_core_url, case, label="lora_core")
                )
            if args.base_core_url:
                path_results.append(
                    await _direct_medgemma(client, args.base_core_url, case, label="base_core")
                )

            rows.append(
                {
                    "request_id": request_id,
                    "fasting_glucose": fasting_glucose,
                    "expected_min_risk": expected_min,
                    "results": [
                        {
                            "name": r.name,
                            "ok": r.ok,
                            "risk_level": r.risk_level,
                            "policy_floor": r.policy_floor,
                            "latency_ms": r.latency_ms,
                            "error": r.error,
                        }
                        for r in path_results
                    ],
                }
            )
            print(f"[bench] case {idx + 1}/{len(cases)} complete: {request_id}")

    summary = _aggregate(rows)
    report = {
        "api_base_url": args.api_base_url,
        "lora_core_url": args.lora_core_url,
        "base_core_url": args.base_core_url,
        "case_count": len(rows),
        "summary": summary,
        "cases": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown_report(summary, rows), encoding="utf-8")

    print(f"[bench] wrote JSON report: {args.output_json}")
    print(f"[bench] wrote Markdown report: {args.output_md}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Momnitrix composition paths")
    parser.add_argument("--api-base-url", required=True, help="Momnitrix orchestrator base URL")
    parser.add_argument(
        "--cases-json",
        type=Path,
        help="Optional JSON file with triage request payloads (list or map)",
    )
    parser.add_argument(
        "--maternal-csv",
        type=Path,
        help="Optional maternal CSV to auto-build benchmark cases",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max number of cases to run")
    parser.add_argument(
        "--lora-core-url",
        help="Optional core GPU URL serving your LoRA MedGemma endpoint",
    )
    parser.add_argument(
        "--base-core-url",
        help="Optional core GPU URL serving base MedGemma endpoint",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/benchmark_path_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/benchmark_path_report.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
