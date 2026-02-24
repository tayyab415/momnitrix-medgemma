"""Audit CSV/JSON maternal input datasets against QA input contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class RowRecord:
    source: str
    row_index: int
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CSV/JSON records against qa_console/input_contract.json",
    )
    parser.add_argument("--csv", action="append", default=[], help="Path to CSV input (repeatable)")
    parser.add_argument("--json", action="append", default=[], help="Path to JSON input (repeatable)")
    parser.add_argument(
        "--contract",
        default="qa_console/input_contract.json",
        help="Path to input contract JSON",
    )
    parser.add_argument(
        "--report",
        default="reports/input_audit_report.md",
        help="Path for markdown report output",
    )
    parser.add_argument(
        "--report-json",
        default="reports/input_audit_report.json",
        help="Path for machine-readable JSON report output",
    )
    return parser.parse_args()


def load_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("required_fields", [])
    payload.setdefault("numeric_fields", {})
    payload.setdefault("enum_fields", {})
    payload.setdefault("field_aliases", {})
    return payload


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "records", "scenarios", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


def load_records(csv_paths: list[str], json_paths: list[str]) -> list[RowRecord]:
    records: list[RowRecord] = []

    for raw_path in csv_paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                cleaned = {str(k).strip(): v for k, v in dict(row).items() if k is not None}
                records.append(RowRecord(source=str(path), row_index=index, row=cleaned))

    for raw_path in json_paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _json_rows(payload)
        for index, row in enumerate(rows, start=1):
            records.append(RowRecord(source=str(path), row_index=index, row=row))

    return records


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _get_nested_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = [row]

    for key in ("patient_context", "vitals", "inputs", "metadata", "profile"):
        value = row.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for value in row.values():
        if isinstance(value, dict) and value not in containers:
            containers.append(value)

    return containers


def _resolve_field(row: dict[str, Any], field: str, aliases: dict[str, list[str]]) -> Any:
    def _norm_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", key.strip().lower())

    def _lookup(container: dict[str, Any], key: str) -> tuple[bool, Any]:
        if key in container:
            return True, container[key]

        wanted_lower = key.strip().lower()
        wanted_norm = _norm_key(key)
        for existing_key, existing_value in container.items():
            k = str(existing_key)
            if k.strip().lower() == wanted_lower or _norm_key(k) == wanted_norm:
                return True, existing_value
        return False, None

    keys = [field]
    keys.extend(aliases.get(field, []))
    containers = _get_nested_candidates(row)

    for key in keys:
        if "." in key:
            parts = key.split(".")
            value: Any = row
            valid = True
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    valid = False
                    break
            if valid and not _is_missing(value):
                return value

        for container in containers:
            found, value = _lookup(container, key)
            if found and not _is_missing(value):
                return value

    return None


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": min(values),
        "p25": _percentile(values, 0.25) or min(values),
        "p50": _percentile(values, 0.50) or min(values),
        "p75": _percentile(values, 0.75) or min(values),
        "max": max(values),
    }


def audit_records(records: list[RowRecord], contract: dict[str, Any]) -> dict[str, Any]:
    required_fields: list[str] = contract.get("required_fields", [])
    numeric_fields: dict[str, dict[str, float]] = contract.get("numeric_fields", {})
    enum_fields: dict[str, list[str]] = contract.get("enum_fields", {})
    aliases: dict[str, list[str]] = contract.get("field_aliases", {})

    missing_counts: dict[str, int] = defaultdict(int)
    invalid_type_counts: dict[str, int] = defaultdict(int)
    out_of_range_counts: dict[str, int] = defaultdict(int)
    enum_invalid_counts: dict[str, int] = defaultdict(int)
    numeric_values: dict[str, list[float]] = defaultdict(list)

    findings: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for record in records:
        source_counts[record.source] += 1
        row = record.row

        for field in required_fields:
            value = _resolve_field(row, field, aliases)
            if field == "temp_c" and _is_missing(value):
                temp_f_value = _resolve_field(row, "temp_f", aliases)
                if not _is_missing(temp_f_value):
                    value = temp_f_value
            if _is_missing(value):
                missing_counts[field] += 1
                findings.append(
                    {
                        "type": "missing_required",
                        "field": field,
                        "source": record.source,
                        "row_index": record.row_index,
                        "value": value,
                    }
                )

        for field, bounds in numeric_fields.items():
            value = _resolve_field(row, field, aliases)

            # Support temperature in degF if only temp_f is present.
            if field == "temp_c" and _is_missing(value):
                temp_f = _resolve_field(row, "temp_f", aliases)
                temp_f_value = _to_number(temp_f)
                if temp_f_value is not None:
                    value = (temp_f_value - 32.0) * 5.0 / 9.0

            if _is_missing(value):
                continue

            number = _to_number(value)
            if number is None:
                invalid_type_counts[field] += 1
                findings.append(
                    {
                        "type": "invalid_numeric",
                        "field": field,
                        "source": record.source,
                        "row_index": record.row_index,
                        "value": value,
                    }
                )
                continue

            numeric_values[field].append(number)
            min_v = float(bounds["min"])
            max_v = float(bounds["max"])
            if number < min_v or number > max_v:
                out_of_range_counts[field] += 1
                findings.append(
                    {
                        "type": "out_of_range",
                        "field": field,
                        "source": record.source,
                        "row_index": record.row_index,
                        "value": number,
                        "min": min_v,
                        "max": max_v,
                    }
                )

        for field, allowed_values in enum_fields.items():
            value = _resolve_field(row, field, aliases)
            if _is_missing(value):
                continue
            normalized = str(value).strip().lower()
            allowed = {x.strip().lower() for x in allowed_values}
            if normalized not in allowed:
                enum_invalid_counts[field] += 1
                findings.append(
                    {
                        "type": "invalid_enum",
                        "field": field,
                        "source": record.source,
                        "row_index": record.row_index,
                        "value": value,
                        "allowed": sorted(allowed),
                    }
                )

        systolic = _to_number(_resolve_field(row, "systolic_bp", aliases))
        diastolic = _to_number(_resolve_field(row, "diastolic_bp", aliases))
        if systolic is not None and diastolic is not None and systolic <= diastolic:
            findings.append(
                {
                    "type": "cross_field",
                    "field": "blood_pressure_pair",
                    "source": record.source,
                    "row_index": record.row_index,
                    "value": f"{systolic}/{diastolic}",
                    "note": "systolic must be greater than diastolic",
                }
            )

        gravidity = _to_number(_resolve_field(row, "gravidity", aliases))
        parity = _to_number(_resolve_field(row, "parity", aliases))
        if gravidity is not None and parity is not None and parity > gravidity:
            findings.append(
                {
                    "type": "cross_field",
                    "field": "obstetric_history",
                    "source": record.source,
                    "row_index": record.row_index,
                    "value": f"G{gravidity}P{parity}",
                    "note": "parity should not exceed gravidity",
                }
            )

    total_rows = len(records)
    distributions = {
        field: _distribution(values)
        for field, values in numeric_values.items()
        if _distribution(values) is not None
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "total_sources": len(source_counts),
        "source_row_counts": dict(sorted(source_counts.items())),
        "missing_counts": dict(sorted(missing_counts.items())),
        "invalid_type_counts": dict(sorted(invalid_type_counts.items())),
        "out_of_range_counts": dict(sorted(out_of_range_counts.items())),
        "enum_invalid_counts": dict(sorted(enum_invalid_counts.items())),
        "findings_count": len(findings),
        "findings": findings,
        "distributions": distributions,
        "contract_path_hint": "qa_console/input_contract.json",
    }
    return summary


def render_markdown(summary: dict[str, Any], contract_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Momnitrix Input Audit Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Contract: `{contract_path}`")
    lines.append(f"- Sources: `{summary['total_sources']}`")
    lines.append(f"- Rows audited: `{summary['total_rows']}`")
    lines.append(f"- Total findings: `{summary['findings_count']}`")
    lines.append("")

    lines.append("## Source Coverage")
    lines.append("")
    if summary["source_row_counts"]:
        for source, count in summary["source_row_counts"].items():
            lines.append(f"- `{source}`: {count} rows")
    else:
        lines.append("- No input rows found.")
    lines.append("")

    lines.append("## Validation Counts")
    lines.append("")
    lines.append("| Category | Field | Count |")
    lines.append("|---|---|---:|")

    has_rows = False
    for category, bucket in (
        ("missing_required", summary["missing_counts"]),
        ("invalid_numeric", summary["invalid_type_counts"]),
        ("out_of_range", summary["out_of_range_counts"]),
        ("invalid_enum", summary["enum_invalid_counts"]),
    ):
        for field, count in bucket.items():
            has_rows = True
            lines.append(f"| {category} | `{field}` | {count} |")
    if not has_rows:
        lines.append("| none | none | 0 |")
    lines.append("")

    lines.append("## Numeric Distributions")
    lines.append("")
    lines.append("| Field | Count | Min | P25 | P50 | P75 | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    distributions = summary["distributions"]
    if distributions:
        for field, stats in distributions.items():
            lines.append(
                "| `{}` | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                    field,
                    int(stats["count"]),
                    stats["min"],
                    stats["p25"],
                    stats["p50"],
                    stats["p75"],
                    stats["max"],
                )
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines.append("")

    lines.append("## Top Findings (first 40)")
    lines.append("")
    findings = summary["findings"][:40]
    if findings:
        for finding in findings:
            lines.append(
                "- `{type}` field=`{field}` source=`{source}` row={row} value=`{value}`".format(
                    type=finding.get("type", "unknown"),
                    field=finding.get("field", "unknown"),
                    source=finding.get("source", "unknown"),
                    row=finding.get("row_index", "?"),
                    value=finding.get("value", ""),
                )
            )
    else:
        lines.append("- No findings. Inputs satisfy the configured contract bounds.")
    lines.append("")

    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.csv and not args.json:
        raise SystemExit("Provide at least one --csv or --json input path.")

    contract_path = Path(args.contract).expanduser().resolve()
    contract = load_contract(contract_path)
    records = load_records(args.csv, args.json)
    summary = audit_records(records, contract)
    markdown = render_markdown(summary, contract_path)

    report_path = Path(args.report).expanduser().resolve()
    report_json_path = Path(args.report_json).expanduser().resolve()
    write_report(report_path, markdown)
    write_json(report_json_path, summary)

    print(f"Audited {summary['total_rows']} rows across {summary['total_sources']} source file(s).")
    print(f"Findings: {summary['findings_count']}")
    print(f"Markdown report: {report_path}")
    print(f"JSON report: {report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
