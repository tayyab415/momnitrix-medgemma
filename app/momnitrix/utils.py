"""Common utility helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter


_RISK_ORDER = {"green": 0, "yellow": 1, "red": 2}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_ms() -> float:
    return perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> int:
    return int(perf_counter() * 1000.0 - start_ms)


def max_risk(a: str, b: str) -> str:
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b


def risk_rank(risk: str) -> int:
    return _RISK_ORDER[risk]


def chunk_text(text: str, size: int = 120) -> list[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]
