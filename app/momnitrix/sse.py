"""Server-sent event helpers."""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {data}\n\n"
