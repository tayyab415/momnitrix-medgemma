"""Artifact and event persistence for Momnitrix.

Primary target is S3. A local filesystem fallback is always kept for dev/test and for
serving `GET /v1/triage/{request_id}` without external dependencies.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from momnitrix.config import Settings
from momnitrix.utils import utc_now


class ArtifactStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._root = Path(settings.local_storage_dir)
        (self._root / "artifacts").mkdir(parents=True, exist_ok=True)
        (self._root / "responses").mkdir(parents=True, exist_ok=True)
        (self._root / "logs").mkdir(parents=True, exist_ok=True)
        self._events_file = self._root / "logs" / "events.jsonl"

        self._s3 = None
        if settings.s3_bucket:
            try:
                import boto3  # type: ignore

                self._s3 = boto3.client("s3", region_name=settings.s3_region)
            except Exception:
                self._s3 = None

    @staticmethod
    def _clean_base64(value: str) -> bytes:
        if "," in value and value.strip().startswith("data:"):
            _, _, value = value.partition(",")
        return base64.b64decode(value, validate=False)

    def _s3_key(self, request_id: str, name: str) -> str:
        prefix = self._settings.s3_prefix.strip("/")
        return f"{prefix}/{request_id}/{name}" if prefix else f"{request_id}/{name}"

    async def store_blob_b64(
        self,
        request_id: str,
        name: str,
        data_b64: str,
        *,
        content_type: str,
    ) -> str:
        content = self._clean_base64(data_b64)
        key = self._s3_key(request_id, name)

        if self._s3 is not None and self._settings.s3_bucket:
            self._s3.put_object(
                Bucket=self._settings.s3_bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
            )
            uri = f"s3://{self._settings.s3_bucket}/{key}"
        else:
            local_path = self._root / "artifacts" / request_id / name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            uri = str(local_path)

        return uri

    async def store_json(self, request_id: str, name: str, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
        key = self._s3_key(request_id, name)

        if self._s3 is not None and self._settings.s3_bucket:
            self._s3.put_object(
                Bucket=self._settings.s3_bucket,
                Key=key,
                Body=raw,
                ContentType="application/json",
            )
            uri = f"s3://{self._settings.s3_bucket}/{key}"
        else:
            local_path = self._root / "artifacts" / request_id / name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(raw)
            uri = str(local_path)

        return uri

    async def write_final_response(self, request_id: str, payload: dict[str, Any]) -> str:
        local_path = self._root / "responses" / f"{request_id}.json"
        local_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

        remote_uri = await self.store_json(request_id, "final_response.json", payload)
        return remote_uri

    def read_final_response(self, request_id: str) -> dict[str, Any] | None:
        local_path = self._root / "responses" / f"{request_id}.json"
        if not local_path.exists():
            return None
        return json.loads(local_path.read_text(encoding="utf-8"))

    async def append_event(self, event_name: str, payload: dict[str, Any]) -> None:
        envelope = {
            "timestamp": utc_now().isoformat(),
            "event": event_name,
            "payload": payload,
        }
        with self._events_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(envelope, ensure_ascii=True, default=str) + "\n")
