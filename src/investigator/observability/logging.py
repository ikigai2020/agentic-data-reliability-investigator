"""Structured JSON logging (FR-1200, FR-1202).

A minimal Milestone-1 structured logger. Every event carries investigation/trace
context and is emitted as a single JSON line. Sensitive fields are redacted
(FR-1202). Full metric aggregation (FR-1203) is deferred to Milestone 4.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from ..config import load_config

_REDACT_PLACEHOLDER = "***redacted***"


def _redact(payload: dict[str, Any], redact_fields: tuple[str, ...]) -> dict[str, Any]:
    lowered = {f.lower() for f in redact_fields}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in lowered:
            out[key] = _REDACT_PLACEHOLDER
        elif isinstance(value, dict):
            out[key] = _redact(value, redact_fields)
        else:
            out[key] = value
    return out


class StructuredLogger:
    """Emits one JSON object per event to stderr (keeps stdout clean for stdio MCP)."""

    def __init__(self, investigation_id: str = "-", trace_id: str = "-") -> None:
        self.investigation_id = investigation_id
        self.trace_id = trace_id
        self._redact_fields = load_config().logging.redact_fields

    def log(self, event_type: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "investigation_id": self.investigation_id,
            "trace_id": self.trace_id,
            "event_type": event_type,
            **fields,
        }
        safe = _redact(record, self._redact_fields)
        print(json.dumps(safe, default=str), file=sys.stderr, flush=True)


def get_logger(investigation_id: str = "-", trace_id: str = "-") -> StructuredLogger:
    return StructuredLogger(investigation_id=investigation_id, trace_id=trace_id)
