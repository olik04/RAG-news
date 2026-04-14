from __future__ import annotations

import json
import logging
from typing import Any


def log_llm_event(
    logger: logging.Logger,
    *,
    operation: str,
    provider: str,
    model: str,
    outcome: str,
    duration_ms: int,
    reason: str | None = None,
    request_id: str | None = None,
    doc_count: int | None = None,
) -> None:
    """Emit a structured, sanitized telemetry event for LLM execution paths."""
    event: dict[str, Any] = {
        "event": "llm_provider_telemetry",
        "operation": operation,
        "provider": provider,
        "model": model,
        "outcome": outcome,
        "duration_ms": max(0, int(duration_ms)),
    }
    if reason:
        event["reason"] = reason
    if request_id:
        event["request_id"] = str(request_id)
    if doc_count is not None:
        event["doc_count"] = max(0, int(doc_count))
    logger.info(json.dumps(event, separators=(",", ":"), sort_keys=True))