"""Minimal JSONL tracing for Week 5 replay and error analysis."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / "output" / "traces.jsonl"
PROMPT_VERSION = "week5-v1"
_LOCK = threading.Lock()

# Conservative redaction for common employee identifiers before disk write.
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# Require an explicit identifier marker; do not redact ordinary phrases such
# as "employee at Bangalore" or "staff can work remotely".
_EMPLOYEE_ID = re.compile(
    r"\b(?:employee|emp|staff|worker)[ _-]?(?:id|no|number)\s*[:#-]?\s*[A-Z0-9-]{3,}\b",
    re.I,
)


def redact(value: str) -> str:
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    return _EMPLOYEE_ID.sub("[REDACTED_EMPLOYEE_ID]", value)


def write_trace(
    *,
    query: str,
    hits,
    output: str,
    refused: bool,
    gate: str = "",
    model: str = "",
    temperature: float | None = None,
    method: str = "semantic",
    strategy: str = "structure",
    rerank: str = "off",
) -> str:
    """Append one complete, replayable trace and return its trace ID."""
    trace_id = str(uuid.uuid4())
    record = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "query": redact(query),
        "settings": {
            "strategy": strategy,
            "method": method,
            "rerank": rerank,
            "model": model,
            "temperature": temperature,
        },
        "retrieved": [
            {
                "rank": h.rank,
                "chunk_id": h.chunk_id,
                "policy_id": h.policy_id,
                "section": h.section,
                "score": h.score,
                "retrieval_rank": h.retrieval_rank,
                "retrieval_score": h.retrieval_score,
                "content": redact(h.content),
            }
            for h in hits
        ],
        "refused": refused,
        "gate": gate,
        "raw_output": redact(output),
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return trace_id
