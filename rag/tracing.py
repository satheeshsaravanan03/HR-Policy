"""Privacy-aware JSONL request traces for RAG observability."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / "output" / "traces.jsonl"
PROMPT_VERSION = "week11-v1"
_LOCK = threading.Lock()

# Conservative redaction for common employee identifiers before disk write.
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# Require an explicit identifier marker; do not redact ordinary phrases such
# as "employee at Bangalore" or "staff can work remotely".
_EMPLOYEE_ID = re.compile(
    r"\b(?:employee|emp|staff|worker)[ _-]?(?:id|no|number)\s*[:#-]?\s*[A-Z0-9-]{3,}\b|\bEMP[-_ ]?\d{3,}\b",
    re.I,
)
_DEMO_NAME = re.compile(r"\bDemo\s+Employee\s+\d{3,}\b", re.I)
_PERSON_NAME = re.compile(
    r"\b(?:(?i:employee|user|person|for|about|does)\s+)(?:named\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)
_PERSONAL_QUERY = re.compile(
    r"\b(?:EMP[-_ ]?\d{3,}|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b"
    r"|\bDemo\s+Employee\s+\d{3,}\b"
    r"|\b(?:list|show|find|which)\s+(?:all\s+)?employees\b"
    r"|\bemployee\s+(?:details|record|balance)\b",
    re.I,
)
_PERSON_NAME_QUERY = re.compile(
    r"\b(?:(?i:employee|user|person|for|about|does)\s+)(?:named\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
)
_API_KEY = re.compile(r"\b(?:gsk_[A-Za-z0-9_-]{12,}|sk-ant-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b")


def redact(value: str) -> str:
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _EMPLOYEE_ID.sub("[REDACTED_EMPLOYEE_ID]", value)
    value = _DEMO_NAME.sub("[REDACTED_EMPLOYEE_NAME]", value)
    value = _PERSON_NAME.sub("[REDACTED_PERSON_NAME]", value)
    for key_name in ("GROQ_API_KEY", "QDRANT_API_KEY", "GOOGLE_API_KEY", "MCP_DEMO_TOKEN"):
        secret = os.environ.get(key_name)
        if secret:
            value = value.replace(secret, "[REDACTED_SECRET]")
    return _API_KEY.sub("[REDACTED_API_KEY]", value)


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
    region: str | None = None,
    top_k: int | None = None,
    timings_ms: dict[str, float | None] | None = None,
    usage: dict[str, int | None] | None = None,
    estimated_cost: float | None = None,
    status: str | None = None,
    error: str | None = None,
) -> str:
    """Append one request trace and return its ID; omit employee-specific answers."""
    trace_id = str(uuid.uuid4())
    employee_specific = bool(_PERSONAL_QUERY.search(query) or _PERSON_NAME_QUERY.search(query))
    safe_output = "[EMPLOYEE-SPECIFIC ANSWER REDACTED]" if employee_specific else redact(output)
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
            "region": region,
            "top_k": top_k,
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
                "content": "[EMPLOYEE-SPECIFIC CONTEXT REDACTED]" if employee_specific else redact(h.content),
            }
            for h in hits
        ],
        "refused": refused,
        "gate": gate,
        "status": status or ("refused" if refused else "success"),
        "employee_specific": employee_specific,
        "employee_details_redacted": employee_specific,
        "raw_output": safe_output,
        "timings_ms": timings_ms or {},
        "usage": usage or {},
        "estimated_cost_usd": estimated_cost,
        "error": redact(error or "") or None,
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return trace_id
