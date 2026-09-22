"""Append-only Week 8 agent trajectory records."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .tracing import redact

TRAJECTORY_PATH = Path(__file__).resolve().parent.parent / "output" / "agent_trajectories.jsonl"
_LOCK = threading.Lock()


def write_trajectory(*, query: str, steps, status: str, workflows_called,
                     answer: str, stop_reason: str, metrics: dict,
                     safety_findings=None, expected_path=None) -> str:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": redact(query),
        "steps": steps,
        "status": status,
        "workflows_called": workflows_called,
        "raw_output": redact(answer),
        "stop_reason": stop_reason,
        "metrics": metrics,
        "safety_findings": safety_findings or [],
        "expected_path": expected_path or [],
    }
    TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, TRAJECTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["timestamp"]


def load_trajectories() -> list[dict]:
    """Load records, including recovery of concatenated JSON lines."""
    if not TRAJECTORY_PATH.exists():
        return []
    text = TRAJECTORY_PATH.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict] = []
    position = 0
    while position < len(text):
        try:
            value, end = decoder.raw_decode(text, position)
            if isinstance(value, dict):
                records.append(value)
            position = end
        except json.JSONDecodeError:
            position += 1
        while position < len(text) and text[position].isspace():
            position += 1
    return records
