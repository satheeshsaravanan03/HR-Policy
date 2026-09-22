"""Week 8 safety checks for untrusted retrieved text."""

from __future__ import annotations

import re

INJECTION_PATTERNS = (
    r"ignore\s+(?:all|any|the)\s+(?:previous|prior|system|developer)\s+instructions",
    r"reveal\s+(?:the\s+)?(?:employee|customer)\s+(?:records|data)",
    r"disregard\s+(?:the\s+)?(?:system|developer)\s+message",
    r"follow\s+these\s+instructions\s+instead",
    r"print\s+(?:the\s+)?(?:api|secret|password)\s+key",
)


def injection_match(text: str) -> str | None:
    lowered = text or ""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.I):
            return pattern
    return None


def unsafe_hits(hits) -> list[dict[str, str]]:
    findings = []
    for hit in hits or []:
        matched = injection_match(hit.content)
        if matched:
            findings.append({"chunk_id": hit.chunk_id, "pattern": matched})
    return findings
