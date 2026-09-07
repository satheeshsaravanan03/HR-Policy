"""Week 6 one-command evaluation suite.

Runs the same cases against the semantic baseline and the current improved
pipeline, then writes machine-readable JSON and a human-readable Markdown
before/after report.  It deliberately uses cheap assertions first; no LLM call
is needed to validate retrieval, refusals, citations, or regression cases.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.generate import refusal_check  # noqa: E402
from rag.questions import QUESTIONS, REFUSALS  # noqa: E402
from rag.retrieve import HYBRID, RERANK_LOCAL, SEMANTIC, search  # noqa: E402

OUT = ROOT / "output"
JSON_OUT = OUT / "week6_eval.json"
MD_OUT = OUT / "week6_eval.md"


def cases() -> list[dict]:
    rows = []
    for q in QUESTIONS:
        rows.append({"id": q.qid, "set": "answerable", "query": q.query,
                     "policy_id": q.policy_id, "pattern": q.answer_pattern,
                     "expect_refusal": False})
    for q in REFUSALS:
        rows.append({"id": q.qid, "set": "refusal", "query": q.query,
                     "policy_id": None, "pattern": None, "expect_refusal": True})

    # Permanent regression tests from previously observed Week 5 failures.
    rows.extend([
        {"id": "REG-01", "set": "regression", "query": "What is the nottice periond for a permanent Acme employee?",
         "policy_id": "ACME-EMP-2026", "pattern": r"30 calendar days of written notice", "expect_refusal": False},
        {"id": "REG-02", "set": "regression", "query": "How many days per week may an eligible Northstar employee work remotely?",
         "policy_id": "NORTHSTAR-REMOTE-2026", "pattern": r"three days per week", "expect_refusal": False},
        {"id": "REG-03", "set": "regression", "query": "Does SoftSuave have a sabbatical leave entitlement?",
         "policy_id": None, "pattern": None, "expect_refusal": True},
    ])
    return rows


def evaluate_case(case: dict, strategy: str, method: str, rerank: str) -> dict:
    hits = search(strategy, case["query"], top_k=5, method=method, rerank=rerank)
    answerable_hit = any(
        h.policy_id == case["policy_id"] and re.search(case["pattern"], h.content, re.I | re.S)
        for h in hits
    ) if not case["expect_refusal"] else None
    refused, gate, _ = refusal_check(case["query"], hits)
    assertions = {
        "retrieved": bool(hits),
        "expected_answer_found": bool(answerable_hit) if not case["expect_refusal"] else True,
        "expected_refusal": refused == case["expect_refusal"],
        "citation_source_available": bool(hits) if not case["expect_refusal"] else True,
    }
    return {
        "id": case["id"], "set": case["set"], "strategy": strategy,
        "method": method, "rerank": rerank, "score": sum(assertions.values()) / len(assertions),
        "assertions": assertions, "hit_rank": next((h.rank for h in hits if not case["expect_refusal"] and h.policy_id == case["policy_id"] and re.search(case["pattern"], h.content, re.I | re.S)), None),
        "refused": refused, "gate": gate,
        "top": [{"rank": h.rank, "chunk_id": h.chunk_id, "policy_id": h.policy_id, "section": h.section, "score": h.score} for h in hits],
    }


def run() -> None:
    all_cases = cases()
    results = []
    for label, method, rerank in (("baseline", SEMANTIC, "off"), ("improved", HYBRID, RERANK_LOCAL)):
        for case in all_cases:
            result = evaluate_case(case, "structure", method, rerank)
            result["version"] = label
            results.append(result)

    summary = {}
    for version in ("baseline", "improved"):
        subset = [r for r in results if r["version"] == version]
        summary[version] = {"overall": round(sum(r["score"] for r in subset) / len(subset), 3)}
        for group in sorted({r["set"] for r in subset}):
            group_rows = [r for r in subset if r["set"] == group]
            summary[version][group] = round(sum(r["score"] for r in group_rows) / len(group_rows), 3)

    payload = {"cases": all_cases, "results": results, "summary": summary,
               "assertion_definition": "retrieved, expected answer/refusal, and source availability are each worth one point"}
    OUT.mkdir(exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Week 6 Evaluation", "", "Rule-based assertions run before any optional LLM judge.", "",
             "| Version | Overall | Answerable | Refusal | Regression |", "|---|---:|---:|---:|---:|"]
    for version in ("baseline", "improved"):
        s = summary[version]
        lines.append(f"| {version} | {s['overall']:.1%} | {s['answerable']:.1%} | {s['refusal']:.1%} | {s['regression']:.1%} |")
    lines += ["", "## Per-case assertions", "", "| Version | Set | ID | Score | Assertions |", "|---|---|---|---:|---|"]
    for r in results:
        marks = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in r["assertions"].items())
        lines.append(f"| {r['version']} | {r['set']} | {r['id']} | {r['score']:.1%} | {marks} |")
    lines += ["", "The improved configuration is hybrid RRF retrieval with the local cross-encoder reranker; the baseline is semantic retrieval without reranking."]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(MD_OUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
