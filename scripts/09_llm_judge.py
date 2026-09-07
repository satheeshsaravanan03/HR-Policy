"""Week 6 LLM-as-judge with a small human-labelled calibration set.

Usage:
  python scripts/09_llm_judge.py           # grades six calibration cases

The calibration labels are deliberately explicit and versioned in this file:
three known answerable questions and three known refusals.  The judge is not
trusted until its pass/fail decision agrees with those labels.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_groq import ChatGroq  # noqa: E402

from rag.generate import answer  # noqa: E402
from rag.questions import BY_QID, REFUSALS  # noqa: E402

OUT = ROOT / "output" / "week6_judge.json"
MODEL = os.environ.get("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")

CALIBRATION = (
    ("Q1", True), ("Q3", True), ("Q7", True),
    ("R1", False), ("R2", False), ("R3", False),
)

JUDGE_PROMPT = """You are a strict evaluator of an HR policy RAG answer.
Return JSON only with integer fields 0 or 1:
faithfulness, relevance, completeness, refusal_correct, pass.
For an answerable question, refusal_correct is 1 only when the answer is not a refusal.
For an unanswerable question, refusal_correct is 1 only when the answer clearly refuses.
Faithfulness means every claim is supported by the supplied context and citations.
Relevance means it answers the exact question. Completeness means it includes the
known answer without inventing policy. pass is 1 only if all applicable checks pass.

QUESTION: {question}
EXPECTED ANSWERABLE: {answerable}
KNOWN ANSWER: {known}
CONTEXT:
{context}
APP ANSWER:
{answer}
"""


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"judge did not return JSON: {text[:300]}")
    value = json.loads(match.group(0))
    return {key: int(bool(value.get(key, 0))) for key in
            ("faithfulness", "relevance", "completeness", "refusal_correct", "pass")}


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required for the LLM judge")
    llm = ChatGroq(model=MODEL, temperature=0.0, max_retries=2)
    rows = []
    for qid, answerable in CALIBRATION:
        if qid.startswith("R"):
            case = next(item for item in REFUSALS if item.qid == qid)
            question, known = case.query, "This question must be refused."
        else:
            case = BY_QID[qid]
            question, known = case.query, case.known_answer
        result = answer("structure", question, top_k=5, method="hybrid", rerank="local")
        context = "\n\n".join(hit.content for hit in result.hits)
        response = llm.invoke(JUDGE_PROMPT.format(
            question=question, answerable=answerable, known=known,
            context=context, answer=result.text,
        ))
        judged = parse_json(str(response.content))
        human_pass = (not result.is_refusal) if answerable else result.is_refusal
        rows.append({"qid": qid, "question": question, "human_pass": int(human_pass),
                     "judge": judged, "app_refused": result.is_refusal})

    agreement = sum(row["human_pass"] == row["judge"]["pass"] for row in rows) / len(rows)
    payload = {"model": MODEL, "calibration_size": len(rows), "human_judge_agreement": agreement,
               "validated": agreement >= 0.80, "rows": rows}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
