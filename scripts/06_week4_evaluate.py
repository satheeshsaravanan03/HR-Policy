"""Week 4: prove the one retrieval change with before/after metrics.

This compares the unchanged semantic baseline with hybrid BM25 + semantic RRF
on the exact same pre-registered challenge questions.  It also prints a
separate inspection case that can expose a generation/context failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunkers import STRUCTURE  # noqa: E402
from rag.retrieve import HYBRID, SEMANTIC, search  # noqa: E402
from rag.week4 import GENERATION_INSPECTIONS, RETRIEVAL_CHALLENGES  # noqa: E402


TOP_K = 3


def rank_for(question, hits):
    return next(
        (
            hit.rank
            for hit in hits
            if hit.policy_id == question.policy_id and question.matches(hit.content)
        ),
        None,
    )


def evaluate(method: str) -> tuple[int, float]:
    ranks: list[int | None] = []
    print(f"\n{'=' * 78}\n{method.upper()} — hit-rate@{TOP_K} and MRR\n{'=' * 78}")
    for question in RETRIEVAL_CHALLENGES:
        hits = search(STRUCTURE, question.query, top_k=TOP_K, method=method)
        hit_rank = rank_for(question, hits)
        ranks.append(hit_rank)
        label = f"HIT rank {hit_rank}" if hit_rank else "RETRIEVAL FAILURE (correct evidence absent)"
        print(f"\n{question.qid}: {question.query}\n  expected: {question.policy_id} s.{question.section}\n  {label}")
        for hit in hits:
            marker = "<<<" if hit.rank == hit_rank else ""
            print(f"    {hit.rank}. {hit.policy_id} s.{hit.section or '-'} score={hit.score:.4f} {marker}")

    hits = sum(rank is not None for rank in ranks)
    mrr = sum(1 / rank if rank else 0 for rank in ranks) / len(ranks)
    print(f"\n{method}: hit-rate@{TOP_K} = {hits}/{len(ranks)} ({hits / len(ranks):.1%}) | MRR = {mrr:.3f}")
    return hits, mrr


def generation_inspection() -> None:
    print(f"\n{'=' * 78}\nGENERATION / CONTEXT INSPECTION (not counted in retrieval metric)\n{'=' * 78}")
    for case in GENERATION_INSPECTIONS:
        hits = search(STRUCTURE, case.query, top_k=TOP_K, method=HYBRID)
        retrieved_sections = {hit.section for hit in hits if hit.policy_id == case.expected_policy_id}
        complete = set(case.required_sections).issubset(retrieved_sections)
        print(f"\n{case.qid}: {case.query}\n  required context: {case.expected_policy_id} sections {', '.join(case.required_sections)}")
        print(f"  retrieved sections: {', '.join(sorted(retrieved_sections)) or '(none)'}")
        print(f"  context complete: {'yes' if complete else 'NO'}")
        print(f"  diagnosis: {case.diagnosis}")


def main() -> None:
    baseline_hits, baseline_mrr = evaluate(SEMANTIC)
    hybrid_hits, hybrid_mrr = evaluate(HYBRID)
    print(f"\n{'=' * 78}\nBEFORE / AFTER — ONE CHANGE: HYBRID BM25 + SEMANTIC RRF\n{'=' * 78}")
    print("| method | hit-rate@3 | MRR |")
    print("| --- | --- | --- |")
    print(f"| semantic baseline | {baseline_hits}/{len(RETRIEVAL_CHALLENGES)} | {baseline_mrr:.3f} |")
    print(f"| hybrid RRF | {hybrid_hits}/{len(RETRIEVAL_CHALLENGES)} | {hybrid_mrr:.3f} |")
    generation_inspection()


if __name__ == "__main__":
    main()
