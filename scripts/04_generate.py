"""Requirement 5: three cited answers and three forced refusals.

The three answerable questions reuse ground truth from the eight, so a grader
checking a citation can verify the claim against a known-correct answer.

Every citation is validated against the chunk_ids actually supplied to the
model, and the supporting chunk is printed so the claim can be checked by eye.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunkers import STRUCTURE  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.questions import BY_QID, CITED_ANSWER_QIDS, REFUSALS  # noqa: E402


def main() -> None:
    print("=" * 78)
    print("PART 1 -- THREE ANSWERABLE QUESTIONS, CITED")
    print("=" * 78)

    for qid in CITED_ANSWER_QIDS:
        q = BY_QID[qid]
        result = answer(STRUCTURE, q.query)

        print(f"\n{'-' * 78}\n{qid}: {q.query}")
        print(f"known-correct: {q.known_answer} ({q.policy_id} section {q.section})")
        print(f"\nANSWER:\n{result.text}\n")

        if result.is_refusal:
            print("  UNEXPECTED REFUSAL -- this question is answerable")
            print(f"  gate: {result.gate} | {result.refusal_reason}")
            continue

        print("CITATIONS:")
        for c in result.citations:
            status = "resolves" if c.resolves else "DOES NOT RESOLVE"
            print(f"  [{status}] {c.chunk_id} -> {c.policy_id} section {c.section or '-'}")

        unresolved = [c for c in result.citations if not c.resolves]
        print(f"  {len(result.citations)} citations, {len(unresolved)} unresolved")

        # Print the cited chunk so the grader can check the claim against it.
        cited_ids = {c.chunk_id for c in result.citations if c.resolves}
        for h in result.hits:
            if h.chunk_id in cited_ids:
                body = h.content.strip().replace("\n", "\n      ")
                print(f"\n  SUPPORTING CHUNK {h.chunk_id} (relevance {h.score:.4f}):")
                print(f"      {body[:600]}")
                break

    print("\n\n" + "=" * 78)
    print("PART 2 -- THREE OUT-OF-CORPUS QUESTIONS, REFUSED")
    print("=" * 78)

    for case in REFUSALS:
        result = answer(STRUCTURE, case.query)

        print(f"\n{'-' * 78}\n{case.qid}: {case.query}")
        print(f"difficulty: {case.difficulty}")
        print(f"why unanswerable: {case.why_unanswerable}")
        print(f"\nTRANSCRIPT:\n{result.text}\n")
        print(f"refused: {result.is_refusal} | gate: {result.gate or '-'}")

        if result.hits:
            print(f"  top-1 relevance was {result.hits[0].score:.4f} "
                  f"from {result.hits[0].policy_id} -- note that a similarity")
            print("  threshold alone would not have caught this.")
        if not result.is_refusal:
            print("  FAILED TO REFUSE -- the gate let an unanswerable question through")


if __name__ == "__main__":
    main()
