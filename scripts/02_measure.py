"""Search-only measurement: hit-in-top-5 for both chunking strategies.

Prints the full per-question record for all 8 questions under both
strategies, because the rubric awards the 30 points for the record, not for
a summary claim. No generation happens here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import STRATEGY_NAMES  # noqa: E402
from rag.retrieve import evaluate  # noqa: E402


def main() -> None:
    measurements = {}

    for strategy in STRATEGY_NAMES:
        measurement = evaluate(strategy)
        measurements[strategy] = measurement

        print("\n" + "=" * 78)
        print(f"SEARCH-ONLY DUMP -- strategy: {strategy}")
        print("=" * 78)

        for result in measurement.results:
            q = result.question
            verdict = f"HIT at rank {result.hit_rank}" if result.is_hit else "MISS"
            print(f"\n{q.qid}: {q.query}")
            print(f"  expected: {q.policy_id} section {q.section}"
                  f"{'  [table row]' if q.from_table_row else ''}")
            print(f"  known answer: {q.known_answer}")
            print(f"  verdict: {verdict}")
            for h in result.hits:
                mark = "<<<" if result.hit_rank == h.rank else "   "
                print(f"    {h.rank}. {h.score:.4f}  {h.label():22} {h.region:10} {h.chunk_id} {mark}")
            if result.diagnosis:
                print(f"  diagnosis: {result.diagnosis}")

    print("\n" + "=" * 78)
    print("HIT-IN-TOP-5 -- same 8 questions, same embedding model")
    print("=" * 78)
    print(f"\n| Chunking strategy | Hit-in-top-5 |")
    print(f"| --- | --- |")
    for strategy, m in measurements.items():
        print(f"| {strategy} | {m.score} |")

    print("\nPer-question comparison:")
    print(f"  {'qid':5} {'expected':24} " + " ".join(f"{s:>12}" for s in STRATEGY_NAMES))
    for i, q in enumerate(measurements[STRATEGY_NAMES[0]].results):
        cells = []
        for s in STRATEGY_NAMES:
            r = measurements[s].results[i]
            cells.append(f"rank {r.hit_rank}" if r.is_hit else "MISS")
        expected = f"{q.question.policy_id} s.{q.question.section}"
        print(f"  {q.question.qid:5} {expected:24} " + " ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
