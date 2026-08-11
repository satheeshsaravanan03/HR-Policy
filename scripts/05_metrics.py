"""Metrics beyond hit-in-top-5, because top-5 saturated at 8/8 for both.

A metric both strategies max out cannot discriminate between them. This adds
tighter rank cutoffs, MRR, and the measure that actually separates the two:
whether the retrieved chunk can be cited to a policy section at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunkers import STRATEGIES  # noqa: E402
from rag.index import STRATEGY_NAMES  # noqa: E402
from rag.manifest import BY_FILE, document_paths  # noqa: E402
from rag.retrieve import evaluate  # noqa: E402


def section_attribution() -> dict[str, str]:
    """Share of leave-policy chunks that know their own section number."""
    out = {}
    for name, chunk in STRATEGIES.items():
        total = withsec = 0
        for path in document_paths():
            if not BY_FILE[path.name].carries_leave_policy:
                continue
            for c in chunk(path):
                total += 1
                if c.metadata.get("section"):
                    withsec += 1
        out[name] = f"{withsec}/{total} ({100 * withsec / total:.1f}%)"
    return out


def main() -> None:
    rows = {}
    for strategy in STRATEGY_NAMES:
        m = evaluate(strategy)
        ranks = [r.hit_rank for r in m.results]
        found = [r for r in ranks if r]
        rows[strategy] = {
            "hit@1": f"{sum(1 for r in found if r <= 1)}/8",
            "hit@3": f"{sum(1 for r in found if r <= 3)}/8",
            "hit@5": f"{sum(1 for r in found if r <= 5)}/8",
            "MRR": f"{sum(1 / r for r in found) / len(ranks):.3f}",
            "top1_citable": f"{sum(1 for r in m.results if r.hits and r.hits[0].section)}/8",
        }

    attribution = section_attribution()

    print("\n| Metric | " + " | ".join(STRATEGY_NAMES) + " |")
    print("| --- | " + " | ".join("---" for _ in STRATEGY_NAMES) + " |")
    for metric in ("hit@1", "hit@3", "hit@5", "MRR", "top1_citable"):
        print(f"| {metric} | " + " | ".join(rows[s][metric] for s in STRATEGY_NAMES) + " |")
    print("| chunks carrying a section number | "
          + " | ".join(attribution[s] for s in STRATEGY_NAMES) + " |")

    print("\ntop1_citable = the rank-1 chunk carries a section number, so a citation")
    print("can name the policy section rather than only the file.")


if __name__ == "__main__":
    main()
