"""Requirement 4: show a region filter changing the top-1 result.

Carry-over is the right query to probe, because all four leave-bearing
documents answer it and they disagree: nine leaves in India, 320 hours in
Arizona, a per-series schedule in New York, and no faculty vacation accrual at
all in Michigan. Unfiltered, whichever phrasing embeds closest wins regardless
of which region the employee actually works in.

Both result lists are printed in full with scores, since that is what the
rubric asks to see.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunkers import STRUCTURE  # noqa: E402
from rag.retrieve import search  # noqa: E402

CANDIDATES = [
    ("How many days of unused leave can I carry over to next year?", "India"),
    ("What is the carry over cap for unused leave?", "India"),
    ("How much accrued vacation carries forward each year?", "Arizona"),
]


def dump(label: str, hits) -> None:
    print(f"\n{label}")
    print(f"  {'#':<3}{'score':<9}{'policy_id':<13}{'section':<10}{'region':<12}chunk_id")
    for h in hits:
        print(f"  {h.rank:<3}{h.score:<9.4f}{h.policy_id:<13}{(h.section or '-'):<10}"
              f"{h.region:<12}{h.chunk_id}")


def main() -> None:
    demonstrated = False

    for query, region in CANDIDATES:
        unfiltered = search(STRUCTURE, query)
        filtered = search(STRUCTURE, query, region=region)

        if not unfiltered or not filtered:
            continue

        changed = unfiltered[0].chunk_id != filtered[0].chunk_id

        print("\n" + "=" * 78)
        print(f"QUERY: {query}")
        print(f"FILTER: region = {region!r}")
        print("=" * 78)
        dump("UNFILTERED (top-5):", unfiltered)
        dump(f"FILTERED region={region} (top-5):", filtered)

        print(f"\n  top-1 unfiltered: {unfiltered[0].policy_id} "
              f"s.{unfiltered[0].section or '-'} (region {unfiltered[0].region})")
        print(f"  top-1 filtered:   {filtered[0].policy_id} "
              f"s.{filtered[0].section or '-'} (region {filtered[0].region})")
        print(f"  TOP-1 CHANGED: {changed}")

        if changed:
            demonstrated = True
            print("\n  Why this matters: an employee in the filtered region would have been")
            print("  shown another jurisdiction's entitlement as their own.")
            break

    if not demonstrated:
        print("\nNo candidate query changed top-1; reporting all attempts above rather")
        print("than searching for a query that flatters the filter.")


if __name__ == "__main__":
    main()
