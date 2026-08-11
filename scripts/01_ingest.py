"""Index the corpus under both chunking strategies.

Run in two phases on purpose, to make requirement 6 checkable rather than
asserted. Phase 1 indexes the Soft Suave base handbook alone, standing in for
the index that already existed. Phase 2 adds the five new documents and
reports that the handbook was skipped, not re-embedded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.index import STRATEGY_NAMES, collection_stats, ingest  # noqa: E402
from rag.manifest import DOCUMENTS  # noqa: E402

BASE_HANDBOOK = "SoftSuave-Employee-Handbook-2025.pdf"
NEW_DOCUMENTS = [d.source_file for d in DOCUMENTS if d.source_file != BASE_HANDBOOK]


def main() -> None:
    for strategy in STRATEGY_NAMES:
        print(f"\n=== PHASE 1: base handbook only [{strategy}] ===")
        ingest(strategy, [BASE_HANDBOOK])

    for strategy in STRATEGY_NAMES:
        print(f"\n=== PHASE 2: the {len(NEW_DOCUMENTS)}-document drop [{strategy}] ===")
        result = ingest(strategy, [BASE_HANDBOOK] + NEW_DOCUMENTS)
        print(f"  added {result['added_chunks']} chunks from {len(result['new_files'])} new files")

    print("\n=== FINAL COLLECTION STATS ===")
    for strategy in STRATEGY_NAMES:
        stats = collection_stats(strategy)
        print(f"  {strategy:10} {stats}")
        assert stats["chunks_missing_source_file"] == 0, "failed ingest: chunk without source_file"
    print("\n  every chunk carries source_file (requirement 1 satisfied)")


if __name__ == "__main__":
    main()
