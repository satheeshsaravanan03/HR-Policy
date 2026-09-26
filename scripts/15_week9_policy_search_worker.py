"""Internal subprocess entry point for the Week 9 MCP policy-search tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# This worker is launched by absolute path from scripts/, so add the root for
# imports of the existing application's rag package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieve import search


def main() -> int:
    request = json.loads(sys.stdin.read())
    hits = search(
        strategy=request["strategy"],
        query=request["query"],
        top_k=request["top_k"],
        region=request.get("region"),
        method=request["method"],
        policy_id=request.get("policy_id"),
    )
    response: dict[str, Any] = {
        "query": request["query"],
        "strategy": request["strategy"],
        "method": request["method"],
        "result_count": len(hits),
        "results": [
            {
                "rank": hit.rank,
                "policy_id": hit.policy_id,
                "section": hit.section,
                "region": hit.region,
                "source_file": hit.source_file,
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "semantic_score": hit.semantic_score,
                "bm25_score": hit.bm25_score,
                "rrf_score": hit.rrf_score,
                "text": hit.content,
            }
            for hit in hits
        ],
    }
    # stdout is reserved for the parent MCP process's JSON payload.
    # Escape non-ASCII and any unpaired PDF text surrogates for a valid UTF-8
    # subprocess result that the MCP server can decode reliably.
    print(json.dumps(response, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
