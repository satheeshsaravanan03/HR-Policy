"""Optional local cross-encoder reranking for retrieved policy chunks."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieve import Hit


# FastEmbed runs this ONNX cross-encoder locally. It downloads once and then
# needs neither a Gemini key nor network access for subsequent queries.
MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@functools.lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=MODEL)


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Score query/chunk pairs locally and return the strongest ``top_k``."""
    if not hits:
        return []
    scores = list(_model().rerank(query, [hit.content for hit in hits]))
    ordered = sorted(zip(hits, scores), key=lambda item: (-float(item[1]), item[0].chunk_id))
    return [
        replace(
            hit,
            rank=rank,
            score=round(float(score), 4),
            rerank_score=round(float(score), 4),
            retrieval_rank=hit.rank,
            retrieval_score=hit.score,
        )
        for rank, (hit, score) in enumerate(ordered[:top_k], start=1)
    ]
