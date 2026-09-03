"""Optional local cross-encoder reranking for retrieved policy chunks."""

from __future__ import annotations

import functools
import os
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieve import Hit


# FastEmbed runs this ONNX cross-encoder locally. Jina Turbo is the improved
# default; set RERANK_MODEL to the MiniLM model to reproduce the baseline.
MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v1-turbo-en")


@functools.lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=MODEL)


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Score query/chunk pairs locally and return the strongest ``top_k``."""
    if not hits:
        return []
    scores = list(_model().rerank(query, [hit.content for hit in hits]))

    def adjusted(hit: Hit, score: float) -> float:
        # A heading-only chunk is not useful evidence even when its words match
        # the query. Apply a clear quality penalty while retaining the model
        # score for substantive chunks.
        text = hit.content.strip()
        if len(text) < 100 or " " not in text or not any(mark in text for mark in ".:;\n"):
            return float(score) - 1.0
        return float(score)

    ranked = [(hit, float(score), adjusted(hit, score)) for hit, score in zip(hits, scores)]
    ordered = sorted(ranked, key=lambda item: (-item[2], item[0].chunk_id))
    return [
        replace(
            hit,
            rank=rank,
            score=round(adjusted_score, 4),
            rerank_score=round(adjusted_score, 4),
            retrieval_rank=hit.rank,
            retrieval_score=hit.score,
        )
        for rank, (hit, _raw_score, adjusted_score) in enumerate(ordered[:top_k], start=1)
    ]
