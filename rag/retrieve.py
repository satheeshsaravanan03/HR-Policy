""""Retrieval methods and the hit-in-top-5 measurement.

Scores here are RELEVANCE, normalised to 0-1 where higher is better. Chroma's
raw similarity_search_with_score returns cosine DISTANCE, where lower is
better; a threshold written against that value silently inverts.

A hit requires both halves:
  - the chunk comes from the expected policy_id, and
  - the chunk actually contains the answer (Question.answer_pattern).

Matching on chunk_id instead would be meaningless, since the two strategies
generate different ids for the same underlying text by construction.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .index import reader
from .questions import QUESTIONS, Question

TOP_K = 5
SEMANTIC = "semantic"
BM25 = "bm25"
HYBRID = "hybrid"
SEARCH_METHODS = (SEMANTIC, BM25, HYBRID)

# RRF uses rank positions rather than mixing incompatible vector and BM25
# score scales. 60 is the conventional constant from the original RRF paper.
RRF_K = 60
FUSION_CANDIDATES = 20
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]*", re.I)


@dataclass
class Hit:
    rank: int
    chunk_id: str
    policy_id: str
    section: str
    region: str
    source_file: str
    score: float
    content: str
    semantic_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None

    def label(self) -> str:
        section = self.section or "-"
        return f"{self.policy_id} s.{section}"


@dataclass
class QuestionResult:
    question: Question
    strategy: str
    hits: list[Hit]
    hit_rank: int | None = None
    diagnosis: str = ""

    @property
    def is_hit(self) -> bool:
        return self.hit_rank is not None


@dataclass
class Measurement:
    strategy: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.is_hit)

    @property
    def score(self) -> str:
        return f"{self.hits}/{len(self.results)}"


def search(
    strategy: str,
    query: str,
    top_k: int = TOP_K,
    region: str | None = None,
    method: str = SEMANTIC,
) -> list[Hit]:
    """Retrieve chunks using semantic, BM25, or hybrid RRF search.

    ``semantic`` is the original Chroma vector search baseline. ``bm25`` uses
    exact-term matching over the same stored chunks. ``hybrid`` retrieves a
    broad candidate list from both methods and uses reciprocal-rank fusion to
    choose the final top-k. No documents are re-embedded for keyword search.
    """
    if method not in SEARCH_METHODS:
        raise ValueError(f"unknown search method {method!r}; use one of {SEARCH_METHODS}")

    if method == SEMANTIC:
        return _semantic_search(strategy, query, top_k, region)
    if method == BM25:
        return _bm25_search(strategy, query, top_k, region)
    return _hybrid_search(strategy, query, top_k, region)


def _hit(doc, rank: int, score: float, **scores) -> Hit:
    """Convert a Chroma document into the one public retrieval record."""
    return Hit(
        rank=rank,
        chunk_id=doc.metadata.get("chunk_id", "?"),
        policy_id=doc.metadata.get("policy_id", "?"),
        section=str(doc.metadata.get("section", "")),
        region=doc.metadata.get("region", "?"),
        source_file=doc.metadata.get("source_file", "?"),
        score=round(float(score), 4),
        content=doc.page_content,
        **scores,
    )


def _semantic_search(
    strategy: str, query: str, top_k: int, region: str | None
) -> list[Hit]:
    """The original vector retrieval path, retained unchanged as baseline."""
    store = reader(strategy)
    kwargs = {"k": top_k}
    if region:
        kwargs["filter"] = {"region": region}
    pairs = store.similarity_search_with_relevance_scores(query, **kwargs)

    return [
        _hit(doc, i + 1, score, semantic_score=round(float(score), 4))
        for i, (doc, score) in enumerate(pairs)
    ]


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class _StoredChunk:
    chunk_id: str
    content: str
    metadata: dict


@functools.lru_cache(maxsize=8)
def _stored_chunks(strategy: str) -> tuple[_StoredChunk, ...]:
    """Read each collection once for BM25; this is the exact Chroma corpus."""
    got = reader(strategy).get(include=["documents", "metadatas"])
    return tuple(
        _StoredChunk(chunk_id, content, metadata)
        for chunk_id, content, metadata in zip(
            got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
        )
        if content and metadata
    )


def _bm25_scores(query: str, chunks: list[_StoredChunk]) -> list[tuple[_StoredChunk, float]]:
    """A compact Okapi BM25 implementation; no extra package is required."""
    terms = _tokens(query)
    if not terms or not chunks:
        return []
    tokenised = [_tokens(chunk.content) for chunk in chunks]
    lengths = [len(tokens) for tokens in tokenised]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency = Counter(term for tokens in tokenised for term in set(tokens))
    query_terms = set(terms)
    k1, b = 1.5, 0.75
    scored: list[tuple[_StoredChunk, float]] = []
    for chunk, tokens, length in zip(chunks, tokenised, lengths):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            # BM25's standard Robertson/Sparck Jones IDF variant.
            idf = math.log(1 + (len(chunks) - document_frequency[term] + 0.5)
                           / (document_frequency[term] + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += idf * frequency * (k1 + 1) / denominator
        if score > 0:
            scored.append((chunk, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))


def _stored_hit(chunk: _StoredChunk, rank: int, score: float, **scores) -> Hit:
    class StoredDocument:
        page_content = chunk.content
        metadata = {**chunk.metadata, "chunk_id": chunk.chunk_id}

    return _hit(StoredDocument(), rank, score, **scores)


def _bm25_search(
    strategy: str, query: str, top_k: int, region: str | None
) -> list[Hit]:
    chunks = list(_stored_chunks(strategy))
    if region:
        chunks = [chunk for chunk in chunks if chunk.metadata.get("region") == region]
    return [
        _stored_hit(chunk, rank, score, bm25_score=round(score, 4))
        for rank, (chunk, score) in enumerate(_bm25_scores(query, chunks)[:top_k], start=1)
    ]


def _hybrid_search(
    strategy: str, query: str, top_k: int, region: str | None
) -> list[Hit]:
    """Fuse broad semantic and keyword candidate lists using RRF."""
    candidate_count = max(FUSION_CANDIDATES, top_k)
    semantic = _semantic_search(strategy, query, candidate_count, region)
    keyword = _bm25_search(strategy, query, candidate_count, region)

    fused: dict[str, dict] = {}
    for hits, kind in ((semantic, "semantic"), (keyword, "bm25")):
        for hit in hits:
            record = fused.setdefault(
                hit.chunk_id,
                {"hit": hit, "rrf": 0.0, "semantic_score": None, "bm25_score": None},
            )
            record["rrf"] += 1 / (RRF_K + hit.rank)
            record[f"{kind}_score"] = hit.score

    ordered = sorted(fused.values(), key=lambda item: (-item["rrf"], item["hit"].chunk_id))[:top_k]
    return [
        Hit(
            rank=rank,
            chunk_id=item["hit"].chunk_id,
            policy_id=item["hit"].policy_id,
            section=item["hit"].section,
            region=item["hit"].region,
            source_file=item["hit"].source_file,
            score=round(item["rrf"], 4),
            content=item["hit"].content,
            semantic_score=item["semantic_score"],
            bm25_score=item["bm25_score"],
            rrf_score=round(item["rrf"], 4),
        )
        for rank, item in enumerate(ordered, start=1)
    ]


def _diagnose(question: Question, hits: list[Hit]) -> str:
    """Why this question missed: the useful half of a failed retrieval."""
    if not hits:
        return "no results returned"
    right_doc = [h for h in hits if h.policy_id == question.policy_id]
    if not right_doc:
        got = ", ".join(sorted({h.policy_id for h in hits}))
        return f"wrong document entirely; top-5 came from {got}"
    ranks = ", ".join(f"#{h.rank} s.{h.section or '-'}" for h in right_doc)
    return (
        f"correct document retrieved ({ranks}) but no chunk contained the answer "
        f"-- the clause was split away from this section"
    )


def evaluate(strategy: str, top_k: int = TOP_K, method: str = SEMANTIC) -> Measurement:
    """Run all known-answer questions against one retrieval method."""
    measurement = Measurement(strategy=strategy)
    for question in QUESTIONS:
        hits = search(strategy, question.query, top_k=top_k, method=method)
        hit_rank = next(
            (
                h.rank
                for h in hits
                if h.policy_id == question.policy_id and question.matches(h.content)
            ),
            None,
        )
        measurement.results.append(
            QuestionResult(
                question=question,
                strategy=strategy,
                hits=hits,
                hit_rank=hit_rank,
                diagnosis="" if hit_rank else _diagnose(question, hits),
            )
        )
    return measurement
