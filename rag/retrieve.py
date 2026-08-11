"""Search-only retrieval and the hit-in-top-5 measurement.

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

from dataclasses import dataclass, field

from .index import reader
from .questions import QUESTIONS, Question

TOP_K = 5


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
) -> list[Hit]:
    """Top-k chunks by relevance, optionally restricted to one region."""
    store = reader(strategy)
    kwargs = {"k": top_k}
    if region:
        kwargs["filter"] = {"region": region}
    pairs = store.similarity_search_with_relevance_scores(query, **kwargs)

    return [
        Hit(
            rank=i + 1,
            chunk_id=doc.metadata.get("chunk_id", "?"),
            policy_id=doc.metadata.get("policy_id", "?"),
            section=str(doc.metadata.get("section", "")),
            region=doc.metadata.get("region", "?"),
            source_file=doc.metadata.get("source_file", "?"),
            score=round(float(score), 4),
            content=doc.page_content,
        )
        for i, (doc, score) in enumerate(pairs)
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


def evaluate(strategy: str, top_k: int = TOP_K) -> Measurement:
    """Run all 8 questions search-only against one strategy."""
    measurement = Measurement(strategy=strategy)
    for question in QUESTIONS:
        hits = search(strategy, question.query, top_k=top_k)
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
