"""Cited generation, with refusal decided in code rather than by the model.

The task is explicit that a grounding prompt saying 'refuse if the context is
insufficient' is not good enough: an invented leave entitlement quoted back to
an employee is a legal problem. So the model never sees an out-of-corpus
question. The gate runs first and short-circuits.

A relevance floor alone is not sufficient here, and this is the interesting
part. Two of our three out-of-corpus questions retrieve confident, topically
adjacent chunks:

  'What is Apple's parental leave entitlement?' -- Apple's Business Conduct
  Policy is in the corpus and scores well on 'Apple', 'employee', 'policy'.
  'parental leave' also exists in the corpus, but only in the Michigan faculty
  handbook, a different employer.

  'What is the gratuity payout formula on resignation?' -- Soft Suave's
  separation policy is retrieved correctly and looks exactly right, but
  'gratuity' appears nowhere in any of the six documents.

So there are three independent gates: a relevance floor, a corpus-vocabulary
check that catches a topic the corpus never discusses, and an entity check
that catches a topic the corpus discusses only for a different organisation.
"""

from __future__ import annotations

import functools
import os
import re
import time
from dataclasses import dataclass, field

from .loader import load_document
from .manifest import DOCUMENTS, document_paths
from .retrieve import RERANK_OFF, SEMANTIC, TOP_K, Hit, search
from .tracing import write_trace

# Groq is used only after local retrieval, gating, and optional local reranking.
# Set GROQ_GENERATION_MODEL in .env to use another model your account exposes.
GENERATION_MODEL = os.environ.get("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")

# Below this top-1 relevance nothing in the corpus is close enough to ground
# an answer. Calibrated against the 8 known-answer questions.
RELEVANCE_FLOOR = 0.30

STOPWORDS = frozenset(
    """a an and are as at about be by can describe do does explain for from get gets
    give how i if in is it its kind kinda me please provide show summarize tell
    listed happens above below many much of on or our per required s that the
    their there they this to under upon was what when where which who whose will
    with you your""".split()
)

# Organisation names a question may name, mapped to the policy that speaks for
# them. Used to catch a question about an employer whose policy is present but
# silent on the topic asked.
ORGANISATIONS = {
    "apple": "APPL-BCP",
    "google": "GOOG-COC",
    "alphabet": "GOOG-COC",
    "soft suave": "SS-HB-2025",
    "softsuave": "SS-HB-2025",
    "arizona": "USM-3-113",
    "michigan": "UM-FH-16",
    "umich": "UM-FH-16",
    "research foundation": "RF-LEAVE",
    "suny": "RF-LEAVE",
}


@dataclass
class Citation:
    chunk_id: str
    policy_id: str
    section: str
    resolves: bool


@dataclass
class Answer:
    query: str
    text: str
    is_refusal: bool
    refusal_reason: str = ""
    gate: str = ""
    hits: list[Hit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    trace_id: str = ""
    metrics: dict = field(default_factory=dict)


@functools.lru_cache(maxsize=1)
def corpus_vocabulary() -> frozenset[str]:
    """Every word the corpus contains, for detecting topics it never covers."""
    words: set[str] = set()
    for path in document_paths():
        words.update(re.findall(r"[a-z]{3,}", load_document(path).lower()))
    return frozenset(words)


def salient_terms(query: str) -> list[str]:
    """Content words from the query, excluding organisation names."""
    text = query.lower()
    for org in ORGANISATIONS:
        text = text.replace(org, " ")
    return [w for w in re.findall(r"[a-z]{4,}", text) if w not in STOPWORDS]


def named_organisation(query: str) -> str | None:
    """The policy_id of an organisation the query names, if any."""
    lowered = query.lower()
    for name, policy_id in ORGANISATIONS.items():
        if name in lowered:
            return policy_id
    return None


def refusal_check(query: str, hits: list[Hit]) -> tuple[bool, str, str]:
    """Decide refusal before the model is involved. -> (refuse, gate, reason)"""
    if not hits:
        return True, "no_results", "retrieval returned nothing"

    # RRF and BM25 scores are rank/term-frequency scores, so they cannot be
    # compared with the calibrated 0--1 semantic relevance floor. Hybrid
    # retrieval retains its semantic candidates; use their strongest relevance
    # score for this safety gate rather than accidentally treating 0.03 RRF as
    # evidence of irrelevance.
    semantic_scores = [h.semantic_score for h in hits if h.semantic_score is not None]
    top = max(semantic_scores) if semantic_scores else hits[0].score
    if top < RELEVANCE_FLOOR:
        return (
            True,
            "relevance_floor",
            f"top-1 relevance {top:.3f} is below the {RELEVANCE_FLOOR} floor",
        )

    terms = salient_terms(query)
    vocabulary = corpus_vocabulary()
    unknown = [t for t in terms if t not in vocabulary]
    if unknown:
        return (
            True,
            "absent_from_corpus",
            f"no document contains {', '.join(sorted(set(unknown)))}",
        )

    org = named_organisation(query)
    if org:
        org_hits = [h for h in hits if h.policy_id == org]
        # Do not require the policy to repeat the user's exact wording.  For
        # example, Soft Suave calls carry-forward "carried over" and puts the
        # technical-team alternative under "leave encashment".  The semantic
        # retrieval score is the appropriate evidence test here: require a
        # sufficiently relevant chunk from the named organisation rather than
        # a literal overlap of query and document tokens.
        org_relevance = max(
            (h.semantic_score if h.semantic_score is not None else h.score for h in org_hits),
            default=0.0,
        )
        if org_relevance < RELEVANCE_FLOOR:
            others = sorted({h.policy_id for h in hits if h.policy_id != org})
            return (
                True,
                "entity_mismatch",
                f"the question names {org}, whose policy is indexed, but no {org} "
                f"chunk reached the semantic relevance floor ({org_relevance:.3f} < {RELEVANCE_FLOOR:.3f})"
                + (f"; that topic appears only in {', '.join(others)}" if others else ""),
            )

    return False, "", ""


SYSTEM_PROMPT = """You answer questions about HR leave policy using ONLY the \
numbered context chunks supplied.

Rules:
1. Every factual claim must carry a citation in the form [CITE: chunk_id],
   copied exactly from the chunk that supports it.
2. Quote figures exactly as the policy states them. Do not convert units,
   round, or generalise across regions.
3. If a claim is not supported by a supplied chunk, do not make the claim.
4. Different regions have different entitlements. Never apply one region's
   figure to another region.
"""


def _context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[CHUNK_ID: {h.chunk_id}]\n"
            f"policy_id: {h.policy_id} | section: {h.section or '-'} | "
            f"region: {h.region} | source_file: {h.source_file}\n"
            f"{h.context_content or h.content}\n---"
        )
    return "\n".join(blocks)


def _citations(text: str, hits: list[Hit]) -> list[Citation]:
    """Parse and validate citations, dropping any chunk_id we did not supply."""
    by_id = {h.chunk_id: h for h in hits}
    found: list[Citation] = []
    for raw in re.findall(r"\[CITE:\s*([^\]]+)\]", text):
        for chunk_id in (item.strip() for item in raw.split(",")):
            if not chunk_id:
                continue
            hit = by_id.get(chunk_id)
            found.append(
                Citation(
                    chunk_id=chunk_id,
                    policy_id=hit.policy_id if hit else "UNRESOLVED",
                    section=hit.section if hit else "",
                    resolves=hit is not None,
                )
            )
    return found


def response_text(content) -> str:
    """Flatten a chat response into plain text.

    Gemini 3 models return content as a list of parts rather than a string, and
    each part carries a long opaque 'signature' field. Stringifying the list
    dumps that blob into the answer and breaks citation parsing, so pull out the
    text parts and join them.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, dict) and "text" in part:
                pieces.append(str(part["text"]))
        return "\n".join(p for p in pieces if p).strip()
    return str(content)


def _stream_groq(messages: list[dict[str, str]]) -> tuple[str, float | None, dict[str, int | None]]:
    """Stream one Groq completion, measuring visible TTFT and provider usage."""
    from groq import Groq

    started = time.perf_counter()
    first_token_ms = None
    pieces: list[str] = []
    usage: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
    }
    client = Groq()
    request = {
        "model": GENERATION_MODEL,
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }
    try:
        # Newer Groq SDKs accept this option and return usage on the final chunk.
        stream = client.chat.completions.create(**request, stream_options={"include_usage": True})
    except TypeError as exc:
        # Older SDK releases reject stream_options before making a network request.
        if "stream_options" not in str(exc):
            raise
        stream = client.chat.completions.create(**request)
    for chunk in stream:
        data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        provider_usage = data.get("usage") if isinstance(data, dict) else None
        if provider_usage:
            prompt_details = provider_usage.get("prompt_tokens_details") or {}
            completion_details = provider_usage.get("completion_tokens_details") or {}
            usage = {
                "input_tokens": provider_usage.get("prompt_tokens"),
                "output_tokens": provider_usage.get("completion_tokens"),
                "cached_input_tokens": prompt_details.get("cached_tokens"),
                "reasoning_tokens": completion_details.get("reasoning_tokens"),
            }
        choices = data.get("choices", []) if isinstance(data, dict) else []
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            visible = response_text(content) if content is not None else ""
            if visible:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                pieces.append(visible)
    return "".join(pieces).strip(), first_token_ms, usage


def _estimated_cost(usage: dict[str, int | None], rates: dict[str, float | None] | None) -> float | None:
    """Estimate request cost only when usage and all applicable rates are known."""
    if not rates:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    input_rate = rates.get("input_per_million")
    output_rate = rates.get("output_per_million")
    if input_tokens is None or output_tokens is None or input_rate is None or output_rate is None:
        return None
    cached_tokens = usage.get("cached_input_tokens") or 0
    cached_rate = rates.get("cached_input_per_million")
    if cached_tokens and cached_rate is None:
        return None
    uncached_tokens = max(0, input_tokens - cached_tokens)
    amount = uncached_tokens * input_rate + output_tokens * output_rate
    if cached_tokens:
        amount += cached_tokens * cached_rate
    return amount / 1_000_000


def answer(
    strategy: str,
    query: str,
    top_k: int = TOP_K,
    region: str | None = None,
    method: str = SEMANTIC,
    rerank: str = RERANK_OFF,
    rates: dict[str, float | None] | None = None,
) -> Answer:
    """Retrieve, gate, generate, and write a timed, privacy-aware request trace."""
    started = time.perf_counter()
    timings: dict[str, float | None] = {
        "retrieval": None, "safety_check": None, "first_token": None,
        "generation": None, "total": None,
    }
    hits: list[Hit] = []
    stage = "retrieval"
    try:
        retrieval_started = time.perf_counter()
        hits = search(strategy, query, top_k=top_k, region=region, method=method, rerank=rerank)

        # BM25/RRF scores are not semantic relevance values. Calibrate refusal
        # with a semantic probe when the selected search did not retain scores.
        safety_hits = hits
        if method != SEMANTIC and not any(h.semantic_score is not None for h in hits):
            safety_hits = search(strategy, query, top_k=top_k, region=region, method=SEMANTIC, rerank=rerank)
        timings["retrieval"] = (time.perf_counter() - retrieval_started) * 1000

        stage = "safety_check"
        safety_started = time.perf_counter()
        refuse, gate, reason = refusal_check(query, safety_hits)
        timings["safety_check"] = (time.perf_counter() - safety_started) * 1000
        if refuse:
            result = Answer(
                query=query,
                text=(
                    "I cannot answer this from the indexed policy documents.\n"
                    f"Reason: {reason}."
                ),
                is_refusal=True,
                refusal_reason=reason,
                gate=gate,
                hits=hits,
            )
            timings["total"] = (time.perf_counter() - started) * 1000
            usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
            result.metrics = {**timings, **usage, "estimated_cost_usd": _estimated_cost(usage, rates)}
            result.trace_id = write_trace(
                query=query, hits=hits, output=result.text, refused=True, gate=gate,
                model=GENERATION_MODEL, temperature=0.0, method=method,
                strategy=strategy, rerank=rerank, region=region, top_k=top_k,
                timings_ms=timings, usage=usage, estimated_cost=result.metrics["estimated_cost_usd"],
                status="refused",
            )
            return result

        stage = "generation"
        generation_started = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context chunks:\n{_context(hits)}\n\nQuestion: {query}"},
        ]
        text, first_token_ms, usage = _stream_groq(messages)
        timings["first_token"] = first_token_ms
        timings["generation"] = (time.perf_counter() - generation_started) * 1000
        timings["total"] = (time.perf_counter() - started) * 1000

        result = Answer(
            query=query,
            text=text,
            is_refusal=False,
            hits=hits,
            citations=_citations(text, hits),
        )
        result.metrics = {**timings, **usage, "estimated_cost_usd": _estimated_cost(usage, rates)}
        result.trace_id = write_trace(
            query=query, hits=hits, output=result.text, refused=False,
            model=GENERATION_MODEL, temperature=0.0, method=method,
            strategy=strategy, rerank=rerank, region=region, top_k=top_k,
            timings_ms=timings, usage=usage, estimated_cost=result.metrics["estimated_cost_usd"],
            status="success",
        )
        return result
    except Exception as exc:
        timings["total"] = (time.perf_counter() - started) * 1000
        try:
            trace_id = write_trace(
                query=query, hits=hits, output="", refused=False, gate=stage,
                model=GENERATION_MODEL, temperature=0.0, method=method,
                strategy=strategy, rerank=rerank, region=region, top_k=top_k,
                timings_ms=timings,
                usage={"input_tokens": None, "output_tokens": None, "cached_input_tokens": None, "reasoning_tokens": None},
                status="error", error=f"{type(exc).__name__}: {exc}",
            )
            setattr(exc, "trace_id", trace_id)
        except Exception:
            pass
        raise
