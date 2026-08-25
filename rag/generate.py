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
import re
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI

from .loader import load_document
from .manifest import DOCUMENTS, document_paths
from .retrieve import SEMANTIC, TOP_K, Hit, search

# An alias rather than a pinned version: gemini-2.5-flash returns 404 for keys
# created recently ("no longer available to new users"), even though it is still
# listed by models.list().
GENERATION_MODEL = "gemini-flash-latest"

# Below this top-1 relevance nothing in the corpus is close enough to ground
# an answer. Calibrated against the 8 known-answer questions.
RELEVANCE_FLOOR = 0.30

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from get gets how i if in is it its
    many much of on or our per required s that the their there they this to under
    upon was what when where which who whose will with you your""".split()
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
        grounded = any(
            any(term in h.content.lower() for term in terms) for h in org_hits
        )
        if not grounded:
            others = sorted({h.policy_id for h in hits if h.policy_id != org})
            return (
                True,
                "entity_mismatch",
                f"the question names {org}, whose policy is indexed, but no {org} "
                f"chunk discusses {', '.join(terms[:4])}"
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
            f"{h.content}\n---"
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


def answer(
    strategy: str,
    query: str,
    top_k: int = TOP_K,
    region: str | None = None,
    method: str = SEMANTIC,
) -> Answer:
    """Retrieve, gate, and only then generate."""
    hits = search(strategy, query, top_k=top_k, region=region, method=method)

    # BM25/RRF scores are not semantic relevance values. Keep the existing
    # relevance safety calibration on a semantic probe while using the selected
    # retrieval method as the actual generation context.
    safety_hits = hits
    if method != SEMANTIC and not any(h.semantic_score is not None for h in hits):
        safety_hits = search(strategy, query, top_k=top_k, region=region, method=SEMANTIC)
    refuse, gate, reason = refusal_check(query, safety_hits)
    if refuse:
        return Answer(
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

    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, temperature=0.0)
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context chunks:\n{_context(hits)}\n\nQuestion: {query}"),
        ]
    )
    text = response_text(response.content)

    return Answer(
        query=query,
        text=text,
        is_refusal=False,
        hits=hits,
        citations=_citations(text, hits),
    )
