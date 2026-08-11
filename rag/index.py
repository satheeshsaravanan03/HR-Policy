"""Chroma indexes, one collection per chunking strategy.

Both collections use the SAME embedding model. Changing the chunker and the
embedding model together would make any movement in hit-in-top-5
unattributable, which the task calls out explicitly as a way to learn nothing.

Two details that are easy to get wrong and expensive to debug:

  Asymmetric embedding. Gemini distinguishes RETRIEVAL_DOCUMENT from
  RETRIEVAL_QUERY. Chroma calls embed_documents when adding and embed_query
  when searching, so we hand it a differently-configured embedder for each
  role rather than one shared object.

  Chroma returns DISTANCE, not similarity -- lower is better. A refusal gate
  written as 'score < threshold -> refuse' therefore inverts and refuses every
  answerable question. Retrieval here goes through
  similarity_search_with_relevance_scores, which normalises to 0-1 where
  higher is better, and the gate is written against that.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .chunkers import RECURSIVE, STRATEGIES, STRUCTURE
from .embed_cache import DiskCachedEmbeddings
from .local_embed import FastEmbedEmbeddings
from .manifest import CORPUS_DIR, DOCUMENTS

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma"

# The library already packs up to 100 texts or 20k estimated tokens into one
# API request, so a batch this size costs roughly one request.
EMBED_BATCH = 40

# Pause between batches. The free tier's embed-request quota is per minute and
# tighter in practice than the documented figure, and an unpaced run exhausts
# it partway through the corpus.
PACE_SECONDS = 6.0

# What the 429 itself asks for. Backing off from one second cannot clear a
# per-minute window, so a quota error waits out the whole window.
QUOTA_WAIT_SECONDS = 65.0


def _require_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY missing; put it in HR-policy/.env")


GEMINI = "gemini"
LOCAL = "local"

# Default to the local backend because it has no quota, so nothing stalls
# mid-session. The graded numbers in results.md were produced with the Gemini
# backend; set EMBED_BACKEND=gemini to reproduce them.
BACKEND = os.environ.get("EMBED_BACKEND", LOCAL).strip().lower()


def collection_name(strategy: str, backend: str | None = None) -> str:
    """Separate collections per backend.

    The two backends produce different dimensionalities (3072 vs 768) and
    incompatible vector spaces, so they must never share a collection. Keeping
    the Gemini names unsuffixed leaves the graded index untouched.
    """
    backend = backend or BACKEND
    return strategy if backend == GEMINI else f"{strategy}_{backend}"


def embeddings(task_type: str, backend: str | None = None) -> Embeddings:
    """An embedder configured for one role: indexing or querying."""
    backend = backend or BACKEND

    if backend == LOCAL:
        # No network, no quota; a disk cache would only add overhead.
        return FastEmbedEmbeddings()

    if backend != GEMINI:
        raise ValueError(f"unknown EMBED_BACKEND {backend!r}; use 'local' or 'gemini'")

    _require_key()
    inner = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, task_type=task_type)
    # Cached because the free tier allows only 1000 embed requests per day and
    # interactive testing re-asks the same questions.
    return DiskCachedEmbeddings(inner, namespace=f"{EMBEDDING_MODEL}:{task_type}")


def _open(strategy: str, task_type: str, backend: str | None = None) -> Chroma:
    return Chroma(
        collection_name=collection_name(strategy, backend),
        embedding_function=embeddings(task_type, backend),
        persist_directory=str(CHROMA_DIR),
        # Cosine, so relevance scores stay interpretable across collections.
        collection_metadata={"hnsw:space": "cosine"},
    )


def writer(strategy: str) -> Chroma:
    return _open(strategy, "RETRIEVAL_DOCUMENT")


def reader(strategy: str) -> Chroma:
    return _open(strategy, "RETRIEVAL_QUERY")


def already_indexed(strategy: str) -> set[str]:
    """source_file values with at least one chunk in this collection.

    Requirement 6: index the new documents only. This is what makes an
    incremental add possible instead of a full rebuild. Note this is
    file-level and therefore only safe to act on together with
    existing_chunk_ids below -- a run interrupted by a rate limit leaves a
    file partially present, and skipping on filename alone would strand the
    remaining chunks while reporting success.
    """
    store = reader(strategy)
    existing = store.get(include=["metadatas"])
    return {m.get("source_file", "") for m in existing["metadatas"]}


def existing_chunk_ids(strategy: str) -> set[str]:
    """Every chunk_id already stored, so an interrupted run can resume."""
    store = reader(strategy)
    got = store.get()
    return set(got.get("ids") or [])


def chunks_for(strategy: str, source_files: list[str]) -> list[Document]:
    chunk = STRATEGIES[strategy]
    out: list[Document] = []
    for name in source_files:
        out.extend(chunk(CORPUS_DIR / name))
    return out


def add(strategy: str, source_files: list[str], verbose: bool = True) -> int:
    """Chunk, embed and store the named documents. Returns chunks added.

    Resumes at chunk granularity and paces itself. The free tier's embed
    request quota is tight enough that a cold run of this corpus trips it, and
    the server asks for a ~60s wait when it does, so exponential backoff from
    one second never recovers.
    """
    docs = chunks_for(strategy, source_files)
    done = existing_chunk_ids(strategy)
    docs = [d for d in docs if d.metadata["chunk_id"] not in done]
    if not docs:
        if verbose:
            print("    nothing new to embed (all chunk_ids already stored)")
        return 0
    store = writer(strategy)

    added = 0
    for start in range(0, len(docs), EMBED_BATCH):
        batch = docs[start : start + EMBED_BATCH]
        for attempt in range(6):
            try:
                store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
                added += len(batch)
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                # 429 carries a retryDelay of ~60s; anything shorter just
                # burns another request against the same exhausted window.
                wait = QUOTA_WAIT_SECONDS if "RESOURCE_EXHAUSTED" in str(exc) else 2**attempt
                if verbose:
                    print(f"    quota hit, waiting {wait}s then retrying batch")
                time.sleep(wait)
        if verbose:
            print(f"    embedded {added}/{len(docs)}")
        # Pacing exists only for the Gemini quota; local embedding has none.
        if BACKEND == GEMINI:
            time.sleep(PACE_SECONDS)
    return added


def ingest(strategy: str, source_files: list[str] | None = None) -> dict:
    """Add only what is missing, at chunk granularity.

    Passing source_files=None means 'every manifest document', which combined
    with the chunk-level skip is the incremental path requirement 6 asks for.
    """
    present = already_indexed(strategy)
    wanted = source_files or [d.source_file for d in DOCUMENTS]
    fresh = [f for f in wanted if f not in present]
    seen = [f for f in wanted if f in present]

    print(f"  [{strategy}] files already present: {len(seen)} | new files: {len(fresh)}")
    for f in seen:
        print(f"    already indexed, not re-embedded: {f}")

    added = add(strategy, wanted)
    return {
        "strategy": strategy,
        "added_chunks": added,
        "new_files": fresh,
        "skipped_files": seen,
    }


def collection_stats(strategy: str) -> dict:
    store = reader(strategy)
    got = store.get(include=["metadatas"])
    metas = got["metadatas"]
    missing_source = sum(1 for m in metas if not m.get("source_file"))
    return {
        "chunks": len(metas),
        "documents": len({m.get("source_file") for m in metas}),
        "chunks_missing_source_file": missing_source,
    }


STRATEGY_NAMES = (RECURSIVE, STRUCTURE)
