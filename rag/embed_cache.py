"""A disk cache in front of an embedding model.

The Gemini free tier allows 1000 embed requests per DAY
(EmbedContentRequestsPerDayPerUserPerProjectPerModel). Indexing this corpus
costs a few dozen, but interactive testing costs one per question and it adds up
fast -- re-asking the same question should not spend quota twice.

LangChain 1.x dropped CacheBackedEmbeddings and LocalFileStore, so this is a
small equivalent: sqlite, keyed by a hash of model + task type + text, storing
float32 vectors. The key includes the task type because Gemini embeds the same
string differently for RETRIEVAL_DOCUMENT than for RETRIEVAL_QUERY, and mixing
those would silently degrade retrieval.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

CACHE_PATH = Path(__file__).resolve().parent.parent / "chroma" / "embed_cache.sqlite"


class DiskCachedEmbeddings(Embeddings):
    """Wraps an embedder, persisting every vector it computes."""

    def __init__(self, inner: Embeddings, namespace: str, path: Path = CACHE_PATH):
        self.inner = inner
        self.namespace = namespace
        self.path = path
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.namespace}\x00{text}".encode("utf-8")).hexdigest()
        return digest

    def _read(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        with self._lock, self._connect() as conn:
            marks = ",".join("?" * len(keys))
            rows = conn.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({marks})", keys
            ).fetchall()
        return {k: np.frombuffer(v, dtype=np.float32).tolist() for k, v in rows}

    def _write(self, pairs: list[tuple[str, list[float]]]) -> None:
        if not pairs:
            return
        blobs = [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in pairs]
        with self._lock, self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", blobs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = self._read(list(dict.fromkeys(keys)))

        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in cached]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)

        if missing:
            fresh = self.inner.embed_documents([t for _, t in missing])
            new_pairs = [(keys[i], vec) for (i, _), vec in zip(missing, fresh)]
            self._write(new_pairs)
            cached.update(dict(new_pairs))

        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        found = self._read([key])
        if key in found:
            self.hits += 1
            return found[key]
        self.misses += 1
        vec = self.inner.embed_query(text)
        self._write([(key, vec)])
        return vec


def cache_stats(path: Path = CACHE_PATH) -> dict:
    """How many vectors are cached, for reporting quota saved."""
    if not path.exists():
        return {"cached_vectors": 0}
    with sqlite3.connect(path) as conn:
        try:
            (count,) = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        except sqlite3.OperationalError:
            return {"cached_vectors": 0}
    return {"cached_vectors": count, "size_kb": round(path.stat().st_size / 1024)}
