"""Local embeddings via fastembed. Free, unlimited, offline, no API key.

Why this exists: the Gemini free tier allows 1000 embed requests per day, which
is enough to build the index and run the measurement once, and not enough to
keep testing afterwards. This backend has no quota at all, so interactive work
never stalls.

fastembed runs ONNX rather than PyTorch, so the install is about 190 MB instead
of the ~2 GB a torch-based sentence-transformers setup pulls in. The model
itself is a further 210 MB, downloaded once and cached.

BGE models are asymmetric: a query is embedded with an instruction prefix that a
passage does not get. fastembed exposes that as query_embed and passage_embed,
and this wrapper maps LangChain's embed_query and embed_documents onto them.
Using the wrong one on either side measurably degrades retrieval.
"""

from __future__ import annotations

import functools

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DIMENSIONS = 768


@functools.lru_cache(maxsize=4)
def _model(name: str):
    """Load once per process; construction reads the ONNX weights from disk."""
    from fastembed import TextEmbedding

    return TextEmbedding(name)


class FastEmbedEmbeddings(Embeddings):
    """LangChain embeddings backed by a local ONNX model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    @property
    def model(self):
        return _model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed([text]))).tolist()
