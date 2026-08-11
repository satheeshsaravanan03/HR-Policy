"""The two chunking strategies the task compares.

Both consume the same markdown from rag.loader and the same metadata from
rag.manifest, and both are embedded with the same model. Only the split
differs, so any movement in hit-in-top-5 is attributable to the chunker.

Strategy A, 'recursive', is the ordinary baseline: fixed-size windows with
overlap, blind to document structure.

Strategy B, 'structure', splits on policy headers so a clause is never
separated from the section number that gives it authority. Two details do
the real work:

  strip_headers=False -- LangChain strips header text from the chunk body by
  default, which would delete '9.3. LEAVES ENCASHMENT' from the very chunk
  that must retain it, defeating the exercise while appearing to work.

  Oversized sections are sub-split with the heading re-prepended to every
  piece. Without that, a long section's tail chunks lose the section number
  just as thoroughly as the baseline loses it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .loader import load_document, split_section_id
from .manifest import base_metadata

RECURSIVE = "recursive"
STRUCTURE = "structure"

# Windows for the baseline. 1000/150 is the ordinary starting point and is
# deliberately left unturned: tuning it would confound the comparison.
BASELINE_CHUNK_CHARS = 1000
BASELINE_OVERLAP_CHARS = 150

# A section longer than this is sub-split rather than embedded whole, since an
# oversized chunk dilutes its own embedding and buries the clause.
MAX_SECTION_CHARS = 1800
SECTION_OVERLAP_CHARS = 120

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _chunk_id(policy_id: str, strategy: str, ordinal: int, text: str) -> str:
    """Deterministic, collision-free, and resolvable back to one chunk.

    The ordinal alone would collide across runs if chunk order shifted, and a
    header-derived id (as the reference guide builds) collides whenever a
    document repeats a heading, silently overwriting chunks in the store.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{policy_id}#{strategy}-{ordinal:04d}-{digest}"


def _finalise(docs: list[Document], strategy: str, source_file: str) -> list[Document]:
    """Attach corpus metadata plus a unique chunk_id to every chunk."""
    base = base_metadata(source_file)
    out: list[Document] = []
    for ordinal, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue
        meta = {**base, **doc.metadata, "strategy": strategy}
        meta["chunk_id"] = _chunk_id(base["policy_id"], strategy, ordinal, text)
        meta["chunk_chars"] = len(text)
        # Requirement 1: a chunk with no source_file is a failed ingest.
        assert meta.get("source_file"), f"chunk {meta['chunk_id']} lost source_file"
        out.append(Document(page_content=text, metadata=meta))
    return out


def chunk_recursive(path: Path) -> list[Document]:
    """Strategy A: fixed-size windows, structure-blind."""
    text = load_document(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_CHUNK_CHARS,
        chunk_overlap=BASELINE_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _finalise(splitter.create_documents([text]), RECURSIVE, path.name)


def _heading_of(meta: dict) -> str:
    """The deepest heading LangChain recorded for this chunk."""
    for key in ("h3", "h2", "h1"):
        if meta.get(key):
            return str(meta[key])
    return ""


def chunk_structure_aware(path: Path) -> list[Document]:
    """Strategy B: split on policy headers, keeping clause and section joined."""
    text = load_document(path)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)

    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SECTION_CHARS,
        chunk_overlap=SECTION_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pieces: list[Document] = []
    for section in sections:
        heading = _heading_of(section.metadata)
        section_id, title = split_section_id(heading)
        meta = {
            "section": section_id or "",
            "section_title": title,
            "heading_path": " > ".join(
                str(section.metadata[k]) for k in ("h1", "h2", "h3") if section.metadata.get(k)
            ),
        }

        if len(section.page_content) <= MAX_SECTION_CHARS:
            pieces.append(Document(page_content=section.page_content, metadata=meta))
            continue

        # Re-prepend the heading so every piece keeps its section number.
        for part in body_splitter.split_text(section.page_content):
            body = part if part.lstrip().startswith(heading) else f"{heading}\n\n{part}"
            pieces.append(Document(page_content=body, metadata=dict(meta)))

    return _finalise(pieces, STRUCTURE, path.name)


STRATEGIES = {
    RECURSIVE: chunk_recursive,
    STRUCTURE: chunk_structure_aware,
}
