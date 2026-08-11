"""Load corpus documents into markdown with headings made explicit.

The structure-aware chunker downstream relies on LangChain's
MarkdownHeaderTextSplitter, which needs '#' markers. PDF text extraction
produces none, so this module reconstructs them.

Headings are found by font, not by regex. Measuring the corpus showed body
text sits at one size in regular weight while every heading is bold at or
above that size -- true across all three PDF families we carry, including
RF SUNY whose headings have no section numbers at all and which a
numbering regex therefore cannot see.

Extraction uses sort=True. Without it PyMuPDF returns Word-generated
headings in text-frame order rather than reading order, which hoists every
heading to the top of the page and strands each clause from the section
number that gives it authority -- the exact failure this task measures.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

# A heading line never runs long; anything past this is a bolded sentence.
MAX_HEADING_CHARS = 110

# Table-of-contents rows: dot leaders, or a bare trailing page number.
TOC_DOT_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")
TOC_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")

# Section identifiers we want to keep attached: 9.2, 16.B.1, 3-113.5, 7.
SECTION_ID = re.compile(r"^\s*(\d{1,2}(?:[.\-][A-Za-z0-9]+)*)\.?\s+(.*)$")

# A heading that is nothing but a section number, its title typeset elsewhere.
NUMBER_ONLY = re.compile(r"\d{1,2}(?:[.\-][A-Za-z0-9]+)*\.?")

BOLD_FLAG = 1 << 4


@dataclass
class Heading:
    """A heading recovered from the document, with its position in the text."""

    level: int
    section_id: str | None
    title: str


def _spans(page: fitz.Page):
    """Yield (text, size, is_bold) for every non-empty span on the page."""
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    yield span["text"], round(span["size"], 1), bool(span["flags"] & BOLD_FLAG)


def _body_size(doc: fitz.Document) -> float:
    """The size carrying the most regular-weight text: the body size."""
    weights: collections.Counter = collections.Counter()
    for page in doc:
        for text, size, is_bold in _spans(page):
            if not is_bold:
                weights[size] += len(text)
    return weights.most_common(1)[0][0] if weights else 11.0


def _is_heading_style(size: float, is_bold: bool, body: float) -> bool:
    """Whether this span's styling marks a heading rather than body text.

    Two signals, because the corpus uses both: Soft Suave, RF SUNY and UMich
    bold their headings, while Apple leaves them regular weight and relies on
    size alone (14.0 headings over a 9.0 body).
    """
    return size > body or (is_bold and size >= body)


def _heading_sizes(doc: fitz.Document, body: float) -> list[float]:
    """Heading-styled sizes, largest first, mapped to levels by that order."""
    found = {
        size
        for page in doc
        for text, size, is_bold in _spans(page)
        if _is_heading_style(size, is_bold, body)
    }
    return sorted(found, reverse=True)


def _is_toc_page(page: fitz.Page) -> bool:
    """True for contents pages, whose entries would become phantom headings.

    The phrase test demands a standalone line. RF SUNY prints a running
    'Return to Table of Contents' footer on all 43 pages, so a substring
    test discards the entire document.
    """
    text = page.get_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(re.fullmatch(r"(TABLE OF )?CONTENTS", ln.upper()) for ln in lines):
        return True
    if len(lines) < 8:
        return False
    # A contents page is mostly bare page numbers or dot-leader rows.
    marks = sum(
        1 for ln in lines if TOC_PAGE_NUMBER.match(ln) or TOC_DOT_LEADER.search(ln)
    )
    return marks / len(lines) > 0.35


def _lines(page: fitz.Page):
    """Group spans into lines, reporting whether the whole line is bold."""
    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            all_bold = all(bool(s["flags"] & BOLD_FLAG) for s in spans)
            size = round(max(s["size"] for s in spans), 1)
            yield text, size, all_bold


def split_section_id(title: str) -> tuple[str | None, str]:
    """Separate a leading section identifier from the heading text.

    '9.2.  ANNUAL LEAVES' -> ('9.2', 'ANNUAL LEAVES')
    '16.B.1 Sabbatical Leaves' -> ('16.B.1', 'Sabbatical Leaves')
    'Eligibility' -> (None, 'Eligibility')
    """
    match = SECTION_ID.match(title)
    if not match:
        return None, title.strip()
    section_id, rest = match.group(1), match.group(2).strip()
    # A number with no text after it is a page number, not a section.
    if not rest:
        return None, title.strip()
    return section_id.rstrip("."), rest


def _merge_orphan_ids(items: list[tuple[bool, int, str]]) -> list[tuple[bool, int, str]]:
    """Rejoin a section number that was typeset apart from its own title.

    Soft Suave's handbook puts '8.7.' and 'EXIT INTERVIEW' in separate text
    frames, so they arrive as two headings and the clause loses the section
    number that gives it authority. Left alone, only 24 of 157 headings keep
    an identifier.
    """
    merged: list[tuple[bool, int, str]] = []
    pending: tuple[bool, int, str] | None = None

    for item in items:
        is_heading, level, text = item
        if pending is not None:
            # The next heading supplies the title for the orphaned number.
            if is_heading:
                merged.append((True, min(pending[1], level), f"{pending[2]} {text}"))
            else:
                merged.append(pending)
                merged.append(item)
            pending = None
            continue
        if is_heading and NUMBER_ONLY.fullmatch(text):
            pending = item
            continue
        merged.append(item)

    if pending is not None:
        merged.append(pending)
    return merged


def pdf_to_markdown(path: Path) -> str:
    """Extract a PDF as markdown, headings promoted to '#' lines."""
    doc = fitz.open(path)
    body = _body_size(doc)
    levels = {size: i + 1 for i, size in enumerate(_heading_sizes(doc, body))}

    items: list[tuple[bool, int, str]] = []
    for page in doc:
        if _is_toc_page(page):
            continue
        for text, size, all_bold in _lines(page):
            is_heading = (
                _is_heading_style(size, all_bold, body)
                and len(text) <= MAX_HEADING_CHARS
                and not TOC_PAGE_NUMBER.match(text)
            )
            items.append((is_heading, levels.get(size, 3), text))
    doc.close()

    out: list[str] = []
    for is_heading, level, text in _merge_orphan_ids(items):
        if is_heading:
            # Cap at h3 so MarkdownHeaderTextSplitter sees a stable set.
            out.extend(["", f"{'#' * min(level, 3)} {text}", ""])
        else:
            out.append(text)
    return "\n".join(out)


def load_document(path: Path) -> str:
    """Load one corpus document as markdown, whatever its source format."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8")
