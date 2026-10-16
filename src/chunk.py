"""Stage 2 - Cleaning and chunking.

Splits each page into overlapping passages. Chunks never cross a page
boundary, so every passage cites exactly one book page. Headings are kept in
the text and the most recent heading is stored as the passage's section.

Usage:
    python -m src.chunk
"""
from __future__ import annotations

import re
import sys

from src.config import load_config, read_jsonl, resolve, write_jsonl
from src.manifest import load_manifest

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}
HEADING_KEYWORD = re.compile(r"^(chapter|section|part|appendix|unit)\s+[\dIVXLC]+\b", re.IGNORECASE)
HEADING_NUMBERED = re.compile(r"^\d+(\.\d+){0,3}\.?\s+[A-Z][^.!?]*$")


def clean_text(text: str) -> str:
    """Fix ligatures and hyphenated line breaks; keep paragraph breaks as blank lines."""
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def is_heading(paragraph: str) -> bool:
    if len(paragraph) > 90 or len(paragraph.split()) > 12:
        return False
    return bool(HEADING_KEYWORD.match(paragraph) or HEADING_NUMBERED.match(paragraph))


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over words for paragraphs longer than one chunk."""
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def chunk_page(text: str, size: int, overlap: int, min_words: int) -> list[str]:
    """Pack whole paragraphs into chunks of about `size` words with `overlap` words carried over."""
    chunks: list[str] = []
    current: list[str] = []
    fresh = 0  # words in `current` not yet emitted in any chunk

    for para in text.split("\n\n"):
        words = para.split()
        if len(words) > size:
            if fresh:
                chunks.append(" ".join(current))
            chunks.extend(split_words(para, size, overlap))
            current, fresh = [], 0
            continue
        if fresh and len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            fresh = 0
        current.extend(words)
        fresh += len(words)
    if fresh:
        chunks.append(" ".join(current))

    # Fold a too-short last chunk into the previous one rather than losing it.
    if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"
    return [c for c in chunks if len(c.split()) >= min_words]


def chunk_id(book_id: str, page: int, n: int) -> str:
    return f"{book_id}_P{page:03d}_C{n:02d}"


def _paragraph_key(paragraph: str) -> str:
    """Compare paragraphs ignoring case and numbers, so 'p. 12' and 'p. 13' footers match."""
    return re.sub(r"\d+", "#", paragraph.lower())


def repeated_paragraphs(pages: list[dict], min_share: float = 0.2, min_pages: int = 20) -> dict[str, set[str]]:
    """Per book, paragraphs that appear on at least `min_share` of its pages: copyright notices,
    running headers, page numbers. Left in, they dominate passages and mislead retrieval."""
    counts: dict[str, dict[str, int]] = {}
    page_totals: dict[str, int] = {}
    for page in pages:
        book_id = page["book_id"]
        page_totals[book_id] = page_totals.get(book_id, 0) + 1
        book_counts = counts.setdefault(book_id, {})
        for key in {_paragraph_key(p) for p in clean_text(page["text"]).split("\n\n") if p}:
            book_counts[key] = book_counts.get(key, 0) + 1
    repeated = {}
    for book_id, book_counts in counts.items():
        total = page_totals[book_id]
        if total >= min_pages:
            threshold = max(5, min_share * total)
            repeated[book_id] = {key for key, n in book_counts.items() if n >= threshold}
    return repeated


def build_passages(pages: list[dict], books: dict[str, dict], group_id: str, subject: str,
                   chunk_cfg: dict) -> list[dict]:
    """Turn page records into passage records that carry their full source metadata."""
    passages = []
    section_by_book: dict[str, str] = {}
    boilerplate = repeated_paragraphs(pages, chunk_cfg.get("boilerplate_page_share", 0.2))
    for page in pages:
        book = books[page["book_id"]]
        skip = boilerplate.get(book["book_id"], set())
        text = "\n\n".join(p for p in clean_text(page["text"]).split("\n\n") if _paragraph_key(p) not in skip)
        for para in text.split("\n\n"):
            if is_heading(para):
                section_by_book[book["book_id"]] = para
                break
        chunks = chunk_page(text, chunk_cfg["chunk_size_words"], chunk_cfg["overlap_words"],
                            chunk_cfg["min_chunk_words"])
        for n, chunk_text in enumerate(chunks, start=1):
            passages.append({
                "group_id": group_id,
                "subject": subject,
                "book_id": book["book_id"],
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "edition": book.get("edition", ""),
                "page": page["page"],
                "page_label": page.get("page_label"),
                "section": section_by_book.get(book["book_id"], ""),
                "chunk_id": chunk_id(book["book_id"], page["page"], n),
                "text": chunk_text,
            })
    return passages


def run_chunking(cfg: dict) -> list[dict]:
    paths = cfg["paths"]
    books = {b["book_id"]: b for b in load_manifest(resolve(paths["manifest"]))}
    pages = read_jsonl(resolve(paths["pages"]))
    passages = build_passages(pages, books, cfg["group_id"], cfg["subject"], cfg["chunking"])
    write_jsonl(passages, resolve(paths["passages"]))
    return passages


def main() -> int:
    passages = run_chunking(load_config())
    books = len({p["book_id"] for p in passages})
    print(f"Wrote {len(passages)} passages from {books} book(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
