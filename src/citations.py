"""Stage 7 - Citation handling: make sure every cited source is a real stored passage."""
from __future__ import annotations

import re

MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def to_source(passage: dict) -> dict:
    return {"book_id": passage["book_id"], "page": passage["page"], "chunk_id": passage["chunk_id"]}


def resolve_sources(sources: list[dict], by_id: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split sources into (valid, invalid). A source is valid only if its chunk_id exists
    and its book_id and page match the stored passage."""
    valid, invalid = [], []
    for s in sources:
        p = by_id.get(s.get("chunk_id"))
        if p and p["book_id"] == s.get("book_id") and p["page"] == s.get("page"):
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid


def renumber_markers(text: str, evidence_count: int) -> tuple[str, list[int]]:
    """Rewrite [n] markers so they number the cited passages 1, 2, 3... in order of first use.

    Returns the new text and the 0-based evidence indexes that were cited. Markers that
    point outside the evidence list are fabricated and are removed.
    """
    order: list[int] = []

    def replace(match: re.Match) -> str:
        kept = []
        for part in match.group(1).split(","):
            n = int(part) - 1
            if 0 <= n < evidence_count:
                if n not in order:
                    order.append(n)
                kept.append(str(order.index(n) + 1))
        return f"[{', '.join(kept)}]" if kept else ""

    new_text = MARKER.sub(replace, text)
    return re.sub(r"\s+([.,;:])", r"\1", new_text).strip(), order


def format_citation(passage: dict) -> str:
    """Human-readable citation, e.g. 'Computer Organization (5th ed.), p. 143'."""
    title = passage.get("title") or passage["book_id"]
    edition = f" ({passage['edition']})" if passage.get("edition") else ""
    label = passage.get("page_label")
    page = f"p. {label}" if label and str(label) != str(passage["page"]) else f"p. {passage['page']}"
    if label and str(label) != str(passage["page"]):
        page += f" (PDF page {passage['page']})"
    return f"{title}{edition}, {page}"
