"""Book manifest: create a skeleton, load, and validate.

Usage:
    python -m src.manifest init     # create 15 empty entries for the group
    python -m src.manifest check    # list everything that is missing or wrong
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config, read_json, resolve, write_json

MIN_BOOKS = 15

FIELDS = (
    "book_id", "title", "author", "edition", "publisher", "year", "isbn",
    "subject", "source_type", "permission_status", "processing_status",
    "file", "source_url", "page_count", "notes",
)
REQUIRED_FIELDS = (
    "book_id", "title", "author", "edition", "subject",
    "permission_status", "processing_status",
)
SOURCE_TYPES = {"", "pdf_text", "pdf_scanned", "other"}
PROCESSING_STATUSES = {"pending", "processed", "needs_ocr", "failed", "approved_exception"}


def book_id(group_id: str, n: int) -> str:
    return f"{group_id}_BOOK_{n:03d}"


def skeleton(group_id: str, subject: str, count: int = MIN_BOOKS, start: int = 1) -> list[dict]:
    """Empty manifest entries for the group to fill in."""
    books = []
    for n in range(start, start + count):
        entry = {field: "" for field in FIELDS}
        entry.update(
            book_id=book_id(group_id, n),
            subject=subject,
            processing_status="pending",
            page_count=None,
        )
        books.append(entry)
    return books


def load_manifest(path: str | Path) -> list[dict]:
    return read_json(path, default=[])


def validate_manifest(books: list[dict], books_dir: str | Path | None = None,
                      group_id: str | None = None) -> list[str]:
    """Return a list of human-readable problems. An empty list means the manifest passes."""
    problems = []
    usable = [b for b in books if b.get("processing_status") != "failed"]
    if len(usable) < MIN_BOOKS:
        problems.append(
            f"Only {len(usable)} usable books listed; at least {MIN_BOOKS} are required "
            "(failed books must be replaced or approved as exceptions)."
        )

    seen = set()
    for i, b in enumerate(books, start=1):
        bid = b.get("book_id") or f"entry #{i}"
        if bid in seen:
            problems.append(f"{bid}: duplicate book_id.")
        seen.add(bid)
        if group_id and b.get("book_id") and not b["book_id"].startswith(f"{group_id}_"):
            problems.append(f"{bid}: book_id should start with '{group_id}_'.")

        missing = [f for f in REQUIRED_FIELDS if not str(b.get(f) or "").strip()]
        if missing:
            problems.append(f"{bid}: missing {', '.join(missing)}.")

        if b.get("source_type", "") not in SOURCE_TYPES:
            problems.append(f"{bid}: source_type '{b['source_type']}' is not one of {sorted(SOURCE_TYPES - {''})}.")
        if b.get("processing_status") not in PROCESSING_STATUSES:
            problems.append(f"{bid}: processing_status '{b.get('processing_status')}' is not one of {sorted(PROCESSING_STATUSES)}.")

        if b.get("processing_status") != "approved_exception":
            if not b.get("file"):
                problems.append(f"{bid}: no file set (put the PDF in data/books/ and write its file name here).")
            elif books_dir is not None and not (Path(books_dir) / b["file"]).exists():
                problems.append(f"{bid}: file '{b['file']}' not found in {books_dir}.")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "check"])
    parser.add_argument("--count", type=int, default=MIN_BOOKS, help="entries to create with init")
    args = parser.parse_args(argv)

    cfg = load_config()
    path = resolve(cfg["paths"]["manifest"])
    books = load_manifest(path)

    if args.command == "init":
        if books:
            print(f"{path} already has {len(books)} entries; not overwriting.")
            return 1
        write_json(skeleton(cfg["group_id"], cfg["subject"], args.count), path)
        print(f"Created {args.count} empty entries in {path}.")
        return 0

    problems = validate_manifest(books, resolve(cfg["paths"]["books_dir"]), cfg["group_id"])
    statuses = {}
    for b in books:
        statuses[b.get("processing_status")] = statuses.get(b.get("processing_status"), 0) + 1
    print(f"{len(books)} books in manifest; status counts: {statuses}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Manifest OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
