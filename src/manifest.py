"""Book manifest: create a skeleton, load, validate, and write the book record.

Usage:
    python -m src.manifest init     # create 15 empty entries for the group
    python -m src.manifest check    # list everything that is missing or wrong
    python -m src.manifest record   # write docs/book_record.md and docs/book_record.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

from src.config import load_config, read_json, read_jsonl, resolve, write_json

MIN_BOOKS = 15

FIELDS = (
    "book_id", "title", "author", "edition", "publisher", "year", "isbn",
    "subject", "source_type", "permission_status", "processing_status",
    "coordinator_approval", "file", "source_url", "page_count", "notes",
)
REQUIRED_FIELDS = (
    "book_id", "title", "author", "edition", "subject",
    "permission_status", "processing_status",
)
SOURCE_TYPES = {"", "pdf_text", "pdf_scanned", "other"}
PROCESSING_STATUSES = {"pending", "processed", "needs_ocr", "failed", "approved_exception"}
# The coordinator approves the book list before development (task 1); record the outcome here.
APPROVAL_STATUSES = {"pending", "approved", "rejected"}
RECORD_MD = "docs/book_record.md"
RECORD_CSV = "docs/book_record.csv"


def permission_confirmed(book: dict) -> bool:
    status = str(book.get("permission_status") or "")
    return bool(status.strip()) and not status.upper().startswith("NOT CONFIRMED")


def identifier(book: dict) -> str:
    """ISBN, or a stable identifier (the source URL) when the book has no ISBN."""
    return str(book.get("isbn") or "").strip() or str(book.get("source_url") or "").strip()


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
            coordinator_approval="pending",
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
        if b.get("coordinator_approval", "pending") not in APPROVAL_STATUSES:
            problems.append(f"{bid}: coordinator_approval '{b['coordinator_approval']}' is not one of {sorted(APPROVAL_STATUSES)}.")
        if b.get("coordinator_approval") == "rejected" and b.get("processing_status") != "failed":
            problems.append(f"{bid}: rejected by the coordinator; replace it or mark it failed.")

        if b.get("processing_status") != "approved_exception":
            if not b.get("file"):
                problems.append(f"{bid}: no file set (put the PDF in data/books/ and write its file name here).")
            elif books_dir is not None and not (Path(books_dir) / b["file"]).exists():
                problems.append(f"{bid}: file '{b['file']}' not found in {books_dir}.")
    return problems


def book_record(books: list[dict], ingest: dict, passage_counts: dict[str, int]) -> tuple[list[dict], str]:
    """Rows for the book record and the Markdown document listing them."""
    pages = {b["book_id"]: b for b in ingest.get("books", [])}
    rows = []
    for b in books:
        ing = pages.get(b["book_id"], {})
        rows.append({
            "book_id": b["book_id"], "title": b.get("title", ""), "author": b.get("author", ""),
            "edition": b.get("edition", ""), "publisher": b.get("publisher", ""), "year": b.get("year", ""),
            "isbn_or_identifier": identifier(b), "source_type": b.get("source_type", ""),
            "pdf_pages": ing.get("page_count", b.get("page_count", "")),
            "pages_with_text": ing.get("pages_with_text", ""),
            "passages": passage_counts.get(b["book_id"], 0),
            "licence_or_permission": b.get("permission_status", ""),
            "permission_confirmed": "yes" if permission_confirmed(b) else "NO",
            "processing_status": b.get("processing_status", ""),
            "coordinator_approval": b.get("coordinator_approval", "pending"),
            "source_url": b.get("source_url", ""),
            "notes": b.get("notes", ""),
        })
    confirmed = sum(r["permission_confirmed"] == "yes" for r in rows)
    approved = sum(r["coordinator_approval"] == "approved" for r in rows)
    lines = [
        "# Book record",
        "",
        f"Generated from `data/book_manifest.json` by `python -m src.manifest record` on {date.today().isoformat()}.",
        "Edit the manifest, not this file, then run the command again.",
        "",
        f"- Books: **{len(rows)}** (minimum {MIN_BOOKS}); processed: "
        f"**{sum(r['processing_status'] == 'processed' for r in rows)}**",
        f"- PDF pages: **{sum(int(r['pdf_pages'] or 0) for r in rows):,}**, of which with text: "
        f"**{sum(int(r['pages_with_text'] or 0) for r in rows):,}**; passages indexed: "
        f"**{sum(r['passages'] for r in rows):,}**",
        f"- Permission to process confirmed: **{confirmed} of {len(rows)}**",
        f"- Approved by the project coordinator: **{approved} of {len(rows)}** "
        "(set `coordinator_approval` to `approved` in the manifest once the coordinator signs off)",
        "",
        "## Bibliographic details",
        "",
        "| ID | Title | Author(s) | Edition | Publisher | Year | ISBN / identifier |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['book_id'][-3:]} | {r['title']} | {r['author']} | {r['edition']} | {r['publisher']} "
                     f"| {r['year']} | {r['isbn_or_identifier']} |")
    lines += ["", "## Licensing, processing and approval", "",
              "| ID | Licence or permission basis | Confirmed | Source type | PDF pages (with text) | Passages "
              "| Processing | Coordinator |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['book_id'][-3:]} | {r['licence_or_permission']} | {r['permission_confirmed']} "
                     f"| {r['source_type']} | {r['pdf_pages']} ({r['pages_with_text']}) | {r['passages']:,} "
                     f"| {r['processing_status']} | {r['coordinator_approval']} |")
    lines += ["", "## Sources and notes", ""]
    for r in rows:
        note = f" {r['notes']}" if r["notes"] else ""
        lines.append(f"- **{r['book_id']}** <{r['source_url']}>.{note}")
    unconfirmed = [r["book_id"] for r in rows if r["permission_confirmed"] != "yes"]
    if unconfirmed:
        lines += ["", f"**Action needed:** permission is not confirmed for {', '.join(unconfirmed)}. "
                      "Confirm a purchase or library licence, or replace these books, before submission."]
    lines += ["", "The PDFs, extracted text and index are not distributed (see `.gitignore`); "
                  "page numbers are PDF page numbers unless a printed page label is shown.", ""]
    return rows, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "check", "record"])
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

    if args.command == "record":
        passages_path = resolve(cfg["paths"]["passages"])
        counts: dict[str, int] = {}
        if passages_path.exists():
            for p in read_jsonl(passages_path):
                counts[p["book_id"]] = counts.get(p["book_id"], 0) + 1
        rows, markdown = book_record(books, read_json(resolve(cfg["paths"]["ingest_report"]), default={}), counts)
        md_path, csv_path = resolve(RECORD_MD), resolve(RECORD_CSV)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads the accents
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {md_path} and {csv_path} ({len(rows)} books).")
        return 0

    problems = validate_manifest(books, resolve(cfg["paths"]["books_dir"]), cfg["group_id"])
    statuses, approvals = {}, {}
    for b in books:
        statuses[b.get("processing_status")] = statuses.get(b.get("processing_status"), 0) + 1
        approvals[b.get("coordinator_approval", "pending")] = approvals.get(b.get("coordinator_approval", "pending"), 0) + 1
    print(f"{len(books)} books in manifest; status counts: {statuses}; coordinator approval: {approvals}")
    unconfirmed = [b["book_id"] for b in books if not permission_confirmed(b)]
    if unconfirmed:
        print(f"Note: permission not confirmed for {', '.join(unconfirmed)}.")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Manifest OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
