"""Stage 1 - Ingestion: extract text page by page from the book PDFs.

Reads every book in the manifest, keeps the book and page number with each
page of text, flags empty or scanned pages, and updates the manifest's
source_type and processing_status.

Usage:
    python -m src.ingest
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from src.config import load_config, resolve, write_json, write_jsonl
from src.manifest import load_manifest


def ocr_available() -> bool:
    """PyMuPDF can only OCR when Tesseract and its language data are installed."""
    try:
        pymupdf.get_tessdata()
        return True
    except Exception:
        return shutil.which("tesseract") is not None


def _page_text(page: pymupdf.Page, textpage=None) -> str:
    """Text blocks in reading order, separated by blank lines so paragraphs survive."""
    blocks = page.get_text("blocks", textpage=textpage, sort=True)
    return "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())


def ingest_book(pdf_path: str | Path, book: dict, ingest_cfg: dict, use_ocr: bool = False) -> tuple[list[dict], dict]:
    """Return (page records, processing summary) for one book."""
    min_chars = ingest_cfg["min_chars_per_page"]
    ocr_cfg = ingest_cfg.get("ocr", {})
    pages, empty_pages, ocr_pages, failed_pages = [], [], [], []

    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
        for i, page in enumerate(doc):
            page_no = i + 1
            try:
                text = _page_text(page)
            except Exception:
                failed_pages.append(page_no)
                continue

            if len(text) < min_chars and use_ocr:
                try:
                    tp = page.get_textpage_ocr(language=ocr_cfg.get("language", "eng"),
                                               dpi=ocr_cfg.get("dpi", 300), full=True)
                    ocr_text = _page_text(page, textpage=tp)
                    if len(ocr_text) >= min_chars:
                        text = ocr_text
                        ocr_pages.append(page_no)
                except Exception:
                    pass

            if len(text) < min_chars:
                empty_pages.append(page_no)
            pages.append({
                "book_id": book["book_id"],
                "page": page_no,
                "page_label": page.get_label() or None,
                "text": text,
            })

    ratio = ingest_cfg["scanned_page_ratio"]
    unreadable = len(empty_pages) + len(failed_pages)
    # Pages with no native text layer, whether or not OCR rescued them.
    scanned = page_count > 0 and (unreadable + len(ocr_pages)) / page_count > ratio
    if page_count == 0 or unreadable == page_count:
        status = "failed" if use_ocr else "needs_ocr"
    elif unreadable / page_count > ratio:
        status = "needs_ocr"
    else:
        status = "processed"

    summary = {
        "book_id": book["book_id"],
        "file": str(book.get("file", "")),
        "page_count": page_count,
        "pages_with_text": page_count - unreadable,
        "empty_pages": empty_pages,
        "ocr_pages": ocr_pages,
        "failed_pages": failed_pages,
        "characters": sum(len(p["text"]) for p in pages),
        "source_type": "pdf_scanned" if scanned else "pdf_text",
        "processing_status": status,
    }
    return pages, summary


def run_ingestion(cfg: dict) -> dict:
    paths = cfg["paths"]
    books_dir = resolve(paths["books_dir"])
    manifest_path = resolve(paths["manifest"])
    books = load_manifest(manifest_path)
    use_ocr = cfg["ingestion"]["ocr"].get("enabled", False) and ocr_available()

    all_pages, summaries, skipped = [], [], []
    for book in books:
        if book.get("processing_status") == "approved_exception":
            continue
        if not book.get("file"):
            skipped.append({"book_id": book["book_id"], "reason": "no file set in manifest"})
            continue
        pdf_path = books_dir / book["file"]
        if not pdf_path.exists():
            skipped.append({"book_id": book["book_id"], "reason": f"file not found: {book['file']}"})
            continue
        try:
            pages, summary = ingest_book(pdf_path, book, cfg["ingestion"], use_ocr)
        except Exception as e:  # corrupt or encrypted PDF
            book["processing_status"] = "failed"
            skipped.append({"book_id": book["book_id"], "reason": f"could not open: {e}"})
            continue
        all_pages.extend(p for p in pages if p["text"])
        summaries.append(summary)
        book["source_type"] = summary["source_type"]
        book["processing_status"] = summary["processing_status"]
        book["page_count"] = summary["page_count"]
        print(f"{book['book_id']}: {summary['pages_with_text']}/{summary['page_count']} pages with text, "
              f"{len(summary['empty_pages'])} empty, {len(summary['ocr_pages'])} OCR -> {summary['processing_status']}")

    write_jsonl(all_pages, resolve(paths["pages"]))
    write_json(books, manifest_path)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ocr_available": use_ocr,
        "books_in_manifest": len(books),
        "books_processed": sum(1 for s in summaries if s["processing_status"] == "processed"),
        "books_needing_ocr": [s["book_id"] for s in summaries if s["processing_status"] == "needs_ocr"],
        "books_skipped": skipped,
        "total_pages": sum(s["page_count"] for s in summaries),
        "total_pages_with_text": sum(s["pages_with_text"] for s in summaries),
        "books": summaries,
    }
    write_json(report, resolve(paths["ingest_report"]))
    return report


def main() -> int:
    report = run_ingestion(load_config())
    print(f"\n{report['books_processed']} book(s) processed, {len(report['books_needing_ocr'])} need OCR, "
          f"{len(report['books_skipped'])} skipped. OCR available: {report['ocr_available']}.")
    for s in report["books_skipped"]:
        print(f"  skipped {s['book_id']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
