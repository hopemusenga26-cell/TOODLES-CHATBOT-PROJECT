"""Ingestion, chunking and manifest checks."""
from src.chunk import build_passages, chunk_page, clean_text, is_heading
from src.ingest import ingest_book
from src.manifest import MIN_BOOKS, book_record, identifier, permission_confirmed, skeleton, validate_manifest
from tests.conftest import BOOK

PASSAGE_FIELDS = {"group_id", "subject", "book_id", "title", "edition", "page", "chunk_id", "text"}


def test_ingest_keeps_book_and_page_numbers(pdf_path, cfg):
    pages, summary = ingest_book(pdf_path, BOOK, cfg["ingestion"])
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert all(p["book_id"] == BOOK["book_id"] for p in pages)
    assert "cache hit" in pages[0]["text"]
    assert "Forwarding" in pages[1]["text"]


def test_ingest_flags_empty_pages_and_text_pdf(pdf_path, cfg):
    _, summary = ingest_book(pdf_path, BOOK, cfg["ingestion"])
    assert summary["page_count"] == 3
    assert summary["empty_pages"] == [3]
    assert summary["source_type"] == "pdf_text"
    assert summary["processing_status"] == "processed"


def test_clean_text_fixes_ligatures_and_hyphenation():
    assert clean_text("eﬃcient pro-\ncessor") == "efficient processor"
    assert clean_text("first para\nline two\n\nsecond") == "first para line two\n\nsecond"


def test_heading_detection():
    assert is_heading("Chapter 4 The Processor")
    assert is_heading("4.2 Pipelined Datapath")
    assert not is_heading("A cache is a small, fast memory that stores copies of data.")


def test_chunks_respect_size_overlap_and_lose_no_words():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_page(text, size=200, overlap=50, min_words=20)
    assert all(len(c.split()) <= 200 for c in chunks)
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(text.split())


def test_passages_carry_required_metadata(passages):
    assert len(passages) == 2
    for p in passages:
        assert PASSAGE_FIELDS <= p.keys()
        assert p["chunk_id"].startswith(f"{p['book_id']}_P{p['page']:03d}_C")
    assert len({p["chunk_id"] for p in passages}) == len(passages)
    assert passages[0]["section"] == "Chapter 1 Cache Memory"


def test_notice_repeated_on_every_page_is_removed(cfg):
    notice = "This book has been published by Example Press. Free for personal use only."
    topics = ["least squares", "ridge penalty", "lasso path", "gradient step", "loss curve", "bias term"]
    pages = [{"book_id": BOOK["book_id"], "page": n, "page_label": None,
              "text": f"{notice}\n\n{n} Chapter heading\n\nRegression page about {topics[n % 6]} "
                      f"and {topics[(n * 5) % 6]} " + f"word{n} " * 25}
             for n in range(1, 31)]
    passages = build_passages(pages, {BOOK["book_id"]: BOOK}, "G000", "ML", cfg["chunking"])
    assert len(passages) == 30
    assert all("Example Press" not in p["text"] for p in passages)
    assert all("Chapter heading" not in p["text"] for p in passages)  # running header, numbers ignored
    assert passages[0]["text"].startswith("Regression page about")


def test_manifest_requires_fifteen_complete_books():
    books = skeleton("G000", "Computer Architecture")
    assert len(books) == MIN_BOOKS
    problems = validate_manifest(books)
    assert any("missing title" in p for p in problems)

    for b in books:
        b.update(title="T", author="A", edition="1st", permission_status="library licence", file="x.pdf")
    assert validate_manifest(books) == []
    assert any("at least 15" in p for p in validate_manifest(books[:14]))


def test_coordinator_approval_values_are_checked():
    books = skeleton("G000", "ML")
    for b in books:
        b.update(title="T", author="A", edition="1st", permission_status="free PDF", file="x.pdf")
    assert all(b["coordinator_approval"] == "pending" for b in books)
    books[0]["coordinator_approval"] = "maybe"
    books[1]["coordinator_approval"] = "rejected"
    problems = validate_manifest(books)
    assert any("G000_BOOK_001: coordinator_approval 'maybe'" in p for p in problems)
    assert any("G000_BOOK_002: rejected" in p for p in problems)


def test_book_record_lists_editions_licences_and_open_issues():
    books = skeleton("G000", "ML", count=2)
    books[0].update(title="Book One", author="A", edition="2nd edition", isbn="978-0", permission_status="CC BY 4.0",
                    processing_status="processed", coordinator_approval="approved")
    books[1].update(title="Book Two", author="B", edition="1st edition", source_url="https://example.org/book2",
                    permission_status="NOT CONFIRMED: commercial book")
    ingest = {"books": [{"book_id": "G000_BOOK_001", "page_count": 10, "pages_with_text": 9}]}
    rows, markdown = book_record(books, ingest, {"G000_BOOK_001": 30})
    assert identifier(books[1]) == "https://example.org/book2"  # no ISBN: stable identifier instead
    assert [r["permission_confirmed"] for r in rows] == ["yes", "NO"]
    assert not permission_confirmed(books[1])
    assert "| 001 | Book One | A | 2nd edition |" in markdown
    assert "Approved by the project coordinator: **1 of 2**" in markdown
    assert "permission is not confirmed for G000_BOOK_002" in markdown
