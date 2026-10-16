"""Ingestion, chunking and manifest checks."""
from src.chunk import build_passages, chunk_page, clean_text, is_heading
from src.ingest import ingest_book
from src.manifest import MIN_BOOKS, skeleton, validate_manifest
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
