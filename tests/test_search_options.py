"""Keyword (BM25) search, hybrid retrieval, the per-book cap, the reranker hook, distinct pages
and searching named books on their own."""
import tempfile
from pathlib import Path

from src.build_index import build_index
from src.keyword_search import BM25, tokenize
from src.retrieve import Retriever, book_names, find_named_books
from tests.conftest import FakeEmbedder


def test_tokenize_drops_stopwords_and_simple_endings():
    assert tokenize("What is the hit rates of caches?") == ["hit", "rate", "cach"]


def test_bm25_ranks_exact_term_matches_first():
    bm25 = BM25(["ridge regression shrinks coefficients", "the lasso sets coefficients to zero",
                 "gradient descent steps downhill"])
    assert [i for i, _ in bm25.search("lasso", 3)] == [1]
    assert bm25.search("unknownword", 3) == []


def _retriever(passages, **options):
    embedder = FakeEmbedder()
    with tempfile.TemporaryDirectory() as d:
        build_index(passages, embedder, Path(d))
        return Retriever.from_dir(Path(d), embedder, **options)


def _passage(book, page, text):
    return {"chunk_id": f"{book}_P{page:03d}_C01", "book_id": book, "title": book, "page": page, "text": text}


PASSAGES = [
    _passage("B1", 1, "Ridge regression adds a squared penalty to least squares."),
    _passage("B1", 2, "Ridge regression shrinks coefficients towards zero."),
    _passage("B1", 3, "Ridge regression keeps all predictors in the model."),
    _passage("B2", 1, "Ridge regression is also called Tikhonov regularization."),
    _passage("B3", 1, "Use StandardScaler to standardize features before fitting."),
]


def test_hybrid_search_returns_vector_score_for_every_result():
    results = _retriever(PASSAGES, hybrid=True).search("StandardScaler standardize", 3)
    assert results[0]["book_id"] == "B3"
    assert all(isinstance(r["score"], float) for r in results)


def test_max_per_book_spreads_results_across_books():
    results = _retriever(PASSAGES, max_per_book=1).search("ridge regression", 3)
    assert len({r["book_id"] for r in results}) == len(results)


def test_reranker_reorders_candidates():
    class PreferB2:
        def score(self, question, texts):
            return [1.0 if "Tikhonov" in t else 0.0 for t in texts]

    results = _retriever(PASSAGES, reranker=PreferB2()).search("ridge regression", 3)
    assert results[0]["book_id"] == "B2" and results[0]["rerank_score"] == 1.0


def test_distinct_pages_keeps_one_passage_per_page():
    passages = PASSAGES + [{**_passage("B1", 2, "Ridge regression shrinks coefficients smoothly."),
                            "chunk_id": "B1_P002_C02"}]
    plain = _retriever(passages).search("ridge regression shrinks coefficients", 6)
    distinct = _retriever(passages, distinct_pages=True).search("ridge regression shrinks coefficients", 6)
    pages = [(r["book_id"], r["page"]) for r in distinct]
    assert len(pages) == len(set(pages))
    assert len([r for r in plain if (r["book_id"], r["page"]) == ("B1", 2)]) == 2


BOOKS = [
    {"book_id": "B1", "title": "The Elements of Statistical Learning: Data Mining", "aliases": ["ESL"]},
    {"book_id": "B2", "title": "Convex Optimization"},
    {"book_id": "B3", "title": "Machine Learning: A First Course for Engineers"},
]


def test_book_names_use_titles_aliases_and_long_prefixes_only():
    names = book_names(BOOKS)
    assert "Elements of Statistical Learning" in names["B1"] and "ESL" in names["B1"]
    assert "Machine Learning" not in names["B3"]  # a two-word prefix would match ordinary questions


def test_named_books_are_found_only_as_written():
    names = book_names(BOOKS)
    assert find_named_books("How do ESL and Convex Optimization define duality?", names) ==         (["B1", "B2"], "How do and define duality?")
    assert find_named_books("What is convex optimization?", names) == ([], "What is convex optimization?")
    assert find_named_books("Why is machine learning useful?", names)[0] == []


def test_question_naming_two_books_gets_pages_from_both():
    names = {"B1": ["Book One"], "B2": ["Book Two"]}
    results = _retriever(PASSAGES, book_names=names).search("How do Book One and Book Two describe ridge regression?", 4)
    assert {r["book_id"] for r in results} == {"B1", "B2"}
    assert [r["book_id"] for r in results[:2]] == ["B1", "B2"]  # the books' best pages are interleaved



def test_chosen_book_is_the_only_one_searched():
    retriever = _retriever(PASSAGES, hybrid=True, distinct_pages=True)
    results = retriever.search("ridge regression", 5, books=["B2"])
    assert results and {r["book_id"] for r in results} == {"B2"}
    assert retriever.search("ridge regression", 5, books=["B3"])[0]["book_id"] == "B3"  # even a poor match
    assert len({r["book_id"] for r in retriever.search("ridge regression", 5)}) > 1  # no choice: all books
