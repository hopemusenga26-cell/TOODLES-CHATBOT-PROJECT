"""Keyword (BM25) search, hybrid retrieval, the per-book cap and the reranker hook."""
import tempfile
from pathlib import Path

from src.build_index import build_index
from src.keyword_search import BM25, tokenize
from src.retrieve import Retriever
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
