"""Stage 5 - Retrieval: find the passages most relevant to a question.

`retrieve(question, top_k)` is the project-wide common interface.

Search runs in up to five steps:
    1. vector search (meaning) and BM25 keyword search (exact terms)
    2. the two ranked lists are merged with reciprocal rank fusion
    3. an optional cross-encoder reranker re-scores the merged candidates
    4. at most `max_per_book` passages are kept per book, so cross-book questions see several books
    5. with `distinct_pages`, at most one passage is kept per page: citations are per page, so a
       second passage from the same page would take the place of another page
When a question names one or more books (by title or a short name from the manifest's `aliases`)
and `book_routing` is on, each named book is searched on its own with the book names removed from
the question, and the books' best pages are interleaved. Questions that name a book otherwise
match its title and contents pages, and cross-book questions return pages from only one book.
`search(..., books=[...])` searches only the given books the same way, whatever the question says
(the chat page's book menu uses it).
Each result's `score` stays the vector (cosine) similarity, which the answer step uses to decide
whether the books contain any relevant evidence at all.
"""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np

from src.build_index import META_FILE, PASSAGES_FILE, load_index
from src.config import load_config, read_json, read_jsonl, resolve
from src.keyword_search import BM25

RESULT_FIELDS = ("chunk_id", "book_id", "title", "edition", "author", "page", "page_label", "section", "text")
RRF_K = 60  # standard constant for reciprocal rank fusion


class CrossEncoderReranker:
    """Scores (question, passage) pairs jointly; more precise than comparing separate embeddings."""

    def __init__(self, model_name: str, revision: str | None = None, max_length: int = 384, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name, self.revision = model_name, revision
        self.model = CrossEncoder(model_name, revision=revision, max_length=max_length, device=device)

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(question, t) for t in texts], show_progress_bar=False)]


def book_names(books: list[dict]) -> dict[str, list[str]]:
    """Names a question may use for each book: the full title, the title before a colon when that
    part is at least three words long ("Machine Learning" alone would match ordinary questions),
    the same without a leading article, and the manifest's `aliases` (acronyms such as ISLP, or an
    author's name such as Bishop's)."""
    names = {}
    for b in books:
        title = str(b.get("title") or "").strip()
        main = title.split(":")[0].strip()
        aliases = b.get("aliases") or []
        candidates = [title, *([aliases] if isinstance(aliases, str) else aliases)]
        for name in (main, re.sub(r"^(The|An|A) ", "", main)):
            if len(name.split()) >= 3:
                candidates.append(name)
        names[b["book_id"]] = sorted({c for c in candidates if c}, key=len, reverse=True)
    return names


def find_named_books(question: str, names: dict[str, list[str]]) -> tuple[list[str], str]:
    """Return (books named in the question, the question with those names removed).

    Names match only as written (title case, or the acronym's capitals), so a topic such as
    "convex optimization" in lower case does not count as naming the book Convex Optimization.
    """
    found, text = [], question
    for book, book_aliases in names.items():
        for name in book_aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
            if pattern.search(text):
                found.append(book)
                text = pattern.sub(" ", text)
                break
    return found, re.sub(r"\s+", " ", text).strip()


class Retriever:
    def __init__(self, index: faiss.Index, passages: list[dict], embedder, hybrid: bool = False,
                 candidates: int = 20, max_per_book: int | None = None, reranker=None,
                 distinct_pages: bool = False, book_names: dict[str, list[str]] | None = None):
        if index.ntotal != len(passages):
            raise ValueError(f"Index has {index.ntotal} vectors but {len(passages)} passages; rebuild the index.")
        self.index = index
        self.passages = passages
        self.by_id = {p["chunk_id"]: p for p in passages}
        self.embedder = embedder
        self.bm25 = BM25([p["text"] for p in passages]) if hybrid else None
        self.candidates = candidates
        self.max_per_book = max_per_book
        self.reranker = reranker
        self.distinct_pages = distinct_pages
        self.book_names = book_names or {}
        book_of = np.array([p["book_id"] for p in passages])
        self.book_rows = {b: np.where(book_of == b)[0] for b in dict.fromkeys(book_of.tolist())}
        self._vectors = None

    @classmethod
    def from_dir(cls, index_dir: str | Path, embedder, **options) -> "Retriever":
        index_dir = Path(index_dir)
        meta = read_json(index_dir / META_FILE, default={})
        model = getattr(embedder, "model_name", None)
        if model and meta.get("embedding_model") not in (None, model):
            raise ValueError(
                f"Index was built with {meta['embedding_model']} but the configured model is {model}. "
                "Re-run `python -m src.build_index` so queries and passages share one embedding space."
            )
        return cls(load_index(index_dir), read_jsonl(index_dir / PASSAGES_FILE), embedder, **options)

    def _result(self, i: int, score: float, **extra) -> dict:
        p = self.passages[i]
        return {**{f: p.get(f) for f in RESULT_FIELDS}, "score": float(score), **extra}

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:  # needed only to search single books
            self._vectors = self.index.reconstruct_n(0, self.index.ntotal)
        return self._vectors

    def search(self, question: str, top_k: int = 5, books: list[str] | None = None) -> list[dict]:
        """Best passages for the question, from all books or only from `books` (book IDs)."""
        if not question.strip():
            return []
        if books:
            return self._search_named_books(books, question, top_k)
        if self.book_names:
            named, core = find_named_books(question, self.book_names)
            if named and core:
                return self._search_named_books(named, core, top_k)
        query = self.embedder.encode([question], kind="query")
        pool = min(max(self.candidates, top_k), self.index.ntotal)
        scores, ids = self.index.search(query, pool)
        dense = {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}

        if self.bm25 is None:
            ranked = list(dense)
        else:
            fused: dict[int, float] = {}
            for ranking in (list(dense), [i for i, _ in self.bm25.search(question, pool)]):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)[:pool]
            for i in ranked:  # keyword-only matches still need a vector score for the evidence threshold
                if i not in dense:
                    dense[i] = float(np.dot(self.index.reconstruct(i), query[0]))

        extra: dict[int, dict] = {i: {} for i in ranked}
        if self.reranker is not None and ranked:
            rerank = self.reranker.score(question, [self.passages[i]["text"] for i in ranked])
            for i, s in zip(ranked, rerank):
                extra[i] = {"rerank_score": s}
            ranked = [i for _, i in sorted(zip(rerank, ranked), key=lambda pair: pair[0], reverse=True)]

        results, per_book, pages = [], {}, set()
        for i in ranked:
            book, page = self.passages[i]["book_id"], self.passages[i]["page"]
            if self.max_per_book and per_book.get(book, 0) >= self.max_per_book:
                continue
            if self.distinct_pages and (book, page) in pages:
                continue
            per_book[book] = per_book.get(book, 0) + 1
            pages.add((book, page))
            results.append(self._result(i, dense[i], **extra[i]))
            if len(results) == top_k:
                break
        return results

    def _search_named_books(self, books: list[str], core: str, top_k: int) -> list[dict]:
        """Search each named book on its own, then interleave the books' best pages."""
        query = self.embedder.encode([core], kind="query")[0]
        per_book = []
        for book in books:
            rows = self.book_rows.get(book)
            if rows is None or not len(rows):
                continue
            scores = self.vectors[rows] @ query
            hits, pages = [], set()
            for j in np.argsort(-scores):
                i = int(rows[j])
                page = self.passages[i]["page"]
                if page in pages:
                    continue
                pages.add(page)
                hits.append(self._result(i, float(scores[j])))
                if len(hits) == top_k:
                    break
            per_book.append(hits)
        results = []
        for rank in range(top_k):
            for hits in per_book:
                if rank < len(hits) and len(results) < top_k:
                    results.append(hits[rank])
        return results


_default: Retriever | None = None


def retriever_options(cfg: dict) -> dict:
    r = cfg["retrieval"]
    reranker = None
    rr = r.get("reranker") or {}
    if rr.get("enabled"):
        reranker = CrossEncoderReranker(rr["model_name"], rr.get("revision"), rr.get("max_length", 384))
    names = None
    if r.get("book_routing"):
        names = book_names(read_json(resolve(cfg["paths"]["manifest"]), default=[]))
    return {"hybrid": r.get("hybrid", False), "candidates": r.get("candidates", 20),
            "max_per_book": r.get("max_per_book"), "reranker": reranker,
            "distinct_pages": r.get("distinct_pages", False), "book_names": names}


def get_retriever(cfg: dict | None = None) -> Retriever:
    """The retriever built from config, loaded once per process."""
    global _default
    if _default is None:
        from src.embeddings import embedder_from_config

        cfg = cfg or load_config()
        _default = Retriever.from_dir(resolve(cfg["paths"]["index_dir"]), embedder_from_config(cfg),
                                      **retriever_options(cfg))
    return _default


def retrieve(question: str, top_k: int = 5) -> list:
    """Return passages with text, score, book_id, title, page, and chunk_id."""
    return get_retriever().search(question, top_k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or input("Question: ")
    for r in retrieve(q):
        print(f"{r['score']:.3f}  {r['chunk_id']}  {r['title']} p.{r['page']}\n    {r['text'][:160]}...")
