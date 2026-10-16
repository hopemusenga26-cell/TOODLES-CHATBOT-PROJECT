"""Stage 5 - Retrieval: find the passages most relevant to a question.

`retrieve(question, top_k)` is the project-wide common interface.

Search runs in up to four steps:
    1. vector search (meaning) and BM25 keyword search (exact terms)
    2. the two ranked lists are merged with reciprocal rank fusion
    3. an optional cross-encoder reranker re-scores the merged candidates
    4. at most `max_per_book` passages are kept per book, so cross-book questions see several books
Each result's `score` stays the vector (cosine) similarity, which the answer step uses to decide
whether the books contain any relevant evidence at all.
"""
from __future__ import annotations

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


class Retriever:
    def __init__(self, index: faiss.Index, passages: list[dict], embedder, hybrid: bool = False,
                 candidates: int = 20, max_per_book: int | None = None, reranker=None):
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

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if not question.strip():
            return []
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

        results, per_book = [], {}
        for i in ranked:
            book = self.passages[i]["book_id"]
            if self.max_per_book and per_book.get(book, 0) >= self.max_per_book:
                continue
            per_book[book] = per_book.get(book, 0) + 1
            results.append(self._result(i, dense[i], **extra[i]))
            if len(results) == top_k:
                break
        return results


_default: Retriever | None = None


def retriever_options(cfg: dict) -> dict:
    r = cfg["retrieval"]
    reranker = None
    rr = r.get("reranker") or {}
    if rr.get("enabled"):
        reranker = CrossEncoderReranker(rr["model_name"], rr.get("revision"), rr.get("max_length", 384))
    return {"hybrid": r.get("hybrid", False), "candidates": r.get("candidates", 20),
            "max_per_book": r.get("max_per_book"), "reranker": reranker}


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
