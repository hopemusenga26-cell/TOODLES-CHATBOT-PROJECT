"""Keyword search (BM25) over the passages, used alongside vector search.

Vector search matches meaning; BM25 matches exact terms such as "lasso", "perceptron"
or "StandardScaler" that vector search can miss.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy import sparse

STOPWORDS = set("""
a an and are as at be by can do does for from has have how i in into is it its of on or
that the their there these this to was what when where which who why will with would
""".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 2 or word in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        tokens.append(word)
    return tokens


class BM25:
    """Okapi BM25 with per-term weights precomputed in a sparse matrix, so a query is a column sum."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, counts = [], [], []
        lengths = np.zeros(len(texts), dtype="float32")
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            lengths[row] = len(tokens)
            term_counts: dict[int, int] = {}
            for token in tokens:
                col = vocab.setdefault(token, len(vocab))
                term_counts[col] = term_counts.get(col, 0) + 1
            rows.extend([row] * len(term_counts))
            cols.extend(term_counts.keys())
            counts.extend(term_counts.values())
        tf = sparse.csr_matrix((np.asarray(counts, dtype="float32"), (rows, cols)),
                               shape=(len(texts), max(1, len(vocab))))
        n_docs = len(texts)
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype("float32")
        avg_len = float(lengths.mean()) if n_docs else 1.0
        # weight = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
        norm = k1 * (1 - b + b * lengths / max(avg_len, 1e-9))
        coo = tf.tocoo()
        data = idf[coo.col] * coo.data * (k1 + 1) / (coo.data + norm[coo.row])
        self.weights = sparse.csc_matrix((data, (coo.row, coo.col)), shape=tf.shape)
        self.vocab = vocab

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        cols = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not cols:
            return []
        scores = np.asarray(self.weights[:, cols].sum(axis=1)).ravel()
        top_n = min(top_n, int((scores > 0).sum()))
        if top_n <= 0:
            return []
        best = np.argpartition(-scores, top_n - 1)[:top_n]
        best = best[np.argsort(-scores[best])]
        return [(int(i), float(scores[i])) for i in best if not math.isclose(scores[i], 0.0)]
