"""Stage 3 - Embeddings with the pinned Sentence Transformers model."""
from __future__ import annotations

import numpy as np


class SentenceEmbedder:
    """Wraps a Sentence Transformers model so the rest of the code only sees `encode`."""

    def __init__(self, model_name: str, revision: str | None = None, batch_size: int = 32,
                 normalize: bool = True, query_prefix: str = "", passage_prefix: str = "",
                 device: str = "cpu", max_seq_length: int | None = None):
        from sentence_transformers import SentenceTransformer

        if revision and revision.startswith("REPLACE"):
            raise ValueError("Set embedding.revision in config/config.yaml to a pinned commit hash.")
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.normalize = normalize
        self.prefixes = {"query": query_prefix, "passage": passage_prefix}
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        if max_seq_length:  # shorter inputs embed much faster on CPU
            self.model.max_seq_length = max_seq_length
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], kind: str = "passage", show_progress: bool = False) -> np.ndarray:
        prefix = self.prefixes[kind]
        vectors = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")


def embedder_from_config(cfg: dict) -> SentenceEmbedder:
    e = cfg["embedding"]
    return SentenceEmbedder(e["model_name"], e.get("revision"), e.get("batch_size", 32),
                            e.get("normalize", True), e.get("query_prefix", ""),
                            e.get("passage_prefix", ""), max_seq_length=e.get("max_seq_length"))
