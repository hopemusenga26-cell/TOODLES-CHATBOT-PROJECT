"""Stage 4 - Indexing: embed every passage and store a local FAISS index.

The index folder holds three files:
    index.faiss      the vectors (inner product = cosine, because vectors are normalised)
    passages.jsonl   the passage records, in the same order as the vectors
    index_meta.json  embedding model, revision and settings used to build it

Usage:
    python -m src.build_index
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from src.config import load_config, read_jsonl, resolve, write_json, write_jsonl

INDEX_FILE = "index.faiss"
PASSAGES_FILE = "passages.jsonl"
META_FILE = "index_meta.json"


def build_index(passages: list[dict], embedder, index_dir: str | Path, meta: dict | None = None) -> faiss.Index:
    if not passages:
        raise ValueError("No passages to index. Run ingestion and chunking first.")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    vectors = embedder.encode([p["text"] for p in passages], kind="passage", show_progress=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Serialise through Python so paths with spaces or non-ASCII characters work on Windows.
    (index_dir / INDEX_FILE).write_bytes(faiss.serialize_index(index).tobytes())
    write_jsonl(passages, index_dir / PASSAGES_FILE)
    write_json({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index_type": "faiss.IndexFlatIP",
        "embedding_model": getattr(embedder, "model_name", type(embedder).__name__),
        "embedding_revision": getattr(embedder, "revision", None),
        "dimension": int(vectors.shape[1]),
        "normalized": True,
        "passages": len(passages),
        "books": len({p["book_id"] for p in passages}),
        **(meta or {}),
    }, index_dir / META_FILE)
    return index


def load_index(index_dir: str | Path) -> faiss.Index:
    data = np.frombuffer((Path(index_dir) / INDEX_FILE).read_bytes(), dtype="uint8")
    return faiss.deserialize_index(data)


def main() -> int:
    from src.embeddings import embedder_from_config

    cfg = load_config()
    passages = read_jsonl(resolve(cfg["paths"]["passages"]))
    embedder = embedder_from_config(cfg)
    build_index(passages, embedder, resolve(cfg["paths"]["index_dir"]), meta={
        "group_id": cfg["group_id"],
        "subject": cfg["subject"],
        "chunking": cfg["chunking"],
        "embedding_settings": {key: cfg["embedding"].get(key) for key in
                               ("max_seq_length", "query_prefix", "passage_prefix", "normalize")},
    })
    print(f"Indexed {len(passages)} passages with {embedder.model_name}@{embedder.revision}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
