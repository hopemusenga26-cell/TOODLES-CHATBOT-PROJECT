"""Run stages 1-4 in order: ingest the books, chunk them, and build the index.

Usage:
    python -m src.pipeline
"""
from __future__ import annotations

import sys

from src import build_index, chunk, ingest


def main() -> int:
    print("== 1. Ingestion ==")
    ingest.main()
    print("\n== 2. Chunking ==")
    chunk.main()
    print("\n== 3-4. Embedding and indexing ==")
    return build_index.main()


if __name__ == "__main__":
    sys.exit(main())
