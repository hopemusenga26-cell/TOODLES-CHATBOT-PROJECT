"""Shared test fixtures: a small synthetic book and a fast stand-in embedder.

The fake embedder hashes words into a fixed-size vector, so tests run offline in
seconds without downloading any model.
"""
from __future__ import annotations

import copy
import hashlib
import re

import pymupdf
import numpy as np
import pytest

from src.build_index import build_index
from src.chunk import build_passages
from src.config import load_config
from src.generate import Answerer
from src.retrieve import Retriever

BOOK_PAGES = [
    "Chapter 1 Cache Memory\n\n"
    "A cache is a small, fast memory that stores copies of data from frequently used main memory "
    "locations. When the processor finds the requested data in the cache it is called a cache hit. "
    "When the data is not in the cache it is called a cache miss, and the block must be fetched "
    "from main memory. The hit rate is the fraction of memory accesses found in the cache.",

    "Chapter 2 Pipelining\n\n"
    "Pipelining overlaps the execution of several instructions. A classic five stage pipeline has "
    "instruction fetch, instruction decode, execute, memory access and write back stages. A data "
    "hazard occurs when an instruction depends on the result of a previous instruction that is "
    "still in the pipeline. Forwarding passes results directly between pipeline stages to reduce stalls.",

    "",  # a blank page, as books often have
]

BOOK = {
    "book_id": "G000_BOOK_001",
    "title": "Test Architecture Textbook",
    "author": "A. Author",
    "edition": "1st edition",
    "subject": "Computer Architecture",
    "file": "test_book.pdf",
    "processing_status": "pending",
    "permission_status": "synthetic test data",
}


class FakeEmbedder:
    model_name = "fake-hash-embedder"
    revision = "test"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts, kind="passage", show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if len(word) > 2:
                    out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["group_id"] = "G000"
    c["subject"] = "Computer Architecture"
    c["retrieval"]["min_score"] = 0.15
    c["generation"]["backend"] = "extractive"
    return c


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / BOOK["file"]
    doc = pymupdf.open()
    for text in BOOK_PAGES:
        page = doc.new_page()
        if text:
            heading, body = text.split("\n\n", 1)
            page.insert_text((72, 80), heading, fontsize=16)
            page.insert_textbox(pymupdf.Rect(72, 110, 520, 760), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def pages():
    return [{"book_id": BOOK["book_id"], "page": n, "page_label": None, "text": t}
            for n, t in enumerate(BOOK_PAGES, start=1) if t]


@pytest.fixture
def passages(pages, cfg):
    return build_passages(pages, {BOOK["book_id"]: BOOK}, cfg["group_id"], cfg["subject"], cfg["chunking"])


@pytest.fixture
def retriever(passages, tmp_path):
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    build_index(passages, embedder, index_dir)
    return Retriever.from_dir(index_dir, embedder)


@pytest.fixture
def answerer(retriever, cfg):
    return Answerer(retriever, cfg)
