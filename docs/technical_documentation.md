# Technical documentation

This file covers deliverable 4: models and versions, configuration, prompts, retrieval settings,
dependencies and limitations. Every value here matches `config/config.yaml`,
`requirements.txt` and the code at the time of writing. If the config changes, update this file.

Contents:

1. [Models and versions](#1-models-and-versions)
2. [Configuration](#2-configuration)
3. [Prompts](#3-prompts)
4. [Retrieval settings](#4-retrieval-settings)
5. [Dependencies and environment](#5-dependencies-and-environment)
6. [Reproducing the system](#6-reproducing-the-system)
7. [Limitations](#7-limitations)

---

## 1. Models and versions

| Role | Model | Exact version | Licence | Where set |
|---|---|---|---|---|
| Embedding (search) | `BAAI/bge-small-en-v1.5` | revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | `embedding.model_name`, `embedding.revision` |
| Answer writing, official (Toodles Ultra) | `qwen2.5:1.5b` in Ollama (4-bit Q4_K_M of Qwen2.5-1.5B-Instruct) | digest `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`, Ollama 0.40.5 | Apache-2.0 | `generation.ollama_model` |
| Answer writing, Transformers reference | `Qwen/Qwen2.5-1.5B-Instruct` | revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 | `generation.model_name`, `generation.revision` |
| Reranker (tested, off) | `cross-encoder/ms-marco-MiniLM-L6-v2` | revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Apache-2.0 | `retrieval.reranker` |
| LoRA experiment base (optional) | `Qwen/Qwen2.5-0.5B-Instruct` | revision `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 | `lora.base_model`, `lora.base_revision` |
| Earlier embedding model (v1, replaced) | `sentence-transformers/all-MiniLM-L6-v2` | revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 | — |

[models.md](models.md) describes what each model does and why it was chosen. Hugging Face
models are loaded at the pinned revision, so an upstream change cannot alter results. The index
stores the embedding model and revision it was built with (`data/index/index_meta.json`), and
the retriever refuses to use an index built with a different model.

## 2. Configuration

All settings live in `config/config.yaml`. To try other settings without editing it, point the
`CHATBOT_CONFIG` environment variable at another YAML file. A file that starts with
`extends: config.yaml` keeps every setting from that file and overrides only the ones it lists.

### Group and paths

| Key | Value | Meaning |
|---|---|---|
| `group_id` | `GROUP 30` | The assigned group ID; it prefixes every `book_id` and `chunk_id` |
| `subject` | `Data Science & Machine Learning` | Stored in every passage record |
| `paths.*` | `data/...`, `evaluation/...` | Books, manifest, processed pages and passages, index, questions, reviews, report, chats |

### Ingestion (`src/ingest.py`)

| Key | Value | Meaning |
|---|---|---|
| `ingestion.min_chars_per_page` | 40 | A page with less text is flagged as empty |
| `ingestion.scanned_page_ratio` | 0.5 | If more than half the pages are empty, the book is treated as scanned |
| `ingestion.ocr.enabled` / `language` / `dpi` | true / `eng` / 300 | OCR through PyMuPDF and Tesseract, used only when Tesseract is installed. None of the 15 books needed it |

Text is extracted page by page with PyMuPDF, block by block in reading order. The printed page
label is kept when the PDF defines one.

### Cleaning and chunking (`src/chunk.py`)

| Key | Value | Meaning |
|---|---|---|
| `chunking.chunk_size_words` | 220 | Target passage length, packed from whole paragraphs |
| `chunking.overlap_words` | 40 | Words carried over from the previous passage on the same page |
| `chunking.min_chunk_words` | 20 | Shorter remainders are merged into the previous passage |
| `chunking.boilerplate_page_share` | 0.2 | A paragraph found on at least 20% of a book's pages (copyright notice, running header, page number) is removed first |

Passages never cross a page boundary, so every passage cites exactly one page. Ligatures and
hyphenated line breaks are repaired. The most recent heading is stored as the passage's
`section`. Passage IDs look like `GROUP 30_BOOK_002_P087_C04` (book, PDF page, chunk on that page).

### Embeddings and index (`src/embeddings.py`, `src/build_index.py`)

| Key | Value | Meaning |
|---|---|---|
| `embedding.batch_size` | 32 | Passages embedded per batch when building the index |
| `embedding.max_seq_length` | 256 | Tokens per passage; 512 was about 3 times slower to index on the CPU |
| `embedding.normalize` | true | Unit-length vectors, so the inner product equals cosine similarity |
| `embedding.query_prefix` | `"Represent this sentence for searching relevant passages: "` | BGE's recommended instruction, added to questions only |
| `embedding.passage_prefix` | `""` | None for BGE |
| Index type | `faiss.IndexFlatIP` | Exact search over 22,216 × 384 vectors (61.2 MB with the passage records) |

### Generation (`src/generate.py`)

| Key | Value | Meaning |
|---|---|---|
| `generation.backend` | `ollama` | `ollama` (Toodles Ultra, official), `hf` (Transformers reference) or `extractive` (quotes only, Toodles Instant) |
| `generation.max_new_tokens` | 200 | Longest answer, in tokens |
| `generation.max_passages` | 3 | Passages the model reads (retrieval still returns 5) |
| `generation.device` / `dtype` | `cpu` / `bfloat16` | For the `hf` backend: bfloat16 about 50–70 s per answer and 3.8 GB RAM; float32 about 30–45 s but 6.6 GB |
| `generation.min_support` | 0.4 | Share of an answer sentence's key words that must appear in a cited passage, or the sentence is removed |
| `generation.fallback_to_quotes` | true | If no model sentence survives the check but relevant passages exist, quote the books' best sentences instead of refusing |
| `generation.ollama_url` | `http://localhost:11434` | Local Ollama server |
| `generation.ollama_keep_alive` | `30m` | Keeps the model loaded between questions |
| `generation.ollama_num_ctx` | 2048 | Context window; prompts use at most about 1,400 tokens plus the answer |
| `generation.ollama_num_gpu` | 99 | Puts all layers on the GPU (GeForce MX450, about 1.1 GB used); remove on a laptop without an NVIDIA GPU |
| `generation.adapter` | `null` | Optional LoRA adapter folder for the `hf` backend |

### App (`src/app.py`)

| Key | Value | Meaning |
|---|---|---|
| `app.name` | `TOODLES` | Page heading and browser tab |
| `app.theme` | `dark` | Starting look (switch in the sidebar) |
| `app.default_answer_writer` | `ollama` | Toodles Ultra preselected; Toodles Instant (`extractive`) is the other choice |

### LoRA experiment (`src/lora_data.py`, `src/lora.py`; optional)

| Key | Value |
|---|---|
| `lora.r` / `alpha` / `dropout` | 8 / 16 / 0.05 |
| `lora.target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `lora.learning_rate` / `epochs` / `grad_accum` | 2e-4 / 2 / 4 |
| `lora.max_length` / `dev_share` / `seed` | 1,536 tokens / 0.08 / 42 |

## 3. Prompts

The prompts are defined in `src/generate.py` and are identical for the Ollama, Transformers
and LoRA models. They are sent as a chat: one system message and one user message, using each
model's own chat template.

### System message (`SYSTEM_PROMPT`)

```text
You are a careful teaching assistant. You answer questions using only the numbered textbook passages you are given, never outside knowledge, and you cite the passage behind every sentence.
```

### User message (`build_prompt` + `INSTRUCTIONS`)

```text
Passages:

[1] {title} ({edition}), p. {page}
{passage text}

[2] {title} ({edition}), p. {page}
{passage text}

[3] {title} ({edition}), p. {page}
{passage text}

Question: {question}

Answer the question in 1 to 4 complete sentences, using only facts stated in the passages above.
End each sentence with the number of the passage it comes from in square brackets.
Example of the format only: Water boils at 100 degrees Celsius at sea level [2].
Only use passage numbers listed above.
If the passages do not discuss the topic of the question, or do not contain the answer, write only: INSUFFICIENT_EVIDENCE
```

When the PDF has a printed page number that differs from the PDF page, the label reads
`p. 77 (PDF page 87)`. The example sentence is deliberately off-topic, so the model copies its
format and not its content.

### Illustrative filled prompt

The passages below were made up for this document. Real prompts contain book text, which is
not reproduced here.

```text
Passages:

[1] Example Statistics Textbook (2nd edition), p. 64
Ridge regression adds a penalty proportional to the sum of squared coefficients to the least squares objective. The penalty shrinks the coefficients towards zero, which lowers variance at the cost of some bias.

[2] Example Learning Handbook (1st edition), p. 212
Cross-validation estimates test error by repeatedly fitting a model on part of the data and evaluating it on the rest.

Question: Why does ridge regression reduce overfitting?

Answer the question in 1 to 4 complete sentences, ...
```

An answer that passes every check: *"Ridge regression adds a penalty on the sum of squared
coefficients, which shrinks them towards zero [1]. This lowers variance at the cost of some
bias [1]."*

### What happens to the model's reply

1. If it contains `INSUFFICIENT_EVIDENCE`, the chatbot answers *"The provided books do not
   contain enough evidence to answer this question."* with no sources.
2. Otherwise every sentence is checked against the passages (`ground_answer`). A sentence is
   kept only if at least 40% of its key words (lower-cased, stop words removed, simple suffixes
   stripped) appear in a passage. It is cited to that passage: the one it cited if that passage
   supports it, otherwise the best-supporting one. Unsupported sentences are removed.
3. If nothing survives but relevant passages were found, the reply is replaced by the books'
   best-matching sentences, prefixed with *"The books say: "*.
4. Citations are renumbered 1, 2, 3 in order of first use. Numbers pointing outside the passage
   list are fabricated and removed. Every source must resolve to a stored passage with the same
   book and page, or the chatbot declines.

### Prompts used only to build LoRA training data (`src/lora_data.py`)

```text
System: You write clear exam questions for a university machine learning course.
User:   Passage:
        {passage text}

        Write one question that this passage answers. The question must make sense on its own, without seeing the passage: do not mention 'the passage', 'the text', 'the author' or 'this chapter'. It must be answerable from the passage alone. Write only the question.
```

The teacher's answers to these questions use the chatbot's own prompt above.

## 4. Retrieval settings

| Setting | Value | Chosen because (validation split, 20 answerable questions) |
|---|---|---|
| `retrieval.top_k` | 5 | The common interface default; the hit rate is measured at 5 |
| `retrieval.hybrid` | true | Vector search alone and hybrid both found the right page for 80%, but hybrid also handles exact names (`train_test_split`, confusion matrix) |
| `retrieval.candidates` | 20 | Each search (vector and BM25) returns 20 before merging |
| Merging | reciprocal rank fusion, k = 60 | Standard constant; a passage's merged score is the sum of 1 / (60 + rank) over both lists |
| `retrieval.reranker.enabled` | false | Reranker: 75% (−5 points) and +1.1 s per question |
| `retrieval.max_per_book` | null | Limit of 2 per book: 70% (−10 points) |
| `retrieval.distinct_pages` | true | At most one passage per page in the results. v4: right page in top 5 on validation 75% → 80% |
| `retrieval.book_routing` | true | A question naming a book (full title, a title prefix of at least 3 words, or a manifest `aliases` entry such as ISLP, ESL, D2L, Bishop's), written as in the title, searches that book on its own with the name removed; several named books are interleaved. v4: 80% → 85% |
| `retrieval.min_score` | 0.67 | Evidence threshold on cosine similarity. Questions the books answer scored 0.72–0.88; off-topic ones 0.49–0.63 |
| BM25 parameters | k1 = 1.5, b = 0.75 | Standard values; the same simple suffix stemming as the grounding check |

Each result's `score` is always the cosine similarity from the embedding model, even for
passages found only by BM25. That keeps one meaning for the threshold. A typical search,
including embedding the question, takes about 55 ms.

`retrieve(question, top_k=5)` returns, for each passage: `chunk_id, book_id, title, edition,
author, page, page_label, section, text, score`.

## 5. Dependencies and environment

### Python packages (`requirements.txt`, pinned)

| Package | Version | Used for |
|---|---|---|
| pymupdf | 1.28.2 | PDF text extraction and OCR hook |
| numpy | 2.5.4 | Vectors |
| scipy | 1.18.1 | Sparse matrices for BM25 |
| pyyaml | 6.0.3 | Configuration |
| torch | 2.14.1 (CPU build) | Model inference and LoRA training |
| transformers | 5.20.0 | Qwen models, tokenizers |
| sentence-transformers | 6.1.0 | Embedding model, reranker |
| huggingface-hub | 1.33.0 | Pinned model downloads |
| faiss-cpu | 1.15.1 | Vector index |
| peft | 0.21.2 | LoRA adapters (optional experiment) |
| accelerate | 1.15.0 | Required by peft |
| streamlit | 1.66.0 | Chat interface |
| psutil | 7.2.2 | Peak RAM in the evaluation report |
| pytest | 9.1.1 | Tests |
| openpyxl | 3.1.5 | Human review spreadsheet |

### Outside Python

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.13.14 | Everything |
| Ollama | 0.40.5, model `qwen2.5:1.5b` | Toodles Ultra (the default answer mode). Without it, use Toodles Instant or the `hf` backend |
| Tesseract | any recent | OCR of scanned books only (not needed for the current 15) |

### Tested hardware and run configuration

- Intel Core i5-1135G7 (4 cores), 16 GB RAM, NVIDIA GeForce MX450 2 GB (used by Ollama only),
  Windows 11, plugged in. On battery the CPU runs about half as fast.
- Minimum: 8 GB RAM for Toodles Ultra (about 1.4 GB for the model plus the index and app);
  16 GB for the `hf` backend. About 5 GB of free disk for the models (3 GB of it is the
  Transformers copy of the answer model) and the index.
- Offline: once the models are downloaded, set `HF_HUB_OFFLINE=1` (`start_chatbot.bat` does this).

## 6. Reproducing the system

```bash
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
python -m src.manifest check          # 15 books, all fields present
python -m src.pipeline                # ingest, chunk, embed, index
pytest                                # automated tests
python -m src.evaluate --split test   # official report: evaluation/report.json
streamlit run src/app.py              # or double-click start_chatbot.bat
```

Optional:

```bash
python -m src.fundamentals            # Transformer fundamentals experiments
python -m src.manifest record         # book record (docs/book_record.md, .csv)
python -m src.lora_data               # LoRA training split
python -m src.lora train --group A --name lora_A
```

## 7. Limitations

**Data and processing**

- The coordinator has not yet approved the book list or the models. Books 014 and 015 were
  replaced on 2026-10-18 with freely available books, so permission is documented for all 15.
- A passage never spans two pages, so an explanation split across a page break is retrieved as
  two passages.
- **50% of passages are longer than the 256-token embedding limit**, so their last sentences do
  not count towards vector search. This was measured by `src/fundamentals.py`; on average a
  passage is 258.3 bge tokens. BM25 still reads the full text. Raising
  `embedding.max_seq_length` to 512 (about 3 times slower indexing) or using shorter chunks are
  candidate fixes. They must be chosen on the validation split and need an index rebuild.
- Equations, tables and figures are extracted as plain text and lose their layout.
- Headers that change every chapter can survive the 20% boilerplate rule.

**Retrieval**

- On the held-out test set the right page is in the top 5 for 60.7% of answerable questions
  (v3), against 75% on validation and a target of 80%. Exact-page matching undercounts, because
  several books often explain the same thing on pages not listed as expected evidence. The
  report also gives a ±1-page hit rate.
- Cross-book questions often miss the expected page of one of the two books.
- Title, contents and preface pages are indexed. Questions that name a book retrieve them instead of the
  pages that answer the question. For example, the cross-book question Q032 retrieved the title,
  contents and notation pages of *Machine Learning: A First Course for Engineers and Scientists*.

**Answers**

- The grounding check compares words, not meaning. A sentence can reuse a passage's words and
  still change their meaning, and a correct paraphrase can be removed.
- The quote fallback can quote loosely related sentences, and it answered 1 of the 5
  not-in-the-books test questions. Overall, 2 of the 5 were answered instead of refused.
- The 1.5B model rarely carries out calculations correctly, even when the formula is in the
  passage: it declined two on the test set and gave a wrong accuracy (85% instead of 90%) on
  validation, which the word-overlap check cannot catch.
- The automatic answer scores (word-overlap F1 and embedding similarity with the reference
  answer) only estimate correctness. Answer correctness still needs the human review
  (`evaluation/human_reviews.json`), which is pending.
- The evaluation questions and reference answers are drafts and need subject-expert review.

**Efficiency**

- Toodles Ultra takes about 6–9 s per answer (8.4 s mean and 12.1 s at the 95th percentile on the v3 test run); the Transformers reference takes about 60 s on this
  CPU. The first question after starting is slower while the models load.
