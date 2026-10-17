# TOODLES: a book-grounded machine learning chatbot (Group G000)

*Answers from 15 machine learning books, with the page to prove it.*

| | |
|---|---|
| **Group ID** | G000 *(replace with your assigned ID, also in `config/config.yaml`)* |
| **Subject** | Machine Learning (linear regression and model training) |
| **Books** | 15, listed in `data/book_manifest.json` (8,256 pages, 21,630 passages) |

The chatbot answers questions using only the group's books. It finds the most
relevant passages, writes an answer from them, and cites the exact book and page
for every claim. If the books don't contain the answer, it says so.

## Books

| # | Title | Author(s) | Year | Permission |
|---|---|---|---|---|
| 001 | Python Data Science Handbook | Jake VanderPlas | 2016 | free from author/publisher |
| 002 | An Introduction to Statistical Learning | Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor | 2023 | free from author/publisher |
| 003 | The Elements of Statistical Learning | Trevor Hastie, Robert Tibshirani, Jerome Friedman | 2009 | free from author/publisher |
| 004 | Regression and Other Stories | Andrew Gelman, Jennifer Hill, Aki Vehtari | 2020 | free from author/publisher |
| 005 | Introduction to Applied Linear Algebra | Stephen Boyd, Lieven Vandenberghe | 2018 | free from author/publisher |
| 006 | Mathematics for Machine Learning | Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong | 2020 | free from author/publisher |
| 007 | Convex Optimization | Stephen Boyd, Lieven Vandenberghe | 2004 | free from author/publisher |
| 008 | A Course in Machine Learning | Hal Daume III | 2017 | free from author/publisher |
| 009 | Pattern Recognition and Machine Learning | Christopher M. Bishop | 2006 | free from author/publisher |
| 010 | Probabilistic Machine Learning | Kevin P. Murphy | 2022 | free from author/publisher |
| 011 | Foundations of Data Science | Avrim Blum, John Hopcroft, Ravindran Kannan | 2020 | free from author/publisher |
| 012 | Dive into Deep Learning | Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | 2023 | free from author/publisher |
| 013 | Think Stats | Allen B. Downey | 2014 | free from author/publisher |
| 014 | Machine Learning with Python Cookbook | Chris Albon | 2018 | ⚠️ not confirmed |
| 015 | Mastering Machine Learning with Python in Six Steps | Manohar Swamynathan | 2017 | ⚠️ not confirmed |

Full bibliographic details, licences and processing status are in `data/book_manifest.json`.
Books 014 and 015 are commercial titles: the group must confirm a purchase or library licence
before submission, or replace them with free books.

## How it works

```
PDFs ──► 1 ingest ──► 2 chunk ──► 3 embed ──► 4 index (FAISS)
                                                   │
question ─────────────────────────────► 5 retrieve top-k passages
                                                   │
                       6 generate answer from those passages only
                                                   │
               7 check every citation resolves to a stored passage,
                 or answer "insufficient evidence"
                                                   │
                                     8 evaluate on held-out questions
```

| Stage | File | Output |
|---|---|---|
| 1. Ingestion | `src/ingest.py` | `data/processed/pages.jsonl`, `ingest_report.json` |
| 2. Cleaning and chunking | `src/chunk.py` | `data/processed/passages.jsonl` |
| 3. Embeddings | `src/embeddings.py` | |
| 4. Indexing | `src/build_index.py` | `data/index/` |
| 5. Retrieval | `src/retrieve.py` | `retrieve(question, top_k=5)` |
| 6. Generation | `src/generate.py` | `answer(question)` |
| 7. Citations and abstention | `src/citations.py`, `src/generate.py` | |
| 8. Evaluation | `src/evaluate.py` | `evaluation/report.json` |
| Interface | `src/app.py` | Streamlit chat page |

## Setup

Requirements: Python 3.13 (or the coordinator's chosen version), about 8 GB RAM
(16 GB recommended), and about 5 GB of free disk for the models and index. No GPU needed.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

The first run downloads the embedding model (about 90 MB) from Hugging Face, and the
answer model (about 3 GB) the first time the language-model backend is used. After that,
everything runs offline. On a slow connection, download the answer model once in advance:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
```

If the connection keeps dropping, download the 3 GB weights file with a tool that can resume,
for example `curl -L -C - -o model.safetensors <file URL>`, into
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<revision>/`, then
check its SHA-256 against the value shown on the model's Hugging Face file page.

To run without internet once everything is downloaded, set `HF_HUB_OFFLINE=1`.

## Run it

1. **Add the books.** Put the PDFs in `data/books/`. Create the manifest and fill in every entry:
   ```bash
   python -m src.manifest init     # creates 15 empty entries
   python -m src.manifest check    # lists anything missing
   ```
2. **Build the index** (ingest, chunk, embed and index in one go):
   ```bash
   python -m src.pipeline
   ```
3. **Ask questions:** on Windows, double-click `start_chatbot.bat` (it opens the browser and
   won't start a second copy), or run:
   ```bash
   streamlit run src/app.py
   python -m src.generate "What is a cache miss?"     # one answer as JSON
   python -m src.retrieve "What is a cache miss?"     # retrieved passages only
   ```
4. **Test and evaluate:**
   ```bash
   pytest
   python -m src.evaluate --split validation
   python -m src.evaluate --split test       # only for the final evaluation
   ```

## The chat page

- **Logo:** the TOODLES mark, a gold T with orb eyes (`src/logo.py`), shown in the sidebar, the greeting
  and the browser tab. While an answer loads it animates (eyes glance, dots ripple); beside each answer
  it smiles for an answer from the books or greys out when the books don't cover the question.
- **Look:** yellow, black, white and grey, with a **Dark mode** switch in the sidebar (starts dark;
  `app.theme` in the config). Buttons, chips, your messages and the message box use a soft clay style;
  the sidebar, passages panel and mode menu use frosted glass (`src/theme.py`).
- **Sidebar:** New chat, Recents (your previous chats, saved on this computer in `data/chats/`,
  which is kept out of Git because chats quote the books), Delete this chat, and book details.
- **New chat:** a greeting with the message box in the middle and suggested questions underneath.
- **Conversation:** your questions appear as bubbles on the right; answers appear as text with
  numbered citations, citation chips with the book and page, and a "Show passages" panel with the
  exact text used. The message box stays at the bottom.
- **Mode menu** beside the message box: Toodles Instant or Toodles Ultra (default),
  and how many passages to search.
- **Thumbs up/down** under each answer is saved with the chat and can help with human review.

## Common interfaces

Every passage in the index keeps `group_id, subject, book_id, title, author, edition, page,
page_label, section, chunk_id, text`. `page` is the PDF page number; `page_label` is the
printed page number when the PDF has one.

```python
from src.retrieve import retrieve
retrieve("What is pipelining?", top_k=5)
# [{"chunk_id", "book_id", "title", "page", "text", "score", ...}, ...]

from src.generate import answer
answer("What is pipelining?")
# {"answer": "... [1]", "sources": [{"book_id", "page", "chunk_id"}], "insufficient_evidence": false}
```

## Configuration

Everything is set in `config/config.yaml`: group ID, subject, chunk size and
overlap, embedding model and pinned revision, retrieval `top_k` and minimum score,
and the answer model and pinned revision.

| Setting | Current value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (MIT), 256 tokens per passage, with BGE's recommended query instruction. Replaces `all-MiniLM-L6-v2`; use the project-approved model when the coordinator publishes one. |
| Answer model | `Qwen/Qwen2.5-1.5B-Instruct`, bfloat16 on CPU (about 3.8 GB RAM); reads the top 3 passages. Set `dtype: float32` for faster answers if about 7 GB of RAM is free. Fast demo mode: the same model via Ollama (`qwen2.5:1.5b`). |
| Chunks | about 220 words, 40-word overlap, never crossing a page; paragraphs repeated on 20%+ of a book's pages (copyright notices, running headers, page numbers) are removed first |
| Index | FAISS `IndexFlatIP` on normalised vectors (cosine similarity) |
| Search | Hybrid: vector search plus BM25 keyword search, merged by reciprocal rank fusion; a passage counts as evidence only if its cosine similarity is at least 0.67. Optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`) and per-book cap are available but off (see below). |

If the embedding model changes, rebuild the index (`python -m src.build_index`). The
app refuses to search an index that was built with a different model.

There are three answer backends, chosen with `generation.backend`. The app offers two of them:

| In the app | Backend | Time per answer |
|---|---|---|
| **Toodles Instant** | `extractive` | under 1 s |
| **Toodles Ultra** (default, official) | `ollama` | about 5 s |
| not in the app (reference only) | `hf` | about 60 s |

- `extractive` answers by quoting the best-matching passage sentences, with no language
  model. It is fast, runs on any laptop, and is the baseline for comparison.
- `hf` has a local Hugging Face chat model write the answer from the retrieved
  passages. `Qwen2.5-0.5B-Instruct` was tested and is too small for this task: it replied
  with bare citation numbers. `Qwen2.5-1.5B-Instruct` writes usable answers.
- `ollama` (Toodles Ultra, the default and the official evaluation backend) sends the same prompt to the same Qwen2.5 1.5B model in a
  compressed 4-bit format (`qwen2.5:1.5b`, Q4_K_M, 0.92 GB, digest `65ec0654…`) served by
  [Ollama](https://ollama.com). Measured on the target laptop: 6–10 s per answer instead of
  44–70 s, about 1.4 GB of memory. The model is preloaded when this mode is chosen and kept in
  memory for 30 minutes. To set it up, install Ollama, then run `ollama pull qwen2.5:1.5b` once.
  The same citation and grounding checks apply. On the held-out test set it matched the `hf`
  copy on retrieval and citations, refused fewer answerable questions, and was about 12 times
  faster, so `hf` was removed from the app. It stays in the code, pinned to its exact revision,
  as the Transformers reference run (`evaluation/report_test_v2_hf.json`).
- If the model's wording can't be verified against the passages but relevant passages were found,
  `hf` and `ollama` fall back to quoting the books' best sentences ("The books say: …") instead of
  refusing (`fallback_to_quotes`).

**Grounding check.** Small models sometimes skip citations, cite the wrong passage, or add
facts from outside the books. After the model answers, every sentence is compared with the
retrieved passages: a sentence is kept only if at least `min_support` (0.4) of its key words
appear in a passage, and it is cited to that passage. Sentences that no passage supports are
removed. In testing, made-up sentences scored 0.00 to 0.21 and correct paraphrases about 0.40
to 0.45. If nothing survives but relevant passages were found, the chatbot quotes the books'
own sentences instead (`fallback_to_quotes`); it refuses only when no relevant passage is found
or the model itself reports insufficient evidence. A meaning-based (embedding) check was
considered but not added: with the quote fallback it would mainly let through confident,
on-topic but wrong sentences.

**Streaming.** In the app, the model's draft appears word by word (dimmed) while it is written;
when it finishes, the checked answer with citations replaces it.

**Choices made on the validation split** (22 questions, 20 answerable; the test split was not used):

| Search setting | Right page in top 5 | Recall@5 | Extra time |
|---|---|---|---|
| Vector search only | 80% | 67% | — |
| **+ hybrid keyword search (used)** | **80%** | **67%** | under 0.01 s |
| + cross-encoder reranker | 75% | 62% | +1.1 s |
| + reranker + max 2 passages per book | 70% | 55% | +1.1 s |

Hybrid search ties with vector search but fixes exact-name questions (`train_test_split`,
confusion matrix), so it is kept. The reranker (trained on web search queries) and the per-book
cap lowered the score and are off.

| Ollama answer model | Mean time | Wrongly refused | Correct (own reading of 20 answers) |
|---|---|---|---|
| **`qwen2.5:1.5b` (used)** | **7.3 s** | **0%** | about 17 |
| `qwen2.5:3b` | 11.8 s | 25% | about 13 |

The 3B model reported "insufficient evidence" for five questions the books answer, so the
1.5B model is kept.

To try other settings without editing the main config, point `CHATBOT_CONFIG` at another YAML file.

## Evaluation dataset

`data/evaluation_questions.json` is a list of questions in this format:

```json
{
  "id": "Q001",
  "question": "What is a write-back cache?",
  "reference_answer": "A cache that writes modified blocks to memory only when they are replaced.",
  "type": "factual",
  "expected_evidence": [{"book_id": "G000_BOOK_003", "page": 412}],
  "answerable": true,
  "split": "test",
  "review_status": "reviewed",
  "reviewer": "name"
}
```

- `type`: `factual`, `reasoning`, `numerical`, `cross_book`, `no_answer` or `citation_check`.
- `split`: `validation` for tuning settings and prompts; `test` is held out until the final
  evaluation; `train` only if a model is fine-tuned.
- Questions whose answer is not in the books use `"answerable": false` and no expected evidence.
- Aim for 30–50 questions covering every type. Write each question by hand from a specific page,
  and keep near-duplicates in the same split so the test set doesn't leak.

**Human review.** Each evaluation run writes `evaluation/answers_for_review.json` with every
answer and the passages it cited. Reviewers record their verdicts in `evaluation/human_reviews.json`:

```json
{ "Q001": {"correct": true, "grounded": true, "citations_support": true, "reviewer": "name", "notes": ""} }
```

To review in Excel instead of editing JSON, export a sheet, fill in the yellow columns
(Yes/No drop-downs; a "How to review" tab explains each one), then import it:

```
python -m src.review_sheet export evaluation/answers_for_review.json
python -m src.review_sheet import evaluation/human_review.xlsx
```

The next run includes them as answer correctness and groundedness.
`python -m src.finish` (or double-click `finish_project.bat`) then runs every final step in order:
import the sheet, check the books, run the tests, evaluate Toodles Ultra (about 6 minutes),
and fill the correctness row of the results table below. Write the error analysis
and limitations into `error_analysis` and `limitations` in `evaluation/report.json`; re-runs keep them.

## Evaluation results

55 questions in `data/evaluation_questions.json`: 22 validation (used to choose settings) and
33 held-out test, covering factual, reasoning, numerical, cross-book, not-in-the-books and
citation-check questions.
They are drafts written from specific book pages and still need subject-expert review.

Final run on the 33 held-out test questions (v2), with the v1 run for comparison:

| Measure (automatic) | v1 (Hugging Face) | Toodles Instant | **Toodles Ultra** | Transformers 1.5B (removed) | Target |
|---|---|---|---|---|---|
| Right page in top 5 (hit rate) | 53.6% | 53.6% | **53.6%** | 53.6% | 80% |
| Recall@5 | 46.4% | 48.2% | **48.2%** | 48.2% | 80% |
| Citations resolve to a stored passage | 100% | 100% | **100%** | 100% | 100% |
| Cited pages match the expected pages | 31.8% | 18.8% | **48.0%** | 47.9% | report |
| Not-in-the-books questions refused | 100% | 60% | **60%** | 60% | report |
| Answerable questions wrongly refused | 21.4% | 0% | **10.7%** | 14.3% | report |
| Mean / p95 answer time | 48.6 / 89.3 s | 0.03 s | **5.0 / 9.4 s** | 60.5 / 101.2 s | report |
| Answer correctness (human review) | pending | pending | **pending** | pending | 80% |

Conditions: Intel i5-1135G7, 16 GB RAM, CPU only, plugged in. The Transformers copy ran in
bfloat16 (peak RAM 3.9 GB; index 59.5 MB). Reports: `evaluation/report.json` (v2, Toodles Ultra,
with error analysis and limitations), `report_test_v2_hf.json`, `report_test_v2_extractive.json`,
`report_v1_test.json`, and the validation runs `report_val_*.json`.

**Main findings.** Every displayed citation is traceable in all modes. The v2 changes cut wrong
refusals (21% to 11% with Toodles Ultra), made citations land on the expected pages more often
(32% to 48%), and Toodles Ultra is about 12 times faster than the Transformers copy. Two regressions: 2 of 5 off-topic
questions were answered (one misleading answer about AWS, one loose quote from the fallback),
and test-set retrieval did not improve (53.6%) even though validation reached 80%. These were
recorded, not tuned against the test set; candidate fixes are listed in the report.

## Features

- Hybrid search (meaning plus exact keywords) over 15 books, 21,630 passages
- Page-level citations that are checked against the stored passages before they are shown
- Answers stream word by word; Toodles Ultra gives answers in about 5 seconds
- Quotes the books instead of refusing when the model's wording can't be verified
- Refuses to answer when retrieval finds no relevant passage, the model reports insufficient
  evidence, or the model gives no checkable citation
- Detects scanned books and runs OCR when Tesseract is installed
- Ingestion report of empty, OCR'd and failed pages per book
- Runs fully offline on a CPU-only laptop once the models are downloaded

## Known limitations

- A passage never spans two pages, so an explanation split across a page break is retrieved as two passages.
- Repeated headers and notices are only removed when they appear on at least 20% of a book's pages;
  headers that change every chapter can still appear inside passages.
- Equations, tables and figures are extracted as plain text and may lose their layout.
- OCR needs [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) installed separately.
- The extractive backend quotes sentences rather than explaining, so it cannot combine
  information or work through calculations.
- On a CPU-only laptop (Intel i5-1135G7) the Hugging Face 1.5B model takes about 30 to 45 seconds
  per answer in float32 (50 to 70 seconds in bfloat16, which needs less RAM); Toodles Ultra takes
  about 5 seconds. Running on battery roughly halves CPU speed.
- The grounding check compares words, not meaning: a sentence that reuses the passage's words
  but changes their meaning can pass, and a correct paraphrase with different wording can be
  removed. Human review of answers is still needed.
- When the model's answer is only partly supported, the unsupported part is removed, and the
  quote fallback can give a choppy answer made of book sentences.
- The small models rarely carry out calculations (e.g. accuracy from a confusion matrix) even
  when the formula is in the retrieved passage.

## Books and copyright

The book PDFs, the extracted text and the index are excluded from Git (see `.gitignore`).
Share them only through the project's approved private storage. To rebuild the index from the
permitted sources, place the PDFs in `data/books/` and run `python -m src.pipeline`.

## Team roles

| Member | Role |
|---|---|
| | Books and manifest |
| | Ingestion and chunking |
| | Retrieval and generation |
| | Evaluation dataset and review |
| | Interface, tests and report |
