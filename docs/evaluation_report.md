# Evaluation report

This report covers deliverable 6, the test dataset and evaluation report, and task 6: does the
chatbot retrieve the right passages, give correct answers, cite the correct sources, and decline
when the books lack evidence?

The machine-readable report is [`evaluation/report.json`](../evaluation/report.json), produced by
`python -m src.evaluate --split test`. Every number below comes from the saved report files named
in each table.

## 1. Test dataset

`data/evaluation_questions.json` contains **55 questions**. Each was written by hand from a
specific book page and records the question, a reference answer, the type, the expected evidence
(book and PDF page), whether it is answerable, the split, and its review status.

| Type | What it tests | Validation | Test |
|---|---|---|---|
| factual | A fact stated on one page | 13 | 14 |
| reasoning | Combining or explaining information from a passage | 3 | 4 |
| numerical | A calculation or formula | 2 | 4 |
| cross_book | Comparing what two books say | 1 | 4 |
| no_answer | Not in the books (cloud deployment, medicine, geography, ...) | 2 | 5 |
| citation_check | Questions naming a book or exact formula, to catch wrong or invented sources | 1 | 2 |
| **Total** | | **22** (20 answerable) | **33** (28 answerable) |

**How the split was made and checked.**

- The **validation** split was used to choose every setting: embedding model, hybrid search,
  evidence threshold, grounding cutoff and answer model.
- The **test** split was held out and run only for the final evaluation. Nothing was changed
  after seeing test results; candidate fixes are listed below instead.
- **Leakage check.** The two splits contain no duplicate or near-duplicate questions. The
  closest pair has a bge-small cosine similarity of 0.835: Q006, *"main disadvantage of ridge
  compared with the lasso"*, and Q022, *"why can the lasso select variables while ridge
  cannot"*. These ask different things.
- The two splits do share **9 evidence pages**: different questions about the same page. This
  is a small risk that page-level tuning carries over.
- The LoRA experiment's **training** examples come from the books, excluding every page within
  2 pages of any validation or test evidence. Generated questions with cosine ≥ 0.85 to an
  evaluation question were dropped (4 were).
- Expected evidence covers 12 of the 15 books on the test split and 13 on validation. Most of
  it comes from *An Introduction to Statistical Learning* (book 002).

**Review status.** All 55 questions and reference answers are drafts (`review_status: draft`)
and still need review by a subject expert or instructor.

## 2. Test conditions

| | |
|---|---|
| Hardware | Intel Core i5-1135G7 (4 cores), 16 GB RAM, NVIDIA GeForce MX450 2 GB, Windows 11, plugged in |
| Official configuration (Toodles Ultra) | bge-small-en-v1.5 @ `5c38ec7…`, hybrid search, top 5, evidence threshold 0.67. Answers by `qwen2.5:1.5b` (Q4_K_M) in Ollama 0.40.5 on the GPU (`num_ctx` 2048), top 3 passages, grounding cutoff 0.4, quote fallback on |
| Index | 15 books, 8,340 PDF pages (8,218 with text, 0 failed, no OCR needed), 21,630 passages, 59.5 MB |
| Date of official run | 2026-10-18, test split; report `evaluation/report.json` |

## 3. Results on the held-out test set

| Measure | v1 (Transformers, MiniLM) | Toodles Instant (quotes only) | **Toodles Ultra (official)** | Transformers 1.5B reference | Target |
|---|---|---|---|---|---|
| Right page in top 5 (hit rate) | 53.6% | 53.6% | **53.6%** | 53.6% | 80% |
| Right page ±1 page in top 5 | – | 60.7% | **60.7%** | – | report |
| Recall@5 | 46.4% | 48.2% | **48.2%** | 48.2% | 80% |
| Citations that resolve to a stored passage | 100% | 100% | **100%** | 100% | 100% |
| Cited pages that are expected evidence pages | 31.8% | 18.8% | **48.1%** | 47.9% | report |
| Not-in-the-books questions declined | 100% (5/5) | 60% (3/5) | **60% (3/5)** | 60% (3/5) | report |
| Answerable questions wrongly declined | 21.4% | 0% | **7.1% (2/28)** | 14.3% | report |
| Answers given as quotes (fallback) | – | – | **7.1%** | – | report |
| Answer word-overlap F1 vs reference (auto) | – | 0.181 | **0.296** | – | report |
| Answer meaning similarity vs reference (auto) | – | 0.758 | **0.822** | – | report |
| Answer correctness (human review) | pending | pending | **pending** | pending | 80% |
| Groundedness (human review) | pending | pending | **pending** | pending | report |
| Mean / 95th-percentile answer time | 48.6 / 89.3 s | 0.03 / 0.03 s | **6.5 / 9.9 s** | 60.5 / 101.2 s | report |
| Peak RAM | 6.8 GB | – | **0.8 GB + about 1.4 GB in Ollama** | 3.9 GB | report |

Reports: `report_v1_test.json`, `report_test_v2_extractive.json`, `report.json` and
`report_test_v2_hf.json`. The v1 and Transformers runs were made before the automatic answer
scores and the ±1-page measure existed. On the 22 validation questions, Toodles Ultra found the
right page for 80% (85% within ±1 page), declined 2 of 2 not-in-the-books questions and 0
answerable ones, and scored F1 0.352 and similarity 0.833 (`report_val_ollama15_gpu.json`).

**Results by question type (Toodles Ultra, test):**

| Type | Questions | Right page in top 5 | Answered | Auto F1 |
|---|---|---|---|---|
| factual | 14 | 71% | 100% | 0.358 |
| reasoning | 4 | 50% | 100% | 0.275 |
| numerical | 4 | 75% | 50% | 0.291 |
| cross_book | 4 | 0% | 100% | 0.183 |
| citation_check | 2 | 0% | 100% | 0.138 |
| no_answer | 5 | – | 40% (should be 0%) | – |

### How to read the automatic answer scores

- **Word-overlap F1** counts the words an answer shares with the reference answer, ignoring
  case, articles and citation markers, as in the SQuAD benchmark. It undercounts correct
  paraphrases, so values around 0.3 are normal for correct free-text answers.
- **Meaning similarity** is the cosine similarity between the answer and the reference answer
  as embedded by bge-small (0 to 1).
- Both are computed only for answerable questions that received an answer. They are useful for
  comparing configurations on the same questions, but they are **not** a measure of
  correctness. Correctness comes from the human review below.

## 4. Task 6 checks, one by one

**Does it retrieve the right passages?** Partly. On the test set the expected page is in the
top 5 for 53.6% of answerable questions (60.7% within ±1 page), below the 80% target. The 80%
reached on validation did not carry over. Comparing the retrieved pages with the expected ones
shows four kinds of miss:

- **Questions that name a book** pull in that book's title, contents or preface pages, because
  the title words match. Q019 retrieved pages 1, 2 and 16 of book 015; Q032 retrieved pages 1–3
  of book 001.
- **Cross-book questions** (0 of 4 hits). In Q030 and Q032 one of the two books is missing. In
  Q031 and Q034 both books appear, but on other pages than the expected ones.
- **Relevant pages elsewhere.** Q015 (minibatch SGD) retrieved the optimisation chapter of the
  same book (pp. 540–550) rather than the expected introduction (pp. 126–127). Q021 retrieved
  passages on overfitting from three other books. Exact-page matching counts these as misses,
  even though they may answer the question.
- **Explanations crossing a page break.** Q024 and Q030 count as hits within ±1 page.

**Does it give correct answers?** This is not measured yet; it needs the human review. Running
`python -m src.review_sheet export evaluation/answers_for_review.json` creates the sheet for
reviewers. What the automatic signals show:

- Every answered factual and reasoning question shares content with its reference answer
  (similarity 0.73–0.94).
- The weakest answers are the quote fallbacks: Q042 quoted a formula from the wrong book.
- Numerical questions are a known weakness: Q025 (R²) and Q029 (correlation) were declined
  because the model does not do the arithmetic, even when the formula is retrieved.

**Does it cite the correct sources?**

- **100% of citations resolve** to a real stored passage with the same book and page. This is
  enforced in code: an answer with an unresolvable citation is replaced by a refusal.
- 48.1% of cited pages are exactly the expected evidence pages. Many of the others are
  relevant pages from other books.
- Whether each cited page actually supports its sentence is checked automatically by word
  overlap (at least 40% of a sentence's key words), and needs human confirmation.

**Does it decline when the books lack evidence?**

- 3 of 5 test questions outside the books were declined: ibuprofen dose, Kubernetes, capital
  of Uganda.
- Q035 (deploying on AWS SageMaker) was answered from *Dive into Deep Learning*'s appendix on
  renting AWS machines. The words match, so the grounding check passed a misleading answer.
- Q038 (Python 3.13 features) received an unrelated quote from the fallback, which needs only
  one shared keyword.
- Only 2 of 28 answerable questions were declined, both calculations.

## 5. Error analysis and candidate fixes

| Problem | Questions | Cause | Candidate fix (to be chosen on validation, not test) |
|---|---|---|---|
| Title, contents and preface pages retrieved | Q019, Q032 (also Q031: page 3 of book 002) | A book's name in the question matches its front matter | Leave front matter (title, contents, preface pages) out of the index, or down-weight it |
| Expected pages of a cross-book or citation-check question not retrieved | Q030, Q031, Q032, Q034, Q041, Q042 | The top 5 is dominated by one book or by pages matching the book's name | Search once per book named in the question, restricted to that book |
| Correct topic, "wrong" book or page | Q005, Q012, Q015, Q021 | Several chapters or books explain the same thing, but expected evidence lists one page | Complete `expected_evidence` with every page that answers the question (reviewer task) |
| Off-topic question answered | Q035, Q038 | Keyword overlap passes the grounding check; the quote fallback needs one shared word | Require several question keywords in a fallback quote; meaning-based check of the answer against the question |
| Calculation declined | Q025, Q029 | The 1.5B model does not compute, and a computed number fails the word-overlap check | Number-aware grounding that accepts values computed from numbers in the passages |
| Passage ends not searchable | (all) | 51% of passages exceed the 256-token embedding limit | `max_seq_length: 512` or shorter chunks; re-index; compare on validation |

## 6. Efficiency

- Toodles Ultra answers in 6.5 s on average (95th percentile 9.9 s) on the GPU. The earlier
  CPU run on 2026-10-17 averaged 5.0 s.
- The evaluation process peaks at 0.8 GB of RAM; Ollama uses about 1.4 GB on top of that. The
  index uses 59.5 MB of disk.
- Search takes about 55 ms per question.
- The Transformers reference took 60.5 s per answer and 3.9 GB of RAM, so it is not offered in
  the app.

## 7. Optional LoRA experiment

Full write-up: [lora_experiment.md](lora_experiment.md). Two LoRA adapters were trained on
`Qwen2.5-0.5B-Instruct`: A on books 1–7 and B on books 8–15. They passed the compatibility
check and were merged 50/50. All models below use the same retrieval and checks.

| Test (33 questions) | 0.5B base | 0.5B + A | 0.5B + B | 0.5B + merged | **1.5B (official)** |
|---|---|---|---|---|---|
| Answers that fell back to quotes | 46.7% | 0% | 4.2% | 0% | 7.1% |
| Answer F1 / similarity (auto) | 0.225 / 0.783 | 0.274 / 0.821 | 0.296 / 0.803 | 0.261 / 0.817 | **0.296 / 0.822** |
| Answerable wrongly declined | 0% | 28.6% | 17.9% | 25.0% | **7.1%** |
| Not-in-the-books declined | 60% | 80% | 80% | 80% | 60% |
| Mean time per answer | 9.9 s (CPU) | 10.3 s | 10.8 s | 9.1 s | **6.5 s (GPU)** |

- LoRA fixed the small model's format problem (quote fallbacks 46.7% → 0–4%) and improved
  declining of off-topic questions.
- It also declines too readily, mostly on questions where retrieval missed the expected page.
- The merged adapter scored below the better separate adapter in every book group, so merging
  cost performance.
- No adapter beats the official 1.5B model. **Neither the adapters nor the merged model is
  adopted.**

## 8. Human review (pending)

Answer correctness and groundedness have not been measured: no human verdicts have been
recorded yet (`evaluation/human_reviews.json` is empty). To complete them:

```bash
python -m src.review_sheet export evaluation/answers_for_review.json   # creates evaluation/human_review.xlsx
# reviewers fill in the yellow columns (correct, grounded, citations support the answer)
python -m src.finish                                                    # imports verdicts, re-runs the official evaluation
```

Until then, the answer-correctness target (80%) is **not demonstrated**, and the automatic
scores above must not be reported as correctness.
