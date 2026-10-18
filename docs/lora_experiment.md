# Optional experiment: LoRA fine-tuning and adapter merging (task 7)

The retrieval chatbot already worked when this experiment began; fine-tuning is not needed for
retrieval. The experiment asks two questions:

1. **Can LoRA teach a smaller model the chatbot's job?** `Qwen2.5-0.5B-Instruct` is 3 times
   smaller than the 1.5B answer model. Out of the box it fails the citation format and falls
   back to quoting.
2. **Can two separately trained adapters be merged without losing performance?** This mirrors
   the project's plan to combine adapters from different groups.

**When it ran.** On 2026-10-18, with the v2.1 book set: books 014 and 015 were then the two
commercial titles that were replaced later that day (see the book record). The training data
and all the results below come from that earlier index. The experiment was not re-run after
the replacement because no adapter was adopted.

**Result in one line:** LoRA fixed the format problem, and the merge worked mechanically, but
the merged adapter scored below the separate adapters. Neither the adapters nor the merge beat
the 1.5B model, so **the chatbot keeps the 1.5B model and no adapter is adopted.**

## 1. Design

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0, 494 M parameters, tokenizer SHA-256 `c0382117…87539` |
| Why this base | Small enough to train on the target laptop's CPU, and the same family and tokenizer as the 1.5B answer model |
| What is learned | The answer format: 1–4 sentences built only from the numbered passages, `[n]` after every sentence, or `INSUFFICIENT_EVIDENCE` when the passages do not cover the question. The prompt is the chatbot's own (prompt version v2) |
| Simulated groups | **Adapter A** trained on books 001–007 and **adapter B** on books 008–015, standing in for two groups training separately |
| LoRA settings | rank r = 8, α = 16, dropout 0.05, on `q, k, v, o, gate, up, down` projections in all 24 blocks: 4,399,104 trainable parameters (0.89%) |
| Training | 2 epochs, AdamW, learning rate 2e-4, warm-up then cosine decay, 4 examples per update, loss on answer tokens only, float32, seed 42, CPU (Intel i5-1135G7) |
| Merge | PEFT `add_weighted_adapter`, method `cat`, weights 0.5 + 0.5, giving *W = W₀ + 0.5 ΔW_A + 0.5 ΔW_B* exactly (rank 16) |
| Decided in advance | The base, settings, the two groups and the 0.5/0.5 merge were fixed before any evaluation. All four models were then evaluated on validation and test, and nothing was tuned on the results |

Code: `src/lora_data.py` builds the training data, and `src/lora.py` trains, checks and merges.
`src/evaluate.py --backend hf --lora-base --adapter …` evaluates. Settings are in the `lora`
section of `config/config.yaml`.

## 2. Training data (`train` split)

The training examples were built from the books by `python -m src.lora_data`. No evaluation
question was used, and the steps below guard against leakage:

1. **Pick passages.** 520 clean prose passages were sampled, 260 per group. Every page within
   2 pages of any validation or test evidence page was excluded.
2. **Write questions.** The teacher model (`qwen2.5:1.5b`, the chatbot's own model) wrote one
   standalone question per passage. Questions mentioning "the passage", equation numbers, tables
   and so on were rejected.
3. **Leakage check.** Questions with cosine ≥ 0.85 to any validation or test question were
   dropped (4), as were near-duplicates of each other (0).
4. **Check retrieval.** Evidence was retrieved exactly as in the chatbot. The example was kept
   only if the source passage was among the 3 passages the model reads.
5. **Write the target answer.** The teacher answered with the chatbot's prompt, and the answer
   was kept only if the grounding check kept at least 80% of it, with citations, and did not
   break it into fragments.
6. **Add refusals.** A quarter of each group's questions became `INSUFFICIENT_EVIDENCE`
   examples. Each was paired with the passages retrieved for a question from another book,
   chosen so they were related but below the evidence threshold (cosine 0.60–0.64).

| Filter | Removed |
|---|---|
| Question unusable | 111 |
| Too close to an evaluation question | 4 |
| Source passage not retrieved | 100 |
| Teacher declined | 5 |
| Answer not grounded | 39 |
| Answer fragmented or formula-heavy | 21 |
| **Kept** | **240** |

The 240 examples split as group A, 86 answers + 29 refusals, and group B, 94 + 31.
`data/processed/lora_train.jsonl` (SHA-256 `da5a5b53…c4014`) quotes the books and is not in
Git; the summary is in `artifacts/lora/train_data_summary.json`.

## 3. Training record

| | Adapter A | Adapter B |
|---|---|---|
| Training / held-out examples (skipped: too long) | 105 / 9 (1) | 114 / 10 (1) |
| Answer / refusal examples in training | 78 / 27 | 85 / 29 |
| Optimizer updates | 54 | 58 |
| Held-out loss before → epoch 1 → epoch 2 | 0.615 → 0.380 → **0.371** | 0.617 → 0.419 → **0.412** |
| Time, peak RAM | 117 min, 3.1 GB | 137 min, 3.0 GB |
| Adapter file | 17.6 MB | 17.6 MB |

The held-out loss fell sharply in epoch 1 and only slightly in epoch 2, without rising, so
there is no sign of overfitting. Full records, including the per-update loss, base-model
details, data hash and software versions (torch 2.14.1+cpu, transformers 5.20.0, peft 0.21.2),
are in `artifacts/lora/lora_A/training_record.json` and `lora_B/training_record.json`.

## 4. Compatibility check before merging

`python -m src.lora check artifacts/lora/lora_A artifacts/lora/lora_B` compares everything that
decides whether two adapters can be combined:

| Must match (else merging is refused) | A vs B |
|---|---|
| Base model and exact revision | same |
| Architecture (`Qwen2ForCausalLM`) | same |
| Tokenizer (SHA-256 of `tokenizer.json`) | same |
| Adapter type (LoRA), task type | same |
| Adapted layers (target modules) | same |
| DoRA setting | same |

| Warn only | A vs B |
|---|---|
| Rank, scaling (α, rsLoRA) | same (only `cat` or `svd` merges apply if ranks differ) |
| Prompt version | same (v2) |

**Result: compatible, no warnings.** Automated tests cover the refusals: adapters on different
base models, revisions, architectures, tokenizers or layers are blocked. An adapter also cannot
be loaded onto a different base model than the one it was trained on. Adapters from other
groups were not available, so the same check is ready for them.

**Merge correctness.** The merged update was compared with *0.5 ΔW_A + 0.5 ΔW_B* computed
directly from the two adapters' weights, and they agree to within about 10⁻¹¹. With unequal
weights (1.0 and 0.25) the result also equals the expected sum. The merged adapter is in
`artifacts/lora/merged_AB/` (rank 16, 35.2 MB) with `merge_record.json`.

## 5. Results

Same retrieval for every model (top 5, the model reads the top 3), the same prompt, grounding
check and quote fallback. The 0.5B models ran in float32 with Transformers on the CPU; the
1.5B reference is the official Toodles Ultra (4-bit, Ollama, GPU). Citations resolved to
stored passages in 100% of answers for every model.

### Validation (22 questions, 20 answerable)

| | 0.5B base | 0.5B + A | 0.5B + B | 0.5B + merged | 1.5B (Toodles Ultra) |
|---|---|---|---|---|---|
| Answers that fell back to quotes | 31.6% | **0%** | **0%** | 5.6% | 5.0% |
| Answer F1 vs reference (auto) | 0.266 | **0.340** | 0.322 | 0.310 | 0.352 |
| Answer similarity vs reference (auto) | 0.801 | **0.834** | 0.822 | 0.830 | 0.833 |
| Answerable questions wrongly declined | 5% (1) | 15% (3) | 10% (2) | 10% (2) | 0% |
| Not-in-the-books questions declined | 100% | 100% | 100% | 100% | 100% |
| Cited pages = expected pages | 47.4% | 47.1% | 58.3% | 58.3% | 57.5% |
| Mean / p95 time per answer | 14.1 / 25.6 s | 10.2 / 14.7 s | 12.4 / 20.1 s | 12.3 / 19.4 s | 6.4 / 8.2 s |

### Held-out test (33 questions, 28 answerable)

| | 0.5B base | 0.5B + A | 0.5B + B | 0.5B + merged | 1.5B (Toodles Ultra) |
|---|---|---|---|---|---|
| Answers that fell back to quotes | 46.7% | **0%** | 4.2% | **0%** | 7.1% |
| Answer F1 vs reference (auto) | 0.225 | 0.274 | **0.296** | 0.261 | 0.296 |
| Answer similarity vs reference (auto) | 0.783 | 0.821 | 0.803 | 0.817 | 0.822 |
| Answerable questions wrongly declined | **0%** | 28.6% (8) | 17.9% (5) | 25.0% (7) | 7.1% (2) |
| Not-in-the-books questions declined | 60% | **80%** | **80%** | **80%** | 60% |
| Cited pages = expected pages | 31.0% | 50.8% | 47.8% | 47.6% | 48.1% |
| Mean / p95 time per answer | 9.9 / 24.0 s | 10.3 / 24.3 s | 10.8 / 24.9 s | 9.1 / 18.4 s | 6.5 / 9.9 s |
| Peak RAM | 2.8 GB | 2.8 GB | 2.8 GB | 2.8 GB | 0.8 GB + 1.4 GB in Ollama |

Reports: `evaluation/report_{val,test}_{lora_base,lora_A,lora_B,merged_AB}.json`;
1.5B: `report_val_ollama15_gpu.json` and `report.json`. Answer correctness has not been
reviewed by people for any model, so the F1 and similarity columns are estimates only.

### Merge study by book group

Questions are grouped by the books their expected evidence comes from. The test set has 17
questions on books 001–007 and 10 on books 008–015; validation has 9 and 9. Mean F1 is over
answered questions, with wrongly declined questions in brackets:

| | Adapter A (books 1–7) | Adapter B (books 8–15) | Merged A+B |
|---|---|---|---|
| Validation, books 1–7 questions | **0.417** (1) | 0.381 (0) | 0.372 (0) |
| Validation, books 8–15 questions | **0.278** (1) | 0.264 (1) | 0.258 (1) |
| Test, books 1–7 questions | 0.294 (4) | **0.298** (2) | 0.273 (3) |
| Test, books 8–15 questions | 0.237 (3) | **0.292** (2) | 0.238 (3) |

## 6. Findings

1. **LoRA fixed what it was trained to fix.**
   - The base 0.5B model's own wording failed the grounding check so often that 31.6%
     (validation) and 46.7% (test) of its answers fell back to quotes. With either adapter this
     fell to 0–4%.
   - Automatic answer scores rose on both splits; on test, F1 went from 0.225 to 0.274–0.296.
   - Declining not-in-the-books questions improved from 60% to 80% on test: Q038 (Python 3.13)
     is now declined instead of answered with an unrelated quote.
2. **It also learned to decline too readily.**
   - Wrongly declined answerable questions rose from 0% to 18–29% on test.
   - This is mostly a retrieval effect. Of the adapters' wrong refusals on test, 7 of 8 (A),
     5 of 5 (B) and 6 of 7 (merged) are questions whose expected page was not among the 3
     passages the model reads. That is true of 14 of the 28 answerable test questions,
     including all cross-book and citation-check questions.
   - The refusal examples taught "decline when the passages don't support an answer". With
     imperfect retrieval, that costs answers the base model gave from partly relevant passages.
     Fewer refusal examples, or refusal examples built from clearly unrelated passages, are the
     next things to try on validation.
3. **The adapters do not beat the 1.5B model.**
   - Adapter B matches its automatic F1 on test (0.296), but declines 5 answerable questions
     against 2.
   - The automatic scores also miss meaning. On test question Q016, adapter A wrote that without
     weight decay "the training error increases but the validation error decreases", which is
     the reverse of the reference answer (training error falls while validation error does not)
     and of the 1.5B model's answer. It still scored well on similarity.
   - Fluent but wrong answers like this are why human review is required before any adoption.
4. **Merging worked mechanically but lost performance.**
   - The compatibility check passed, and the merged weights are the exact weighted sum.
   - The merged adapter kept the format fix (0% quote fallback on test) and the better
     declining of off-topic questions.
   - But on both splits its F1 is below the better of the two separate adapters in **every**
     book group (table above), and on test it wrongly declined more questions (7) than adapter B
     (5).
   - Each adapter was also not clearly best on "its own" books: on test, B scored as well as A
     even on books 1–7 (F1 0.298 vs 0.294, with 2 wrong refusals against 4). The adapters mostly learned the answer format, not book-specific
     knowledge, so merging had little to combine and some to lose.
   - Following the project rule, *do not adopt a merged model unless comparison shows benefit
     without unacceptable regressions*, **the merged adapter is not adopted**, and the separate
     adapters and the baseline are kept.
5. **Speed.**
   - The 0.5B models answer in about 10 s on the CPU in float32. That is about 6 times faster
     than the 1.5B Transformers reference on the same CPU (60.5 s), but slower than Toodles
     Ultra on the GPU (6.5 s).
   - A 4-bit Ollama build of a merged 0.5B model would likely be faster still. It was not tested.

**Small samples.** Validation has 20 answerable questions and test has 28. One question moves a
rate by 5 points (validation) or 3.6 points (test), so differences of one or two questions
between adapters are not meaningful on their own.

## 7. Reproduce

```bash
python -m src.lora_data --per-group 260                    # about 70 minutes (teacher via Ollama)
python -m src.lora train --group A --name lora_A           # about 2 hours on the CPU
python -m src.lora train --group B --name lora_B
python -m src.lora check artifacts/lora/lora_A artifacts/lora/lora_B
python -m src.lora merge artifacts/lora/lora_A artifacts/lora/lora_B --weights 0.5 0.5 --method cat --name merged_AB
python -m src.evaluate --split validation --backend hf --lora-base --dtype float32 --name val_lora_base
python -m src.evaluate --split validation --backend hf --lora-base --dtype float32 --adapter artifacts/lora/lora_A --name val_lora_A
```

The last command is repeated for `lora_B` and `merged_AB`, and all four for `--split test`.

## 8. Artifacts and sharing

| Path | Contents | In Git |
|---|---|---|
| `artifacts/lora/{lora_A,lora_B}/adapter_config.json`, `training_record.json` | Configuration, base-model revision, licence, training record | yes |
| `artifacts/lora/merged_AB/adapter_config.json`, `merge_record.json` | Merge sources, weights, method, compatibility check | yes |
| `artifacts/lora/train_data_summary.json` | Training data rules and filter counts | yes |
| `artifacts/lora/*/adapter_model.safetensors` | Adapter weights (17.6 MB, merged 35.2 MB) | **no** |
| `data/processed/lora_train.jsonl` | Training examples (quote the books) | **no** |

The adapters were trained on passages from the group's books. Their weights are therefore
shared only through the project's private storage, like the index. The base model is not
duplicated: it is downloaded from Hugging Face at the pinned revision.
