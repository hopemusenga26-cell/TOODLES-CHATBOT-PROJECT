# Selected models and their roles

TOODLES uses two pretrained Transformer models with separate jobs. An **embedding model**
finds the relevant passages and a **generative model** writes the answer from them. A
keyword search and a citation check sit between the two, and neither of them is a neural model.

```
question ──► embedding model ──► vector ─┐
                                         ├─► FAISS (meaning) + BM25 (keywords) ──► top 5 passages
passages ──► embedding model ──► vectors ┘                                              │
                                                   evidence check (cosine ≥ 0.67) ◄─────┘
                                                              │
                                    generative model reads the top 3 passages
                                                              │
                       grounding + citation check ──► answer with [n] citations, or "insufficient evidence"
```

## 1. Embedding model: `BAAI/bge-small-en-v1.5`

| | |
|---|---|
| Role | Converts every passage (once, when the index is built) and every question (at query time) into a 384-number vector in the same space, so passages that mean the same thing as the question are close together |
| Revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (pinned in `config/config.yaml`) |
| Licence | MIT |
| Architecture | Encoder-only Transformer (BERT): 12 layers, hidden size 384, 12 attention heads, 33.4 M parameters, WordPiece vocabulary of 30,522 tokens |
| Library | Sentence Transformers 6.1.0 (`src/embeddings.py`) |
| Settings | 256 tokens per passage (the model accepts 512), vectors normalised to length 1, BGE's query instruction `"Represent this sentence for searching relevant passages: "` in front of questions only |
| Size and speed | about 130 MB download; embeds a question in about 30 ms on the CPU (a full hybrid search takes about 55 ms); the 22,216-passage index is 61.2 MB |
| Also used for | The evidence threshold. If no retrieved passage has cosine similarity of at least 0.67 with the question, the chatbot declines. It also gives the automatic answer-similarity score in the evaluation |

**Why this model.** It is small enough for a CPU-only laptop and ranks well for its size on
retrieval benchmarks. It replaced `sentence-transformers/all-MiniLM-L6-v2` (revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, used in v1) after validation testing. If the
coordinator publishes a project-wide embedding model, it goes in `embedding.model_name` and
`embedding.revision`, and the index is rebuilt (`python -m src.build_index`). The app refuses to
search an index that was built with a different model.

## 2. Generative model: `Qwen/Qwen2.5-1.5B-Instruct`

| | |
|---|---|
| Role | Reads the question and the 3 best passages, and writes a 1 to 4 sentence answer that cites a passage number after every sentence. When the passages do not cover the question it writes only `INSUFFICIENT_EVIDENCE` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Hugging Face; pinned) |
| Licence | Apache-2.0 |
| Architecture | Decoder-only Transformer (Qwen2): 28 layers, hidden size 1,536, 12 query heads sharing 2 key/value heads (grouped-query attention), SwiGLU feed-forward of 8,960, RoPE positions, RMSNorm, 1.54 B parameters, byte-level BPE vocabulary of 151,936 tokens, 32,768-token context |
| How it runs: **Toodles Ultra** (default, official evaluation) | The same model in a 4-bit format (`qwen2.5:1.5b`, Q4_K_M, 0.92 GB, digest `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`) served by Ollama 0.40.5. About 6–9 s per answer (8.4 s mean on the v3 test run, 6.5 s on v2.1) and 1.4 GB of memory; on a GeForce MX450 (2 GB) the whole model fits on the GPU with a 2,048-token context |
| How it runs: Transformers reference | The full-precision weights at the pinned revision, loaded with Hugging Face Transformers 5.20.0 in bfloat16 on the CPU (`generation.backend: hf`). Same accuracy on the test set, about 60 s per answer, 3.9 GB peak RAM. Kept as the reproducible reference run |
| Decoding | Greedy (temperature 0), at most 200 new tokens |

**Why this model.** It is the smallest instruction-tuned model that followed the citation format
in testing:

| Candidate | Result | Decision |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (rev `7ae557604adf67be50417f59c2c2f167def9a775`) | Replied with bare citation numbers, or refused questions the passages answer | Too small as is; became the base for the LoRA experiment ([lora_experiment.md](lora_experiment.md)) |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | Usable answers; on 20 validation questions about 17 correct, 0% wrongly refused | **Selected** |
| `qwen2.5:3b` (Ollama, digest `357c53fb659c…`) | 25% of answerable validation questions wrongly refused; 11.8 s vs 7.3 s per answer | Rejected |

## 3. Supporting components (not neural generators)

| Component | Role | Status |
|---|---|---|
| BM25 keyword search (`src/keyword_search.py`) | Finds exact terms the embedding model can miss (function names such as `train_test_split`); merged with vector search by reciprocal rank fusion | On |
| FAISS `IndexFlatIP` (faiss-cpu 1.15.1) | Exact inner-product search over the normalised vectors, which equals cosine similarity | On |
| Cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L6-v2` (rev `233902d25c440f23af6f7d6e94d2946bac0bee0a`, Apache-2.0) | Re-scores question–passage pairs jointly | Off: lowered validation hit rate from 80% to 75% and added 1.1 s |
| Extractive answerer (**Toodles Instant**) | Quotes the passage sentences that share the most words with the question; no language model | Baseline and instant mode |
| Grounding and citation check (`src/generate.py`, `src/citations.py`) | Keeps only answer sentences whose key words appear in a cited passage (at least 40%) and checks that every citation resolves to a stored passage | On |

## 4. Embedding model vs generative model

| | Embedding model (bge-small) | Generative model (Qwen2.5) |
|---|---|---|
| Transformer type | Encoder only | Decoder only |
| Attention | Bidirectional: every token sees every other token | Causal: each token sees only earlier tokens |
| Output | One vector per text (the `[CLS]` token's final state, normalised) | A probability for every vocabulary token, produced one token at a time |
| Used for | Comparing meaning (search) | Writing text |
| Cost per question | One forward pass over about 20 tokens | One pass over the prompt (about 1,040 tokens on average, 750 to 1,380), then one pass per answer token |

The fundamentals experiments in [transformer_fundamentals.md](transformer_fundamentals.md) show
these differences on the actual models.

## 5. Approval status

| Model | Coordinator approval |
|---|---|
| `BAAI/bge-small-en-v1.5` @ `5c38ec7…` | **Pending.** The coordinator has not yet published a project-wide model. Replace this one if a different model is published |
| `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79…` / `qwen2.5:1.5b` Q4_K_M | **Pending** |
| `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576…` + LoRA adapters | Optional experiment only, not used by the chatbot |

Record the approval date here once the coordinator signs off.
