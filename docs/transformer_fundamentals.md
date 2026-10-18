# Transformer fundamentals, on our own models

These notes cover the six topics of task 3. Each one is shown on the models TOODLES actually
uses, not on textbook toy examples. The numbers come from `python -m src.fundamentals`, which
writes them to [`fundamentals_results.json`](fundamentals_results.json). It takes about a minute
on the target laptop (Intel i5-1135G7, CPU only).

| Short name | Model | Job in TOODLES |
|---|---|---|
| **bge-small** | `BAAI/bge-small-en-v1.5` @ `5c38ec7…` | Embedding model: turns text into vectors for search |
| **Qwen-1.5B** | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79…` | Generative model: writes the answers |
| **Qwen-0.5B** | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576…` | Base model of the optional LoRA experiment |

---

## 1. Tokenization and embeddings

A Transformer never sees letters or words. A **tokenizer** splits text into pieces from a fixed
vocabulary and replaces each piece with its number (token ID). The model's first layer, the
**embedding table**, turns each ID into a learned vector. Our two models tokenize differently.

The sentence *"Ridge regression shrinks the coefficients of a linear model towards zero."*:

| | bge-small | Qwen-1.5B |
|---|---|---|
| Method | WordPiece, lower-cased | byte-level BPE, case kept |
| Vocabulary | 30,522 | 151,665 tokens (embedding table of 151,936 rows) |
| Tokens | `[CLS] ridge regression shrink ##s the coefficients of a linear model towards zero . [SEP]` | `R idge ␣regression ␣shr inks ␣the ␣coefficients ␣of ␣a ␣linear ␣model ␣towards ␣zero .` |
| Count | 15 (including the special `[CLS]` and `[SEP]`) | 14 |

- WordPiece marks a piece that continues a word with `##` (`shrink ##s`). BPE stores the space as
  part of the token (`␣regression`), and it can spell anything from bytes, so no word is ever
  "unknown".
- The same text costs a different number of tokens in each model. Model limits are counted in
  tokens, not words.

**What this means for TOODLES (measured on 889 sampled passages).** A passage averages 171.5
words, which is **258.3 bge tokens** and 267.7 Qwen tokens. Passages are embedded with a limit of
256 tokens (`embedding.max_seq_length`), chosen because 512 made indexing about 3 times slower.
As a result, **50% of passages are cut off** when embedded: their last few sentences do not
count towards vector search. BM25 keyword search still reads the whole passage, which partly
makes up for it. This is a documented limitation, and a candidate fix for the next version (see
the technical documentation).

**Embeddings of whole texts.** bge-small turns a whole question or passage into one 384-number
vector: the final state of the `[CLS]` token, normalised to length 1. Similar meanings give
vectors pointing the same way, and the **cosine similarity** (the dot product of unit vectors)
measures how close they are. Here are three questions against four sentences:

| Question ↓ / sentence → | Ridge penalty | Kernel trick | Gradient descent | World Cup final |
|---|---|---|---|---|
| How does ridge regression reduce overfitting? | **0.751** | 0.587 | 0.599 | 0.324 |
| What is the kernel trick in SVMs? | 0.499 | **0.721** | 0.589 | 0.263 |
| Who won the football world cup in 1998? | 0.292 | 0.332 | 0.305 | **0.564** |

- Each question scores highest with the sentence on its topic.
- Only the two machine learning questions reach the chatbot's evidence threshold of **0.67**.
  The football question's best match is 0.564, so the chatbot would decline. This is how
  `retrieval.min_score` makes the chatbot abstain.
- bge-small expects a short instruction in front of search queries. With it, the gap between
  the right sentence and the next best is 0.152 for the ridge question; without it, 0.113. The
  instruction makes matches clearer even though raw scores drop a little.

## 2. Self-attention

Self-attention lets every token build its new representation as a weighted mix of all the
tokens it can see. Each token's vector *h* is projected three ways: a **query** *Q = hW_Q*, a
**key** *K = hW_K* and a **value** *V = hW_V*. Then:

  **Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V**

- *Q Kᵀ* scores how relevant each token is to each other token.
- Dividing by √*d* (*d* = head size) stops the scores growing with the vector length.
- Softmax turns each row of scores into weights that add up to 1.
- The output is the weighted sum of the value vectors.

**Multi-head** attention does this several times in parallel with different projections. Each
head can track a different kind of relationship, and the heads' outputs are joined back together.

**Checked by hand.** `src/fundamentals.py` takes bge-small's first layer and the sentence *"The
model overfits because it memorises the noise in the training data."* (19 tokens). It computes
*Q* and *K* from the layer's own weights, splits them into **12 heads of size 32**, and applies
the formula. The result matches the attention weights reported by the library to within
**1.8 × 10⁻⁷**, which is floating-point rounding, and every row sums to 1.

In the last layer, averaged over heads, the token **"it" pays most attention to "model" (0.121)**,
then "memo" (from "memorises", 0.111). The model has linked the pronoun to what it refers to.
Nobody programmed this; it was learned during pretraining.

## 3. Transformer architecture

Both models stack identical **Transformer blocks**. Each block has:

1. a multi-head **self-attention** layer (tokens exchange information);
2. a **feed-forward** network applied to each token separately (most parameters live here);
3. **residual connections** around both, so each layer adds to the representation instead of
   replacing it, and **normalisation** to keep the values stable;
4. **position information**, because attention by itself ignores word order.

| | bge-small (encoder) | Qwen-1.5B (decoder) | Qwen-0.5B (decoder) |
|---|---|---|---|
| Class | `BertModel` | `Qwen2ForCausalLM` | `Qwen2ForCausalLM` |
| Blocks | 12 | 28 | 24 |
| Hidden size | 384 | 1,536 | 896 |
| Attention heads (key/value heads) | 12 | 12 (2) | 14 (2) |
| Feed-forward size | 1,536 | 8,960 | 4,864 |
| Vocabulary | 30,522 | 151,936 | 151,936 |
| Max positions | 512 | 32,768 | 32,768 |
| Parameters | 33.4 M | 1,543.7 M | 494.0 M |
| Positions | learned position embeddings | rotary embeddings (RoPE) | RoPE |
| Normalisation, activation | LayerNorm, GELU | RMSNorm, SwiGLU | RMSNorm, SwiGLU |

Qwen uses **grouped-query attention**: its 12 query heads share only 2 key/value heads. This
makes the key/value cache 6 times smaller, which matters when generating on a laptop.

## 4. Pretrained models and inference

Both models are **pretrained**: they learned general language from very large text collections
before we downloaded them. We use them as they are. Every version is pinned to an exact commit
hash, so results can be reproduced.

**Inference** is running the trained model forward without changing its weights. A generative
model predicts a probability for every token in its vocabulary as the next token. For the prompt
*"In linear regression, the coefficients are usually estimated by the method of least"* (14
tokens), Qwen-1.5B (bfloat16, CPU) predicts:

| Next token | Probability |
|---|---|
| ` squares` | **0.990** |
| ` square` | 0.0036 |
| `-s` | 0.0019 |
| ` absolute` | 0.0013 |
| ` mean` | 0.0005 |

**Decoding** chooses the tokens one at a time. TOODLES uses **greedy decoding** (always pick the
most likely token, temperature 0), so the same question always gives the same answer. Here the
continuation is *" squares. This means that we minimize the sum of the squared"*. Each new
token needs another forward pass. On this laptop that is **about 0.4 s per token** for the
full-precision 1.5B model, which is why the 4-bit Ollama copy (about 6–9 s per whole answer) is
used in the app.

A chatbot answer costs one pass over the whole prompt (instructions plus 3 passages, about
1,040 tokens on average) and then one pass per answer token. Prompt length mainly decides the
answer time on a CPU, so the model reads only the top 3 of the 5 retrieved passages.

## 5. Embedding models vs generative models

Both are Transformers, but they are built and used differently:

| | Embedding model (bge-small) | Generative model (Qwen) |
|---|---|---|
| Blocks used | Encoder only | Decoder only |
| Attention mask | **Bidirectional:** every token sees the whole text | **Causal:** a token sees only itself and earlier tokens |
| Measured | average attention weight on *later* tokens = 0.037 (non-zero: it looks ahead) | largest attention weight on any later token = **0.0** exactly |
| Output | one 384-number vector per text | a probability for each of 151,936 tokens, one token at a time |
| Trained to | place texts with the same meaning close together | predict the next token, then follow instructions |
| Job in TOODLES | search: compare the question with 22,216 passages in about 55 ms | write a cited answer from 3 passages |

A generative model must not peek at future tokens, because it has to write them. An embedding
model reads the whole text at once to understand it. Retrieval-augmented generation (RAG) uses
each where it is strong: the cheap encoder narrows 22,216 passages down to 5, and the expensive
decoder reads only those.

## 6. Fine-tuning and LoRA

**Fine-tuning** continues training a pretrained model on task-specific examples. **Full
fine-tuning** updates every weight. That needs memory for the weights, their gradients and two
optimizer values per weight, about 16 bytes per parameter in float32, or 7.9 GB for Qwen-0.5B
and 24.7 GB for Qwen-1.5B. It also produces a whole new copy of the model.

**LoRA** (low-rank adaptation) freezes the pretrained weights *W₀* and learns a small update
for selected weight matrices:

  **W = W₀ + (α / r) · B A**,  with *A* of size *r × inputs* and *B* of size *outputs × r*

Only *A* and *B* train. *B* starts at zero, so training starts from exactly the pretrained model.
With rank *r* = 8 on all seven projection matrices of every block (`q, k, v, o, gate, up, down`):

| Base model | All parameters | LoRA parameters (r = 8) | Share | Adapter file (float32) | Full model file (bfloat16) |
|---|---|---|---|---|---|
| Qwen-0.5B | 494.0 M | 4.40 M | 0.89% | 16.8 MB | 942 MB |
| Qwen-1.5B | 1,543.7 M | 9.23 M | 0.60% | 35.2 MB | 2,944 MB |

- The adapter is small enough to share and swap.
- Several adapters can be kept for one base model.
- Adapters from different groups can be combined if they share the same base model, revision,
  tokenizer and adapted layers. The combined update is *W₀ + Σ wᵢ ΔWᵢ*.

**Fine-tuning is not needed for retrieval:** the books are found by the embedding model, which
is not trained at all. Our optional experiment
([lora_experiment.md](lora_experiment.md)) uses LoRA for something else: teaching the small
Qwen-0.5B the chatbot's answer format, which it fails to follow out of the box. It also tests
whether two separately trained adapters can be merged without losing accuracy.

---

### Reproduce

```bash
python -m src.fundamentals
```

The run needs the three models in the Hugging Face cache, or internet access the first time.
