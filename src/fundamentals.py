"""Transformer fundamentals, shown on the project's own models (task 3).

Runs seven small experiments and saves the numbers to docs/fundamentals_results.json;
docs/transformer_fundamentals.md explains them.

    1. tokenization        how the embedding model and the answer model split text into tokens
    2. embeddings          questions and sentences as vectors; cosine similarity; BGE's query instruction
    3. self-attention      layer-1 attention of the embedding model recomputed by hand from Q, K and V
    4. architecture        sizes of the encoder (embedding) and decoder (answer) models
    5. inference           next-token probabilities and greedy decoding with the answer model
    6. embedding vs generative   bidirectional vs causal attention masks
    7. fine-tuning and LoRA      trainable parameters for full fine-tuning vs LoRA

Usage:
    python -m src.fundamentals            # about 1 minute on a CPU laptop, offline once models are cached
"""
from __future__ import annotations

import math
import statistics
import sys
import time

from src.config import load_config, read_jsonl, resolve, write_json

OUTPUT = "docs/fundamentals_results.json"
SENTENCE = "Ridge regression shrinks the coefficients of a linear model towards zero."
QUESTIONS = ["How does ridge regression reduce overfitting?",
             "What is the kernel trick in support vector machines?",
             "Who won the football world cup in 1998?"]
SENTENCES = ["Ridge regression adds a penalty on the squared size of the coefficients, which shrinks them and lowers variance.",
             "A kernel computes inner products in a high-dimensional feature space without constructing the features.",
             "Gradient descent updates the parameters in the direction of the negative gradient.",
             "The final match was played at the Stade de France in Saint-Denis."]
ATTENTION_SENTENCE = "The model overfits because it memorises the noise in the training data."
PROMPT = "In linear regression, the coefficients are usually estimated by the method of least"


def lora_parameters(config, r: int, modules: list[str]) -> int:
    """Trainable LoRA parameters: r * (inputs + outputs) for every adapted matrix in every layer."""
    h, ffn = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * (h // config.num_attention_heads)
    shapes = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
              "gate_proj": (h, ffn), "up_proj": (h, ffn), "down_proj": (ffn, h)}
    return config.num_hidden_layers * sum(r * (shapes[m][0] + shapes[m][1]) for m in modules)


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    e, g, lc = cfg["embedding"], cfg["generation"], cfg["lora"]
    results: dict = {"models": {"embedding": f"{e['model_name']}@{e['revision']}",
                                "generation": f"{g['model_name']}@{g['revision']}",
                                "lora_base": f"{lc['base_model']}@{lc['base_revision']}"}}

    # 1. Tokenization -------------------------------------------------------------------
    enc_tok = AutoTokenizer.from_pretrained(e["model_name"], revision=e["revision"])
    dec_tok = AutoTokenizer.from_pretrained(g["model_name"], revision=g["revision"])
    enc_ids = enc_tok(SENTENCE)["input_ids"]
    dec_ids = dec_tok(SENTENCE)["input_ids"]
    passages = read_jsonl(resolve(cfg["paths"]["index_dir"]) / "passages.jsonl")
    sample = passages[::25]
    enc_lengths = [len(enc_tok(p["text"])["input_ids"]) for p in sample]
    dec_lengths = [len(dec_tok(p["text"])["input_ids"]) for p in sample]
    words = [len(p["text"].split()) for p in sample]
    results["tokenization"] = {
        "sentence": SENTENCE,
        "embedding_model": {"method": "WordPiece (BERT vocabulary, lower-cased)", "vocab_size": enc_tok.vocab_size,
                            "tokens": enc_tok.convert_ids_to_tokens(enc_ids), "ids": enc_ids},
        "answer_model": {"method": "byte-level BPE (Qwen2 vocabulary)", "vocab_size": len(dec_tok),
                         "tokens": [dec_tok.decode([i]) for i in dec_ids], "ids": dec_ids},
        "passages_sampled": len(sample),
        "words_per_passage_mean": round(statistics.mean(words), 1),
        "embedding_tokens_per_passage_mean": round(statistics.mean(enc_lengths), 1),
        "answer_model_tokens_per_passage_mean": round(statistics.mean(dec_lengths), 1),
        "share_of_passages_longer_than_embedding_limit": round(
            sum(n > e["max_seq_length"] for n in enc_lengths) / len(enc_lengths), 3),
        "embedding_limit_tokens": e["max_seq_length"],
    }

    # 2. Embeddings ----------------------------------------------------------------------
    from src.embeddings import embedder_from_config

    embedder = embedder_from_config(cfg)
    q_vec = embedder.encode(QUESTIONS, kind="query")
    q_plain = embedder.model.encode(QUESTIONS, normalize_embeddings=True)
    s_vec = embedder.encode(SENTENCES, kind="passage")
    results["embeddings"] = {
        "dimension": int(s_vec.shape[1]),
        "vector_norm": round(float(np.linalg.norm(s_vec[0])), 4),
        "first_5_values_of_sentence_1": [round(float(x), 4) for x in s_vec[0][:5]],
        "questions": QUESTIONS,
        "sentences": SENTENCES,
        "cosine_with_query_instruction": [[round(float(x), 3) for x in row] for row in q_vec @ s_vec.T],
        "cosine_without_query_instruction": [[round(float(x), 3) for x in row] for row in q_plain @ s_vec.T],
        "evidence_threshold_min_score": cfg["retrieval"]["min_score"],
    }

    # 3. Self-attention, recomputed by hand -------------------------------------------------
    bert = AutoModel.from_pretrained(e["model_name"], revision=e["revision"], attn_implementation="eager").eval()
    batch = enc_tok(ATTENTION_SENTENCE, return_tensors="pt")
    with torch.no_grad():
        out = bert(**batch, output_attentions=True, output_hidden_states=True)
        attn = bert.encoder.layer[0].attention.self
        h = out.hidden_states[0]                      # token + position embeddings after LayerNorm
        T, H, d = h.shape[1], attn.num_attention_heads, attn.attention_head_size

        def heads(x):
            return x.view(1, T, H, d).transpose(1, 2)  # (1, heads, tokens, head size)

        q, k = heads(attn.query(h)), heads(attn.key(h))
        manual = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)
    tokens = enc_tok.convert_ids_to_tokens(batch["input_ids"][0])
    library = out.attentions[0]
    last = out.attentions[-1][0].mean(0)              # last layer, averaged over heads
    focus = tokens.index("it")
    top = torch.topk(last[focus], 4)
    results["self_attention"] = {
        "sentence": ATTENTION_SENTENCE,
        "tokens": tokens,
        "layers": len(out.attentions),
        "heads_per_layer": H,
        "head_size": d,
        "formula": "softmax(Q K^T / sqrt(d)) V, with Q = h W_q, K = h W_k, V = h W_v",
        "max_difference_manual_vs_library_layer1": float((manual - library).abs().max()),
        "rows_sum_to_one": bool(torch.allclose(library.sum(-1), torch.ones_like(library.sum(-1)), atol=1e-5)),
        "last_layer_top_attended_by_it": [{"token": tokens[i], "weight": round(float(w), 3)}
                                          for w, i in zip(top.values, top.indices)],
    }
    del bert

    # 4. Architecture ----------------------------------------------------------------------
    def describe(name: str, revision: str) -> dict:
        c = AutoConfig.from_pretrained(name, revision=revision)
        info = {"model": name, "architecture": (c.architectures or [c.model_type])[0], "layers": c.num_hidden_layers,
                "hidden_size": c.hidden_size, "attention_heads": c.num_attention_heads,
                "feed_forward_size": c.intermediate_size, "vocab_size": c.vocab_size,
                "max_positions": c.max_position_embeddings}
        if hasattr(c, "num_key_value_heads"):
            info["key_value_heads"] = c.num_key_value_heads
        return info

    results["architecture"] = {
        "embedding_model": {**describe(e["model_name"], e["revision"]),
                            "parameters": sum(p.numel() for p in embedder.model.parameters()),
                            "type": "encoder-only (BERT): bidirectional attention, CLS vector -> sentence embedding"},
        "answer_model": {**describe(g["model_name"], g["revision"]),
                         "type": "decoder-only (Qwen2): causal attention, RoPE positions, RMSNorm, SwiGLU feed-forward, "
                                 "grouped-query attention"},
        "lora_base_model": describe(lc["base_model"], lc["base_revision"]),
    }

    # 5. Pretrained inference and 6. causal mask -------------------------------------------
    lm = AutoModelForCausalLM.from_pretrained(g["model_name"], revision=g["revision"], dtype=torch.bfloat16,
                                              attn_implementation="eager").eval()
    results["architecture"]["answer_model"]["parameters"] = sum(p.numel() for p in lm.parameters())
    batch = dec_tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        start = time.perf_counter()
        out = lm(**batch, output_attentions=True)
        forward_s = time.perf_counter() - start
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, 5)
        start = time.perf_counter()
        generated = lm.generate(**batch, max_new_tokens=12, do_sample=False)
        generate_s = time.perf_counter() - start
    new = generated[0][batch["input_ids"].shape[1]:]
    causal = out.attentions[0][0]                     # (heads, tokens, tokens)
    upper = torch.triu(torch.ones_like(causal[0], dtype=torch.bool), diagonal=1)
    results["inference"] = {
        "prompt": PROMPT,
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "next_token_top5": [{"token": dec_tok.decode([int(i)]), "probability": round(float(p), 4)}
                            for p, i in zip(top.values, top.indices)],
        "greedy_continuation": dec_tok.decode(new, skip_special_tokens=True),
        "forward_pass_s": round(forward_s, 2),
        "seconds_per_generated_token": round(generate_s / max(1, len(new)), 3),
        "dtype": "bfloat16",
    }
    results["embedding_vs_generative"] = {
        "answer_model_attention_above_diagonal_max": float(causal[:, upper].abs().max()),
        "embedding_model_attention_above_diagonal_mean": float(
            library[0][:, torch.triu(torch.ones_like(library[0][0], dtype=torch.bool), diagonal=1)].mean()),
        "embedding_model_output": f"one {results['embeddings']['dimension']}-number vector per text",
        "answer_model_output": f"a probability for each of {lm.config.vocab_size:,} tokens, one token at a time",
    }
    del lm

    # 7. Fine-tuning and LoRA ------------------------------------------------------------
    rows = []
    for name, revision, params in ((lc["base_model"], lc["base_revision"], None),
                                   (g["model_name"], g["revision"], results["architecture"]["answer_model"]["parameters"])):
        c = AutoConfig.from_pretrained(name, revision=revision)
        if params is None:
            params = sum(p.numel() for p in AutoModelForCausalLM.from_config(c).parameters())
        lora = lora_parameters(c, lc["r"], list(lc["target_modules"]))
        rows.append({"model": name, "full_fine_tuning_parameters": params, "lora_parameters": lora,
                     "lora_share": round(lora / params, 5), "lora_file_mb_float32": round(lora * 4 / 2**20, 1),
                     "full_model_file_mb_bfloat16": round(params * 2 / 2**20)})
    results["lora"] = {"r": lc["r"], "alpha": lc["alpha"], "target_modules": list(lc["target_modules"]),
                       "formula": "W = W0 + (alpha / r) * B A, with A: r x inputs, B: outputs x r; only A and B train",
                       "models": rows}

    write_json(results, resolve(OUTPUT))
    t, emb, sa, inf = results["tokenization"], results["embeddings"], results["self_attention"], results["inference"]
    print(f"1. '{SENTENCE}' -> {len(t['embedding_model']['ids'])} WordPiece tokens, "
          f"{len(t['answer_model']['ids'])} BPE tokens. Passages: {t['embedding_tokens_per_passage_mean']} tokens on average; "
          f"{t['share_of_passages_longer_than_embedding_limit']:.0%} exceed the {t['embedding_limit_tokens']}-token embedding limit.")
    print(f"2. {emb['dimension']}-number vectors; question 1 vs sentence 1 cosine {emb['cosine_with_query_instruction'][0][0]}, "
          f"off-topic question vs best sentence {max(emb['cosine_with_query_instruction'][2])}.")
    print(f"3. Hand-computed attention matches the library to {sa['max_difference_manual_vs_library_layer1']:.1e}.")
    print(f"5. Next token after '...least': {inf['next_token_top5'][0]}; {inf['seconds_per_generated_token']} s per token.")
    print(f"6. Causal mask: answer model attention above the diagonal = "
          f"{results['embedding_vs_generative']['answer_model_attention_above_diagonal_max']}.")
    print(f"7. LoRA trains {rows[0]['lora_parameters']:,} of {rows[0]['full_fine_tuning_parameters']:,} parameters "
          f"({rows[0]['lora_share']:.2%}) on the 0.5B model.")
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
