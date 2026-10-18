"""Training data for the optional LoRA experiment (task 7).

Builds the training split from the books themselves, never from the evaluation questions:

    1. pick clean passages, skipping every page within 2 pages of a validation or test
       question's expected evidence
    2. a teacher model (the chatbot's own Ollama model) writes one question each passage answers
    3. questions too close to an evaluation question (cosine >= 0.85) or to an earlier
       generated question (>= 0.92) are dropped, so the test set cannot leak in
    4. evidence is retrieved for the question exactly as the chatbot does it; the example is
       kept only if the source passage is among the passages the model would read
    5. the teacher answers with the chatbot's own prompt, and the answer is kept only if the
       grounding check keeps (almost) all of it, with citations, and does not cut it into scraps
    6. about a quarter of the questions become "insufficient evidence" examples: the question is
       paired with passages retrieved for a different question from another book that are
       related but below the evidence threshold

Books 001-007 form training group A and books 008-015 group B, standing in for two groups
that train adapters separately (for the merging test in src.lora).

Output: data/processed/lora_train.jsonl (quotes the books, so it is kept out of Git) and
artifacts/lora/train_data_summary.json.

Usage:
    python -m src.lora_data --per-group 240
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time

import numpy as np

from src.config import load_config, read_json, read_jsonl, resolve, write_json
from src.generate import (INSUFFICIENT_TOKEN, MIN_ANSWER_WORDS, SYSTEM_PROMPT, OllamaGenerator,
                          build_prompt, ground_answer)
from src.citations import MARKER

OUTPUT = "data/processed/lora_train.jsonl"
CACHE = "data/processed/lora_candidates.jsonl"
SUMMARY = "artifacts/lora/train_data_summary.json"
GROUP_A_BOOKS = {f"BOOK_{n:03d}" for n in range(1, 8)}

QUESTION_SYSTEM = "You write clear exam questions for a university machine learning course."
QUESTION_PROMPT = (
    "Passage:\n{text}\n\n"
    "Write one question that this passage answers. The question must make sense on its own, "
    "without seeing the passage: do not mention 'the passage', 'the text', 'the author' or 'this "
    "chapter'. It must be answerable from the passage alone. Write only the question."
)
BANNED = re.compile(r"\b(passage|text|author|chapter|excerpt|above|this section|figure|exercise)\b"
                    r"|\(\d+(\.\d+)*\)"                                   # equation numbers such as (17.2)
                    r"|\b(table|equation|eq\.|example|section|algorithm|problem)\s*\d", re.I)

EVAL_PAGE_MARGIN = 2
EVAL_LEAK_COSINE = 0.85
DUPLICATE_COSINE = 0.92
NEGATIVE_SHARE = 0.25
NEGATIVE_MAX_COSINE = 0.64   # below the 0.67 evidence threshold: related, but not evidence
MIN_KEPT_SHARE = 0.8         # share of the teacher's words that must survive the grounding check


def group_of(book_id: str) -> str:
    return "A" if book_id.split("_", 1)[1] in GROUP_A_BOOKS else "B"


def is_clean(text: str) -> bool:
    """Prose passages only: long enough, mostly letters, several sentences."""
    words = text.split()
    if len(words) < 120:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text.replace(" ", ""))) < 0.8:
        return False
    return len(re.findall(r"[a-z][.!?](\s|$)", text)) >= 3


def excluded_pages(questions: list[dict]) -> set[tuple[str, int]]:
    """Pages near any validation/test evidence page; training passages may not come from them."""
    pages = set()
    for q in questions:
        if q.get("split") in ("validation", "test"):
            for e in q.get("expected_evidence", []):
                for d in range(-EVAL_PAGE_MARGIN, EVAL_PAGE_MARGIN + 1):
                    pages.add((e["book_id"], int(e["page"]) + d))
    return pages


def clean_question(text: str) -> str | None:
    q = text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""
    q = re.sub(r"^(question\s*:\s*|\d+[.)]\s*)", "", q, flags=re.I).strip()
    if not q.endswith("?") or not 5 <= len(q.split()) <= 35 or BANNED.search(q):
        return None
    return q


def words_without_markers(text: str) -> int:
    return len(MARKER.sub("", text).split())


def is_fragmented(text: str) -> bool:
    """True when the target is formula-heavy or the grounding check cut it into scraps."""
    body = MARKER.sub("", text)
    if sum(c.isalpha() or c == " " for c in body) / max(1, len(body)) < 0.85:
        return True
    for i, piece in enumerate(re.split(r"\[\d+(?:,\s*\d+)*\][.!?]?", text)):
        piece = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", piece).strip()
        if not piece:
            continue
        if len(re.findall(r"[A-Za-z]{2,}", piece)) < 3 or (i and not (piece[0].isupper() or piece[0].isdigit())):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-group", type=int, default=240, help="candidate passages sampled per group")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from src.retrieve import get_retriever

    cfg = load_config()
    g, r = cfg["generation"], cfg["retrieval"]
    retriever = get_retriever(cfg)
    embedder = retriever.embedder
    position = {p["chunk_id"]: i for i, p in enumerate(retriever.passages)}
    teacher = OllamaGenerator(g["ollama_model"], g.get("ollama_url", "http://localhost:11434"),
                              g.get("max_new_tokens", 200), keep_alive=g.get("ollama_keep_alive", "30m"),
                              num_ctx=g.get("ollama_num_ctx"), num_gpu=g.get("ollama_num_gpu"))
    short = OllamaGenerator(g["ollama_model"], g.get("ollama_url", "http://localhost:11434"), 60,
                            keep_alive=g.get("ollama_keep_alive", "30m"),
                            num_ctx=g.get("ollama_num_ctx"), num_gpu=g.get("ollama_num_gpu"))

    questions = read_json(resolve(cfg["paths"]["eval_questions"]), default=[])
    eval_questions = [q for q in questions if q.get("split") in ("validation", "test")]
    eval_vectors = embedder.encode([q["question"] for q in eval_questions], kind="query")
    banned_pages = excluded_pages(questions)

    rng = random.Random(args.seed)
    pools = {"A": [], "B": []}
    for p in retriever.passages:
        if (p["book_id"], p["page"]) not in banned_pages and is_clean(p["text"]):
            pools[group_of(p["book_id"])].append(p)
    sampled = []
    for grp in ("A", "B"):
        sampled += rng.sample(pools[grp], min(args.per_group, len(pools[grp])))
    print(f"Clean passages outside the evaluation pages: A {len(pools['A'])}, B {len(pools['B'])}; "
          f"sampled {len(sampled)}.", flush=True)

    # Candidates are cached so an interrupted run resumes where it stopped.
    cache_path = resolve(CACHE)
    done = {c["source_chunk_id"]: c for c in read_jsonl(cache_path)} if cache_path.exists() else {}
    stats = {"sampled": len(sampled), "bad_question": 0, "eval_leak": 0, "duplicate": 0,
             "source_not_retrieved": 0, "teacher_refused": 0, "not_grounded": 0, "fragmented": 0}
    accepted_vectors: list[np.ndarray] = []
    start = time.time()
    with open(cache_path, "a", encoding="utf-8") as cache:
        for n, p in enumerate(sampled, 1):
            c = done.get(p["chunk_id"])
            if c is None:
                q = clean_question(short(QUESTION_SYSTEM, QUESTION_PROMPT.format(text=p["text"])))
                c = {"source_chunk_id": p["chunk_id"], "group": group_of(p["book_id"]), "question": q}
                if q:
                    hits = retriever.search(q, r["top_k"])
                    evidence = [h for h in hits if h["score"] >= r["min_score"]][: g.get("max_passages", 3)]
                    c["evidence"] = [h["chunk_id"] for h in evidence]
                    if p["chunk_id"] in c["evidence"]:
                        raw = teacher(SYSTEM_PROMPT, build_prompt(q, evidence))
                        c["raw"] = raw
                        c["grounded"] = "" if INSUFFICIENT_TOKEN in raw else ground_answer(raw, evidence, g["min_support"])
                cache.write(json.dumps(c, ensure_ascii=False) + "\n")
                cache.flush()
            done[p["chunk_id"]] = c
            if n % 20 == 0:
                print(f"  {n}/{len(sampled)} passages, {time.time() - start:.0f} s", flush=True)

    # Filters, in a fixed order so the summary counts add up.
    kept = []
    for p in sampled:
        c = done[p["chunk_id"]]
        if not c.get("question") or clean_question(c["question"]) is None:  # re-checked: rules may have tightened
            stats["bad_question"] += 1
            continue
        vec = embedder.encode([c["question"]], kind="query")[0]
        if len(eval_vectors) and float((eval_vectors @ vec).max()) >= EVAL_LEAK_COSINE:
            stats["eval_leak"] += 1
            continue
        if accepted_vectors and float((np.stack(accepted_vectors) @ vec).max()) >= DUPLICATE_COSINE:
            stats["duplicate"] += 1
            continue
        if p["chunk_id"] not in c.get("evidence", []):
            stats["source_not_retrieved"] += 1
            continue
        if INSUFFICIENT_TOKEN in c.get("raw", ""):
            stats["teacher_refused"] += 1
            continue
        target = c.get("grounded", "")
        if (not MARKER.search(target) or words_without_markers(target) < MIN_ANSWER_WORDS
                or words_without_markers(target) < MIN_KEPT_SHARE * words_without_markers(c["raw"])):
            stats["not_grounded"] += 1
            continue
        if is_fragmented(target):
            stats["fragmented"] += 1
            continue
        accepted_vectors.append(vec)
        kept.append({**c, "vector": vec})

    # Turn about a quarter of each group's questions into insufficient-evidence examples.
    examples = []
    for grp in ("A", "B"):
        items = [k for k in kept if k["group"] == grp]
        rng.shuffle(items)
        n_negative = round(NEGATIVE_SHARE * len(items))
        negatives, positives = items[:n_negative], items[n_negative:]
        for k in positives:
            examples.append({"group": grp, "kind": "answer", "question": k["question"],
                             "evidence": k["evidence"], "target": k["grounded"],
                             "source_chunk_id": k["source_chunk_id"]})
        for k in negatives:
            source_book = retriever.by_id[k["source_chunk_id"]]["book_id"]
            best, best_score = None, -1.0
            for other in kept:
                if other is k or k["source_chunk_id"] in other["evidence"]:
                    continue
                books = {retriever.by_id[c]["book_id"] for c in other["evidence"]}
                if source_book in books:
                    continue
                vectors = np.stack([retriever.index.reconstruct(position[c]) for c in other["evidence"]])
                score = float((vectors @ k["vector"]).max())
                if score < NEGATIVE_MAX_COSINE and score > best_score:
                    best, best_score = other, score
            if best is not None:
                examples.append({"group": grp, "kind": "insufficient", "question": k["question"],
                                 "evidence": best["evidence"], "target": INSUFFICIENT_TOKEN,
                                 "source_chunk_id": k["source_chunk_id"], "max_cosine": round(best_score, 3)})
    rng.shuffle(examples)
    for i, e in enumerate(examples, 1):
        e["id"] = f"T{i:04d}"
        e["split"] = "train"

    out = resolve(OUTPUT)
    with open(out, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    summary = {
        "output": OUTPUT,
        "sha256": digest,
        "seed": args.seed,
        "teacher_model": g["ollama_model"],
        "rules": {"eval_page_margin": EVAL_PAGE_MARGIN, "eval_leak_cosine": EVAL_LEAK_COSINE,
                  "duplicate_cosine": DUPLICATE_COSINE, "negative_share": NEGATIVE_SHARE,
                  "negative_max_cosine": NEGATIVE_MAX_COSINE, "min_kept_share": MIN_KEPT_SHARE,
                  "group_A_books": sorted(GROUP_A_BOOKS)},
        "filtering": stats,
        "examples": len(examples),
        "by_group_and_kind": {f"{grp}/{kind}": sum(e["group"] == grp and e["kind"] == kind for e in examples)
                              for grp in ("A", "B") for kind in ("answer", "insufficient")},
        "minutes": round((time.time() - start) / 60, 1),
    }
    write_json(summary, resolve(SUMMARY))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
