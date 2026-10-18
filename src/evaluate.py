"""Stage 8 - Evaluation: run the question set and write the standard report.

Automatic metrics (retrieval, citation traceability, abstention, efficiency) are
computed here, plus two automatic estimates of answer quality against the reference
answers (word-overlap F1 and embedding similarity). They help compare settings but do
not replace people: answer correctness and groundedness come from reviewers, who fill in
evaluation/human_reviews.json using evaluation/answers_for_review.json; the next run
includes their verdicts in the report.

Usage:
    python -m src.evaluate                 # held-out test split
    python -m src.evaluate --split validation
    python -m src.evaluate --backend extractive
    python -m src.evaluate --split validation --backend hf --lora-base --adapter artifacts/lora/lora_A --name val_lora_A
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.citations import MARKER, format_citation, resolve_sources
from src.config import load_config, read_json, resolve, write_json
from src.manifest import identifier, load_manifest, permission_confirmed

QUESTION_TYPES = {"factual", "reasoning", "numerical", "cross_book", "no_answer", "citation_check"}
SPLITS = {"train", "validation", "test"}
TARGETS = {"retrieval_recall_at_k": 0.80, "answer_correctness": 0.80, "citation_traceability": 1.0}


def validate_questions(questions: list[dict]) -> list[str]:
    problems, seen_ids, seen_text = [], set(), {}
    for i, q in enumerate(questions, start=1):
        qid = q.get("id") or f"question #{i}"
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id.")
        seen_ids.add(qid)
        for field in ("id", "question", "type", "split", "answerable"):
            if field not in q or q[field] in ("", None):
                problems.append(f"{qid}: missing {field}.")
        if q.get("type") and q["type"] not in QUESTION_TYPES:
            problems.append(f"{qid}: type '{q['type']}' is not one of {sorted(QUESTION_TYPES)}.")
        if q.get("split") and q["split"] not in SPLITS:
            problems.append(f"{qid}: split '{q['split']}' is not one of {sorted(SPLITS)}.")
        if q.get("answerable") and not q.get("expected_evidence"):
            problems.append(f"{qid}: answerable question has no expected_evidence.")
        key = " ".join(str(q.get("question", "")).lower().split())
        if key in seen_text and seen_text[key] != q.get("split"):
            problems.append(f"{qid}: same question text appears in splits '{seen_text[key]}' and '{q.get('split')}' (leakage).")
        seen_text.setdefault(key, q.get("split"))
    return problems


def _page_keys(items: list[dict]) -> set[tuple[str, int]]:
    return {(e["book_id"], int(e["page"])) for e in items if e.get("book_id") and e.get("page") is not None}


def _near(keys: set[tuple[str, int]], margin: int = 1) -> set[tuple[str, int]]:
    """Pages within `margin` of the given pages: passages never cross a page break, so an
    explanation that starts on the expected page often continues on the next one."""
    return {(b, p + d) for b, p in keys for d in range(-margin, margin + 1)}


def answer_words(text: str) -> list[str]:
    """Lower-case words of an answer or reference, without citation markers, quote prefix or articles."""
    from src.generate import QUOTE_PREFIX

    text = MARKER.sub(" ", text.replace(QUOTE_PREFIX, " ")).lower()
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in {"a", "an", "the"}]


def token_f1(answer: str, reference: str) -> float:
    """Word-overlap F1 between an answer and the reference answer (as used for SQuAD)."""
    a, r = answer_words(answer), answer_words(reference)
    common = sum(min(a.count(w), r.count(w)) for w in set(a) & set(r))
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(r)
    return 2 * precision * recall / (precision + recall)


def peak_ram_mb() -> float | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return round(info.peak_wset / 2**20, 1)
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux reports KB
    except Exception:
        return None


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20, 2)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate(answerer, questions: list[dict], k: int = 5, reviews: dict | None = None) -> tuple[dict, list[dict]]:
    """Return (metrics, per-question records)."""
    from src.generate import QUOTE_PREFIX

    reviews = reviews or {}
    records = []
    embedder = answerer.retriever.embedder
    for q in questions:
        relevant = _page_keys(q.get("expected_evidence", []))
        hits = answerer.retriever.search(q["question"], k)
        retrieved = [(h["book_id"], h["page"]) for h in hits]

        start = time.perf_counter()
        response, cited = answerer.run(q["question"])
        latency = time.perf_counter() - start

        _, invalid = resolve_sources(response["sources"], answerer.retriever.by_id)
        cited_keys = [(s["book_id"], s["page"]) for s in response["sources"]]
        answered_here = not response["insufficient_evidence"]
        reference = q.get("reference_answer", "")
        f1 = similarity = None
        if answered_here and q.get("answerable") and reference:
            f1 = round(token_f1(response["answer"], reference), 4)
            plain = MARKER.sub("", response["answer"].replace(QUOTE_PREFIX, ""))
            vectors = embedder.encode([plain, reference], kind="passage")
            similarity = round(float(vectors[0] @ vectors[1]), 4)
        records.append({
            "id": q["id"],
            "type": q.get("type"),
            "split": q.get("split"),
            "answerable": bool(q.get("answerable")),
            "question": q["question"],
            "reference_answer": q.get("reference_answer", ""),
            "expected_evidence": q.get("expected_evidence", []),
            "response": response,
            "cited_passages": [{"citation": format_citation(p), "chunk_id": p["chunk_id"], "text": p["text"]} for p in cited],
            "retrieved": [{"book_id": b, "page": p} for b, p in retrieved],
            "recall_at_k": _ratio(len(relevant & set(retrieved)), len(relevant)) if relevant else None,
            "hit_at_k": bool(relevant & set(retrieved)) if relevant else None,
            "hit_at_k_pm1": bool(_near(relevant) & set(retrieved)) if relevant else None,
            "citations_traceable": not invalid,
            "cited_pages_expected": _ratio(sum(c in relevant for c in cited_keys), len(cited_keys)) if relevant and cited_keys else None,
            "quote_fallback": answered_here and response["answer"].startswith(QUOTE_PREFIX),
            "answer_f1_auto": f1,
            "answer_similarity_auto": similarity,
            "latency_s": round(latency, 3),
            "review": reviews.get(q["id"]),
        })

    answerable = [r for r in records if r["answerable"]]
    no_answer = [r for r in records if not r["answerable"]]
    with_recall = [r for r in answerable if r["recall_at_k"] is not None]
    answered = [r for r in records if not r["response"]["insufficient_evidence"]]
    page_match = [r["cited_pages_expected"] for r in answered if r["cited_pages_expected"] is not None]
    reviewed = [r for r in records if r["review"] and r["review"].get("correct") is not None]
    grounded = [r for r in records if r["review"] and r["review"].get("grounded") is not None]
    latencies = sorted(r["latency_s"] for r in records)
    scored = [r for r in records if r["answer_f1_auto"] is not None]

    metrics = {
        "questions": len(records),
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "by_type": {t: sum(r["type"] == t for r in records) for t in sorted({r["type"] for r in records})},
        "retrieval_recall_at_k": round(statistics.mean(r["recall_at_k"] for r in with_recall), 4) if with_recall else None,
        "retrieval_hit_rate_at_k": _ratio(sum(r["hit_at_k"] for r in with_recall), len(with_recall)),
        "retrieval_hit_rate_at_k_pm1_page": _ratio(sum(r["hit_at_k_pm1"] for r in with_recall), len(with_recall)),
        "citation_traceability": _ratio(sum(r["citations_traceable"] for r in answered), len(answered)),
        "cited_pages_match_expected_auto": round(statistics.mean(page_match), 4) if page_match else None,
        "abstention_on_no_answer": _ratio(sum(r["response"]["insufficient_evidence"] for r in no_answer), len(no_answer)),
        "false_abstention_on_answerable": _ratio(sum(r["response"]["insufficient_evidence"] for r in answerable), len(answerable)),
        "quote_fallback_rate": _ratio(sum(r["quote_fallback"] for r in answered), len(answered)),
        "answer_f1_auto": round(statistics.mean(r["answer_f1_auto"] for r in scored), 4) if scored else None,
        "answer_similarity_auto": round(statistics.mean(r["answer_similarity_auto"] for r in scored), 4) if scored else None,
        "answers_scored_auto": len(scored),
        "answer_correctness": _ratio(sum(bool(r["review"]["correct"]) for r in reviewed), len(reviewed)),
        "answers_human_reviewed": len(reviewed),
        "groundedness": _ratio(sum(bool(r["review"]["grounded"]) for r in grounded), len(grounded)),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p95_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        "by_type_results": by_type_results(records),
    }
    return metrics, records


def by_type_results(records: list[dict]) -> dict:
    """The main measures for each question type (factual, reasoning, cross_book, ...)."""
    results = {}
    for t in sorted({r["type"] for r in records}):
        group = [r for r in records if r["type"] == t]
        hits = [r["hit_at_k"] for r in group if r["hit_at_k"] is not None]
        scored = [r["answer_f1_auto"] for r in group if r["answer_f1_auto"] is not None]
        results[t] = {
            "questions": len(group),
            "hit_rate_at_k": _ratio(sum(hits), len(hits)),
            "answered": _ratio(sum(not r["response"]["insufficient_evidence"] for r in group), len(group)),
            "answer_f1_auto": round(statistics.mean(scored), 4) if scored else None,
        }
    return results


def document_processing_summary(cfg: dict) -> dict:
    books = load_manifest(resolve(cfg["paths"]["manifest"]))
    ingest = read_json(resolve(cfg["paths"]["ingest_report"]), default={})
    required = ("title", "author", "edition", "publisher", "year", "permission_status")
    # A book without an ISBN (e.g. self-published) counts as complete with a stable identifier (its URL).
    complete = [b for b in books if all(str(b.get(f) or "").strip() for f in required) and identifier(b)]
    return {
        "books_in_manifest": len(books),
        "books_processed": ingest.get("books_processed"),
        "books_needing_ocr": ingest.get("books_needing_ocr", []),
        "books_skipped": ingest.get("books_skipped", []),
        "total_pages": ingest.get("total_pages"),
        "pages_with_text": ingest.get("total_pages_with_text"),
        "failed_pages": {b["book_id"]: b["failed_pages"] for b in ingest.get("books", []) if b["failed_pages"]},
        "ocr_available": ingest.get("ocr_available"),
        "metadata_complete_books": len(complete),
        "permission_confirmed_books": sum(permission_confirmed(b) for b in books),
        "coordinator_approved_books": sum(b.get("coordinator_approval") == "approved" for b in books),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=sorted(SPLITS | {"all"}))
    parser.add_argument("--backend", choices=["hf", "ollama", "extractive"], help="override generation.backend")
    parser.add_argument("--name", help="save as evaluation/report_<name>.json instead of the main report")
    parser.add_argument("--ollama-model", help="override generation.ollama_model, e.g. qwen2.5:3b")
    parser.add_argument("--lora-base", action="store_true",
                        help="hf backend: answer with the LoRA experiment's base model (lora.base_model)")
    parser.add_argument("--adapter", help="hf backend: LoRA adapter folder to load on the base model")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], help="override generation.dtype")
    parser.add_argument("--summary", nargs="+", metavar="REPORT", help="print key results of saved reports and stop")
    args = parser.parse_args(argv)
    if args.summary:
        for path in args.summary:
            m = read_json(resolve(path), default={}).get("metrics", {})
            print(f"  {path}: hit rate {m.get('retrieval_hit_rate_at_k')}, correctness {m.get('answer_correctness')}, "
                  f"groundedness {m.get('groundedness')}, answers reviewed {m.get('answers_human_reviewed')}, "
                  f"auto F1 {m.get('answer_f1_auto')}, auto similarity {m.get('answer_similarity_auto')}")
        return 0

    cfg = load_config()
    if args.backend:
        cfg["generation"]["backend"] = args.backend
    if args.ollama_model:
        cfg["generation"]["ollama_model"] = args.ollama_model
    if (args.lora_base or args.adapter) and cfg["generation"]["backend"] != "hf":
        parser.error("--lora-base and --adapter need --backend hf")
    if args.lora_base:
        cfg["generation"]["model_name"] = cfg["lora"]["base_model"]
        cfg["generation"]["revision"] = cfg["lora"]["base_revision"]
    if args.adapter:
        cfg["generation"]["adapter"] = args.adapter
    if args.dtype:
        cfg["generation"]["dtype"] = args.dtype
    paths = cfg["paths"]
    questions = read_json(resolve(paths["eval_questions"]), default=[])
    problems = validate_questions(questions)
    for p in problems:
        print(f"warning: {p}")
    selected = [q for q in questions if args.split == "all" or q.get("split") == args.split]
    if not selected:
        print(f"No questions with split '{args.split}' in {paths['eval_questions']}.")
        return 1

    from src.generate import Answerer
    from src.retrieve import get_retriever

    answerer = Answerer(get_retriever(cfg), cfg)
    k = cfg["retrieval"]["top_k"]
    metrics, records = evaluate(answerer, selected, k, read_json(resolve(paths["human_reviews"]), default={}))

    index_dir = resolve(paths["index_dir"])
    report = {
        "group_id": cfg["group_id"],
        "subject": cfg["subject"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "conditions": {
            "embedding_model": cfg["embedding"]["model_name"],
            "embedding_revision": cfg["embedding"]["revision"],
            "generation_backend": cfg["generation"]["backend"],
            "generation_model": {"hf": cfg["generation"]["model_name"],
                                 "ollama": cfg["generation"].get("ollama_model")}.get(cfg["generation"]["backend"]),
            "generation_revision": cfg["generation"]["revision"] if cfg["generation"]["backend"] == "hf" else None,
            "generation_adapter": cfg["generation"].get("adapter") if cfg["generation"]["backend"] == "hf" else None,
            "generation_settings": {key: cfg["generation"].get(key) for key in
                                    ("dtype", "max_new_tokens", "max_passages", "min_support")}
                                   if cfg["generation"]["backend"] != "extractive" else None,
            "top_k": k,
            "min_score": cfg["retrieval"]["min_score"],
            "chunking": cfg["chunking"],
        },
        "document_processing": document_processing_summary(cfg),
        "metrics": metrics,
        "targets": {name: {"target": t, "actual": metrics.get(name),
                           "met": None if metrics.get(name) is None else metrics[name] >= t}
                    for name, t in TARGETS.items()},
        "efficiency": {
            "latency_mean_s": metrics["latency_mean_s"],
            "latency_p95_s": metrics["latency_p95_s"],
            "peak_ram_mb": peak_ram_mb(),
            "index_size_mb": dir_size_mb(index_dir) if index_dir.exists() else None,
        },
        "question_file_warnings": problems,
        "failures": [
            {"id": r["id"], "type": r["type"], "issue": issue}
            for r in records
            for issue in (
                ["relevant page not retrieved"] if r["hit_at_k"] is False else []
            ) + (
                ["answered a no-answer question"] if not r["answerable"] and not r["response"]["insufficient_evidence"] else []
            ) + (
                ["abstained on an answerable question"] if r["answerable"] and r["response"]["insufficient_evidence"] else []
            ) + (
                ["citation does not resolve"] if not r["citations_traceable"] else []
            )
        ],
    }
    # Written by the group after reading the failures; kept across re-runs.
    report_path, sheet_path = resolve(paths["report"]), resolve(paths["review_sheet"])
    if args.name:
        report_path = report_path.with_name(f"report_{args.name}.json")
        sheet_path = sheet_path.with_name(f"answers_for_review_{args.name}.json")
    previous = read_json(report_path, default={})
    report["error_analysis"] = previous.get("error_analysis", "")
    report["limitations"] = previous.get("limitations", "")
    # Notes the group added by hand (e.g. version, compared runs) are kept too.
    for key, value in previous.items():
        report.setdefault(key, value)
    write_json(report, report_path)
    write_json(records, sheet_path)

    print(f"Evaluated {len(records)} '{args.split}' questions.")
    for name, value in metrics.items():
        if name not in ("by_type", "by_type_results"):
            print(f"  {name}: {value}")
    print(f"Report: {report_path}")
    print(f"Review sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
