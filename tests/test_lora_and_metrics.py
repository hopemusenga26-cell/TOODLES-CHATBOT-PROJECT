"""Tests for the automatic answer metrics and the optional LoRA tools (no model downloads)."""
from __future__ import annotations

import json

import pytest

from src.evaluate import _near, evaluate, token_f1
from src.lora import adapter_card, check_adapter_base, compatibility
from src.lora_data import clean_question, excluded_pages, group_of, is_clean, is_fragmented


# ---------------------------------------------------------------- automatic answer metrics

def test_token_f1_ignores_citations_case_and_articles():
    assert token_f1("A cache miss happens [1].", "the cache miss happens") == 1.0
    assert token_f1("The books say: cache miss [2]", "cache miss") == 1.0
    assert token_f1("pipelining", "cache") == 0.0
    assert token_f1("", "cache") == 0.0
    assert 0 < token_f1("a cache stores copies of data", "a cache is small and fast") < 1


def test_near_pages_include_neighbours_in_the_same_book_only():
    assert _near({("B1", 10)}) == {("B1", 9), ("B1", 10), ("B1", 11)}


def test_evaluate_reports_automatic_answer_scores(answerer):
    questions = [
        {"id": "Q1", "question": "What is a cache miss?", "type": "factual", "split": "test",
         "answerable": True, "reference_answer": "When the data is not in the cache it is a cache miss.",
         "expected_evidence": [{"book_id": "G000_BOOK_001", "page": 1}]},
        {"id": "Q2", "question": "Who won the 1998 football world cup?", "type": "no_answer",
         "split": "test", "answerable": False, "expected_evidence": []},
    ]
    metrics, records = evaluate(answerer, questions, k=3)
    assert metrics["answers_scored_auto"] == 1
    assert 0 < metrics["answer_f1_auto"] <= 1
    assert -1 <= metrics["answer_similarity_auto"] <= 1
    assert metrics["retrieval_hit_rate_at_k_pm1_page"] == 1.0
    assert metrics["quote_fallback_rate"] == 0.0  # the extractive backend quotes without the fallback prefix
    assert records[1]["answer_f1_auto"] is None
    by_type = metrics["by_type_results"]
    assert by_type["factual"]["questions"] == 1 and by_type["factual"]["hit_rate_at_k"] == 1.0
    assert by_type["no_answer"]["answered"] == 0.0 and by_type["no_answer"]["hit_rate_at_k"] is None


# ---------------------------------------------------------------- training data rules

def test_training_groups_split_the_books():
    assert group_of("G000_BOOK_001") == "A" and group_of("G000_BOOK_007") == "A"
    assert group_of("G000_BOOK_008") == "B" and group_of("G000_BOOK_015") == "B"


def test_training_passages_avoid_pages_near_evaluation_evidence():
    questions = [{"split": "test", "expected_evidence": [{"book_id": "B", "page": 10}]},
                 {"split": "train", "expected_evidence": [{"book_id": "C", "page": 5}]}]
    pages = excluded_pages(questions)
    assert {("B", p) for p in range(8, 13)} <= pages
    assert ("B", 13) not in pages and ("C", 5) not in pages


def test_generated_questions_must_stand_alone():
    assert clean_question("Question: What does the lasso penalty do?") == "What does the lasso penalty do?"
    assert clean_question("What does the passage say about ridge regression?") is None
    assert clean_question("Ridge regression shrinks coefficients.") is None
    assert clean_question("Why?") is None


def test_only_prose_passages_are_used():
    prose = " ".join(["The model is fitted to the data by least squares."] * 15)
    assert is_clean(prose)
    assert not is_clean("short text.")
    assert not is_clean(" ".join(["x = 1 + 2 * y ^ 3 ;"] * 40))


def test_fragmented_or_formula_targets_are_rejected():
    good = "The goal is to choose parameters that minimise the loss [1]. This is called model fitting [1]."
    listed = "The steps are:\n1. Choose a class of model [2].\n2. Fit the model to the data [1]."
    scraps = "Dwc is defined as Dwc ij = sup Dij, i, j = 1, [1]. , m, i = j P [1]. The columns need not add up [1]."
    assert not is_fragmented(good) and not is_fragmented(listed)
    assert is_fragmented(scraps)


# ---------------------------------------------------------------- adapter compatibility

def card(**changes):
    base = {"path": "a", "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "base_revision": "rev1",
            "architecture": "Qwen2ForCausalLM", "tokenizer_sha256": "tok", "peft_type": "LORA",
            "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"], "r": 8, "lora_alpha": 16,
            "use_rslora": False, "use_dora": False, "prompt_version": "v2"}
    return {**base, **changes}


def test_identical_adapters_are_compatible():
    assert compatibility(card(), card(path="b")) == ([], [])


@pytest.mark.parametrize("field, value", [
    ("base_model", "Qwen/Qwen2.5-1.5B-Instruct"), ("base_revision", "rev2"),
    ("architecture", "LlamaForCausalLM"), ("tokenizer_sha256", "other"),
    ("target_modules", ["q_proj", "k_proj"]), ("peft_type", "IA3"),
])
def test_adapters_on_different_bases_or_layers_are_blocked(field, value):
    problems, _ = compatibility(card(), card(**{field: value}))
    assert problems


def test_unknown_tokenizer_blocks_merging():
    problems, _ = compatibility(card(), card(tokenizer_sha256=None))
    assert any("not recorded" in p for p in problems)


def test_different_rank_or_prompt_only_warns():
    problems, warnings = compatibility(card(), card(r=16, lora_alpha=32, prompt_version="v1"))
    assert not problems and len(warnings) == 3


def write_adapter(folder, base="Qwen/Qwen2.5-0.5B-Instruct", revision="rev1"):
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base, "revision": revision, "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "target_modules": ["v_proj", "q_proj"], "r": 8, "lora_alpha": 16}))
    (folder / "training_record.json").write_text(json.dumps({
        "base_model": {"revision": revision, "architecture": "Qwen2ForCausalLM", "tokenizer_sha256": "tok"},
        "prompt_version": "v2"}))
    return folder


def test_adapter_card_reads_config_and_training_record(tmp_path):
    c = adapter_card(write_adapter(tmp_path / "a"))
    assert c["target_modules"] == ["q_proj", "v_proj"] and c["tokenizer_sha256"] == "tok"
    assert compatibility(c, adapter_card(write_adapter(tmp_path / "b"))) == ([], [])


def test_adapter_is_refused_on_another_base_model(tmp_path):
    folder = write_adapter(tmp_path / "a")
    check_adapter_base(folder, "Qwen/Qwen2.5-0.5B-Instruct", "rev1")
    with pytest.raises(ValueError):
        check_adapter_base(folder, "Qwen/Qwen2.5-1.5B-Instruct", "rev1")
    with pytest.raises(ValueError):
        check_adapter_base(folder, "Qwen/Qwen2.5-0.5B-Instruct", "rev2")
