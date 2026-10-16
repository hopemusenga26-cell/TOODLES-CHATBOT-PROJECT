"""Retrieval, answer format, citations and insufficient-evidence behaviour."""
from src.citations import renumber_markers, resolve_sources
from src.evaluate import evaluate, validate_questions
from src.generate import INSUFFICIENT_TOKEN, QUOTE_PREFIX, Answerer, ground_answer

RESULT_FIELDS = {"text", "score", "book_id", "title", "page", "chunk_id"}


def test_retrieve_returns_common_fields(retriever):
    results = retriever.search("What is a cache hit?", top_k=5)
    assert results
    assert all(RESULT_FIELDS <= r.keys() for r in results)
    assert results[0]["page"] == 1


def test_retrieve_respects_top_k_and_orders_by_score(retriever):
    results = retriever.search("pipeline data hazard forwarding", top_k=1)
    assert len(results) == 1
    assert results[0]["page"] == 2
    both = retriever.search("pipeline data hazard forwarding", top_k=5)
    assert [r["score"] for r in both] == sorted((r["score"] for r in both), reverse=True)


def test_answer_has_common_format(answerer):
    response = answerer.answer("What is a cache miss?")
    assert set(response) == {"answer", "sources", "insufficient_evidence"}
    assert isinstance(response["answer"], str) and response["answer"]
    assert response["insufficient_evidence"] is False
    assert all(set(s) == {"book_id", "page", "chunk_id"} for s in response["sources"])


def test_every_citation_resolves_to_a_stored_passage(answerer, retriever):
    response = answerer.answer("What does forwarding do in a pipeline?")
    assert response["sources"]
    valid, invalid = resolve_sources(response["sources"], retriever.by_id)
    assert invalid == [] and valid == response["sources"]
    assert "[1]" in response["answer"]


def test_fabricated_citations_are_rejected(retriever):
    real = retriever.passages[0]
    fake = [{"book_id": real["book_id"], "page": 999, "chunk_id": "G000_BOOK_001_P999_C01"},
            {"book_id": "OTHER", "page": real["page"], "chunk_id": real["chunk_id"]}]
    valid, invalid = resolve_sources(fake, retriever.by_id)
    assert valid == [] and len(invalid) == 2


def test_marker_renumbering_drops_out_of_range_citations():
    text, order = renumber_markers("Second claim [3]. First claim [1, 7]. Again [3].", evidence_count=3)
    assert order == [2, 0]
    assert text == "Second claim [1]. First claim [2]. Again [1]."


def test_off_topic_question_reports_insufficient_evidence(answerer):
    response = answerer.answer("Who won the football world cup final in 1966?")
    assert response["insufficient_evidence"] is True
    assert response["sources"] == []


def _model_answerer(retriever, cfg, output):
    cfg["generation"]["backend"] = "hf"
    return Answerer(retriever, cfg, generator=lambda system, user: output)


OUTSIDE_KNOWLEDGE = ("Virtual memory uses page tables to translate logical addresses into physical "
                     "addresses [1]. The operating system swaps pages to disk [1].")


def test_model_refusal_counts_as_insufficient(retriever, cfg):
    response = _model_answerer(retriever, cfg, INSUFFICIENT_TOKEN).answer("What is a cache?")
    assert response["insufficient_evidence"] is True and response["sources"] == []


def test_unverifiable_model_output_falls_back_to_book_quotes(retriever, cfg):
    for output in ("Made-up claim [9].", "[1] [2]", OUTSIDE_KNOWLEDGE):
        response = _model_answerer(retriever, cfg, output).answer("What is a cache hit?")
        assert response["insufficient_evidence"] is False
        assert response["answer"].startswith(QUOTE_PREFIX)
        assert "Virtual memory" not in response["answer"] and "Made-up" not in response["answer"]
        assert "cache hit" in response["answer"]
        valid, invalid = resolve_sources(response["sources"], retriever.by_id)
        assert invalid == [] and response["sources"][0]["page"] == 1


def test_unverifiable_output_is_refused_when_quote_fallback_is_off(retriever, cfg):
    cfg["generation"]["fallback_to_quotes"] = False
    for output in ("Made-up claim [9].", "[1] [2]", OUTSIDE_KNOWLEDGE):
        response = _model_answerer(retriever, cfg, output).answer("What is a cache?")
        assert response["insufficient_evidence"] is True and response["sources"] == []


def test_model_answer_with_valid_citation_is_accepted(retriever, cfg):
    response = _model_answerer(retriever, cfg, "A cache stores copies of data [1].").answer("What is a cache?")
    assert response["insufficient_evidence"] is False
    assert response["sources"][0]["page"] == 1


def test_supported_sentence_without_citation_gets_cited(retriever, cfg):
    output = "A cache hit is when the processor finds the requested data in the cache."
    response = _model_answerer(retriever, cfg, output).answer("What is a cache hit?")
    assert response["insufficient_evidence"] is False
    assert response["answer"].endswith("cache [1].")
    assert response["sources"][0]["page"] == 1


def test_list_answers_keep_their_lines():
    evidence = [{"text": "A classic five stage pipeline has instruction fetch, instruction decode, execute, "
                         "memory access and write back stages."}]
    raw = "The five pipeline stages are:\n1. Instruction fetch [1]\n2. Instruction decode [1]\n- Write back"
    assert ground_answer(raw, evidence, 0.5) == (
        "The five pipeline stages are:\n1. Instruction fetch [1]\n2. Instruction decode [1]\n- Write back [1]"
    )


def test_wrong_citation_number_is_corrected_and_unsupported_sentence_dropped(retriever, cfg):
    output = ("Forwarding passes results directly between pipeline stages [5]. "
              "It was invented at a famous laboratory in 1962 [1].")
    response = _model_answerer(retriever, cfg, output).answer("What does forwarding do in a pipeline?")
    assert response["answer"] == "Forwarding passes results directly between pipeline stages [1]."
    assert [s["page"] for s in response["sources"]] == [2]


def test_evaluation_metrics(answerer):
    questions = [
        {"id": "Q1", "question": "What is a cache hit?", "type": "factual", "split": "test", "answerable": True,
         "expected_evidence": [{"book_id": "G000_BOOK_001", "page": 1}]},
        {"id": "Q2", "question": "Who painted the Mona Lisa?", "type": "no_answer", "split": "test",
         "answerable": False, "expected_evidence": []},
    ]
    assert validate_questions(questions) == []
    metrics, records = evaluate(answerer, questions, k=5)
    assert metrics["retrieval_recall_at_k"] == 1.0
    assert metrics["citation_traceability"] == 1.0
    assert metrics["abstention_on_no_answer"] == 1.0
    assert len(records) == 2
