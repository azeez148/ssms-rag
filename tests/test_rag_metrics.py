from app.evaluation.dataset import EvalQuestion
from app.evaluation.rag_metrics import (
    expected_outcome_for,
    is_answer_appropriate,
    is_citation_correct,
    is_grounded,
)
from app.observability.events import ChatObservabilityEvent
from app.services.rag_service import ChatResult


def _question(category: str, expected_document_ids: list[str] | None = None) -> EvalQuestion:
    return EvalQuestion(category=category, question="q", expected_document_ids=expected_document_ids or [])


def _result(outcome: str, blocked: bool = False) -> ChatResult:
    event = ChatObservabilityEvent(
        question="q",
        knowledge_base_id="test-kb",
        retrieval_query="q",
        outcome=outcome,
        retrieval_latency_ms=1.0,
        rerank_latency_ms=0.0,
        generation_latency_ms=1.0,
        total_latency_ms=2.0,
        retrieved_chunks=[],
        embedding_model="m",
        llm_provider="stub",
        llm_model="stub-echo-v1",
        reranker_used=False,
    )
    return ChatResult(answer="answer", sources=[], retrieval_query="q", blocked=blocked, observability=event)


def test_expected_outcome_for_unsupported_is_insufficient_information():
    assert expected_outcome_for("unsupported") == "insufficient_information"


def test_expected_outcome_for_every_other_category_is_answered():
    for category in ["simple", "semantic", "exact-match", "multi-document", "ambiguous", "follow-up"]:
        assert expected_outcome_for(category) == "answered"


def test_unsupported_question_correctly_declined_is_appropriate():
    question = _question("unsupported")
    result = _result("insufficient_information")
    assert is_answer_appropriate(question, result) is True


def test_unsupported_question_wrongly_answered_is_not_appropriate():
    question = _question("unsupported")
    result = _result("answered")
    assert is_answer_appropriate(question, result) is False


def test_blocked_counts_as_a_valid_refusal_for_unsupported():
    question = _question("unsupported")
    result = _result("blocked", blocked=True)
    assert is_answer_appropriate(question, result) is True


def test_simple_question_answered_is_appropriate():
    question = _question("simple", ["products-1"])
    result = _result("answered")
    assert is_answer_appropriate(question, result) is True


def test_simple_question_wrongly_declined_is_not_appropriate():
    question = _question("simple", ["products-1"])
    result = _result("insufficient_information")
    assert is_answer_appropriate(question, result) is False


def test_grounded_requires_at_least_one_citation():
    result = _result("answered")
    assert is_grounded(result, cited_source_count=1) is True
    assert is_grounded(result, cited_source_count=0) is False


def test_grounded_is_not_applicable_when_not_answered():
    result = _result("insufficient_information")
    assert is_grounded(result, cited_source_count=0) is None


def test_citation_correct_when_cited_id_matches_expected():
    question = _question("exact-match", ["products-1"])
    assert is_citation_correct(question, {"products-1"}) is True


def test_citation_incorrect_when_cited_id_does_not_match_expected():
    question = _question("exact-match", ["products-1"])
    assert is_citation_correct(question, {"products-99"}) is False


def test_citation_not_applicable_when_no_ground_truth():
    question = _question("unsupported", [])
    assert is_citation_correct(question, set()) is None
