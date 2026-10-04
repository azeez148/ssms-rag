from app.evaluation.dataset import EvalQuestion
from app.services.rag_service import ChatResult

# Which outcome each question category should produce. Everything not listed
# here is expected to be answered; only genuinely out-of-domain ("unsupported")
# questions are expected to be correctly declined.
EXPECTED_OUTCOME_BY_CATEGORY: dict[str, str] = {
    "unsupported": "insufficient_information",
}
DEFAULT_EXPECTED_OUTCOME = "answered"

_REFUSAL_OUTCOMES = {"insufficient_information", "blocked"}


def expected_outcome_for(category: str) -> str:
    return EXPECTED_OUTCOME_BY_CATEGORY.get(category, DEFAULT_EXPECTED_OUTCOME)


def actual_outcome_of(result: ChatResult) -> str:
    if result.observability is not None:
        return result.observability.outcome
    return "blocked" if result.blocked else "answered"


def is_answer_appropriate(question: EvalQuestion, result: ChatResult) -> bool:
    """A behavioral proxy for "answer relevance": did the system make the
    right decision to answer or decline, given what this category of
    question calls for? This does NOT judge the content of the generated
    text — with a stub LLM there's no real content to judge — only whether
    answering vs. refusing was the correct call.
    """
    expected = expected_outcome_for(question.category)
    actual = actual_outcome_of(result)

    if expected == "insufficient_information":
        return actual in _REFUSAL_OUTCOMES
    return actual == "answered"


def is_grounded(result: ChatResult, cited_source_count: int) -> bool | None:
    """A structural proxy for "groundedness": did the answer cite at least
    one of its retrieved sources? Returns None (not applicable) when the
    system didn't answer at all — groundedness is only a property of answers
    that exist.
    """
    if actual_outcome_of(result) != "answered":
        return None
    return cited_source_count > 0


def is_citation_correct(question: EvalQuestion, cited_document_ids: set[str]) -> bool | None:
    """Of the sources actually cited (not just retrieved), does at least one
    match a real correct document for this question? Returns None when the
    question has no ground truth to check against (e.g. "unsupported").
    """
    if not question.expected_document_ids:
        return None
    return bool(cited_document_ids & set(question.expected_document_ids))
