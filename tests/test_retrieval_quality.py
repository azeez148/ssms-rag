from app.guardrails.retrieval_quality import retrieval_is_too_weak
from app.schemas.retrieval import RetrievedChunk


def _chunk(similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        content="x",
        document_id="d",
        chunk_index=0,
        document_title="d",
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=similarity,
        distance=1.0 - similarity,
    )


def test_empty_retrieval_is_too_weak():
    assert retrieval_is_too_weak([], similarity_threshold=0.35) is True


def test_below_threshold_is_too_weak():
    assert retrieval_is_too_weak([_chunk(0.2)], similarity_threshold=0.35) is True


def test_above_threshold_is_not_too_weak():
    assert retrieval_is_too_weak([_chunk(0.6)], similarity_threshold=0.35) is False


def test_only_the_top_result_matters():
    assert retrieval_is_too_weak([_chunk(0.6), _chunk(0.01)], similarity_threshold=0.35) is False
