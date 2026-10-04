from app.guardrails.context_budget import limit_to_token_budget
from app.schemas.retrieval import RetrievedChunk


def _chunk(content: str, document_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        document_id=document_id,
        chunk_index=0,
        document_title=document_id,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=0.9,
        distance=0.1,
    )


def test_keeps_all_chunks_within_budget():
    chunks = [_chunk("short", "d1"), _chunk("also short", "d2")]

    assert limit_to_token_budget(chunks, max_tokens=1000) == chunks


def test_truncates_when_over_budget():
    long_content = "word " * 500
    chunks = [_chunk(long_content, f"d{i}") for i in range(5)]

    result = limit_to_token_budget(chunks, max_tokens=200)

    assert 1 <= len(result) < len(chunks)


def test_always_includes_at_least_one_chunk_even_if_it_alone_exceeds_budget():
    huge_content = "word " * 5000
    chunks = [_chunk(huge_content, "d1")]

    result = limit_to_token_budget(chunks, max_tokens=10)

    assert len(result) == 1


def test_empty_input_returns_empty_list():
    assert limit_to_token_budget([], max_tokens=1000) == []
