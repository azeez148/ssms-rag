from app.generation.context_builder import build_context
from app.schemas.retrieval import RetrievedChunk


def _chunk(content: str, title: str | None, document_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        document_id=document_id,
        chunk_index=0,
        document_title=title,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=0.9,
        distance=0.1,
    )


def test_build_context_empty_list_returns_empty_string():
    assert build_context([]) == ""


def test_build_context_numbers_chunks_and_uses_title():
    chunks = [
        _chunk("Argentina Home Jersey details.", "Argentina Jersey", "products-1"),
        _chunk("Portugal Home Jersey details.", "Portugal Jersey", "products-2"),
    ]

    context = build_context(chunks)

    assert "[1] Argentina Jersey" in context
    assert "Argentina Home Jersey details." in context
    assert "[2] Portugal Jersey" in context
    assert "Portugal Home Jersey details." in context


def test_build_context_falls_back_to_document_id_when_title_missing():
    chunks = [_chunk("Some content.", None, "products-99")]
    context = build_context(chunks)

    assert "[1] products-99" in context
