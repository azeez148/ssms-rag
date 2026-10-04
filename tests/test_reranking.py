import asyncio

import pytest

from app.core.config import Settings
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.reranking.factory import get_reranker
from app.reranking.noop_reranker import NoOpReranker
from app.schemas.retrieval import RetrievedChunk


def _chunk(document_id: str, content: str, similarity: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        document_id=document_id,
        chunk_index=0,
        document_title=document_id,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=similarity,
        distance=1.0 - similarity,
    )


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def test_cross_encoder_corrects_a_bi_encoder_mistake(reranker):
    # Deliberately handed to the reranker in the "wrong" order, simulating a
    # first-stage retriever mistake — the cross-encoder should fix it.
    chunks = [
        _chunk("blender", "Stainless steel kitchen blender with 5 speed settings."),
        _chunk("jersey", "Argentina Home Jersey 2026, official football shirt, sizes S to XL."),
    ]

    results = asyncio.run(reranker.rerank("Do you have Argentina jerseys?", chunks, top_k=2))

    assert results[0].document_id == "jersey"


def test_cross_encoder_limits_to_top_k(reranker):
    chunks = [_chunk(f"c{i}", f"Item number {i} description text.") for i in range(10)]
    results = asyncio.run(reranker.rerank("query", chunks, top_k=3))

    assert len(results) == 3


def test_cross_encoder_replaces_similarity_with_its_own_score(reranker):
    chunks = [_chunk("a", "Argentina jersey.", similarity=0.01)]
    results = asyncio.run(reranker.rerank("Argentina jersey", chunks, top_k=1))

    assert results[0].similarity != 0.01
    assert results[0].distance == -results[0].similarity


def test_cross_encoder_empty_input_returns_empty(reranker):
    assert asyncio.run(reranker.rerank("query", [], top_k=5)) == []


def test_noop_reranker_passes_through_truncated_to_top_k():
    chunks = [_chunk(f"c{i}", "text") for i in range(5)]
    results = asyncio.run(NoOpReranker().rerank("query", chunks, top_k=2))

    assert results == chunks[:2]


def test_factory_returns_cross_encoder_reranker():
    reranker = get_reranker(Settings(reranker_provider="cross_encoder"))
    assert isinstance(reranker, CrossEncoderReranker)


def test_factory_returns_noop_reranker():
    reranker = get_reranker(Settings(reranker_provider="none"))
    assert isinstance(reranker, NoOpReranker)


def test_factory_raises_clear_error_for_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        get_reranker(Settings(reranker_provider="bogus"))


def test_factory_reuses_the_loaded_cross_encoder_instead_of_reloading_per_request():
    from app.core.config import Settings
    from app.reranking.factory import get_reranker

    settings = Settings(reranker_provider="cross_encoder")

    assert get_reranker(settings) is get_reranker(settings)
