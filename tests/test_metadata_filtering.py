import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import engine
from app.embeddings.factory import get_embedding_provider
from app.models import Chunk, Document
from app.retrieval.vector_retriever import VectorRetriever

SEED_ITEMS = [
    ("filter-jersey", "Argentina Home Jersey 2026, official football shirt for fans", "Argentina Jersey", "Jersey"),
    ("filter-shoe", "Running shoes with breathable mesh for jogging", "Running Shoes", "Shoes"),
    ("filter-blender", "Stainless steel kitchen blender with 5 speed settings", "Kitchen Blender", "Kitchen"),
]


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="RAG postgres not running")


@pytest.fixture(scope="module")
def embedding_provider():
    return get_embedding_provider(Settings())


@pytest.fixture
def isolated_session_factory():
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection)

    yield factory

    transaction.rollback()
    connection.close()


@pytest.fixture
def seeded_chunks(isolated_session_factory, embedding_provider):
    embeddings = asyncio.run(embedding_provider.embed_batch([item[1] for item in SEED_ITEMS]))

    session = isolated_session_factory()
    session.execute(text("DELETE FROM chunks"))
    session.execute(text("DELETE FROM documents"))

    for (external_id, content, title, category), embedding in zip(SEED_ITEMS, embeddings):
        document = Document(
            external_id=external_id,
            knowledge_base_id="test-kb",
            source="test_suite",
            source_type="database",
            title=title,
            content_hash=f"hash-{external_id}",
            metadata_={"category": category},
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                content=content,
                chunk_index=0,
                token_count=10,
                embedding=embedding,
                metadata_={"category": category},
            )
        )
    session.commit()
    session.close()


def test_no_filter_returns_the_best_semantic_match(seeded_chunks, isolated_session_factory, embedding_provider):
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("Do you have Argentina jerseys?", "test-kb", top_k=3))

    assert results[0].document_id == "filter-jersey"
    assert len(results) == 3


def test_filter_excludes_a_more_semantically_similar_item_in_another_category(
    seeded_chunks, isolated_session_factory, embedding_provider
):
    """The jersey chunk is the best semantic match for this query, but filtering
    to category=Shoes must still exclude it entirely — pre-filtering guarantees
    the filter wins over similarity ranking, not the other way around."""
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("Do you have Argentina jerseys?", "test-kb", top_k=3, filters={"category": "Shoes"}))

    assert len(results) == 1
    assert results[0].document_id == "filter-shoe"


def test_filter_on_nonexistent_value_returns_empty_list(seeded_chunks, isolated_session_factory, embedding_provider):
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(
        retriever.retrieve("Do you have Argentina jerseys?", "test-kb", top_k=3, filters={"category": "Electronics"})
    )

    assert results == []


def test_filter_is_generic_equality_not_semantic_reasoning(
    seeded_chunks, isolated_session_factory, embedding_provider
):
    """The retriever has no idea what 'category' or 'Kitchen' mean — filtering
    to category=Kitchen must return the blender even for a totally unrelated
    query, because this is a plain equality filter, not another semantic pass."""
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("Argentina jersey", "test-kb", top_k=3, filters={"category": "Kitchen"}))

    assert len(results) == 1
    assert results[0].document_id == "filter-blender"
