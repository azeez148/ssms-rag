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
    ("test-jersey", "Argentina Home Jersey 2026, official football shirt for fans", "Argentina Jersey"),
    ("test-shoe", "Running shoes with breathable mesh for jogging", "Running Shoes"),
    ("test-blender", "Stainless steel kitchen blender with 5 speed settings", "Kitchen Blender"),
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
    """A transaction that's rolled back at the end, so seeded test chunks never
    become visible to other connections (e.g. the 1000 real Sports Store chunks
    already ingested in this same database stay untouched and unseen here)."""
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
    # Uncommitted within this transaction only — rolled back afterward, so the
    # real Sports Store data (committed by earlier plan sets) is untouched and
    # doesn't compete with these tiny synthetic test chunks for top-K ranking.
    session.execute(text("DELETE FROM chunks"))
    session.execute(text("DELETE FROM documents"))

    for (external_id, content, title), embedding in zip(SEED_ITEMS, embeddings):
        document = Document(
            external_id=external_id,
            knowledge_base_id="test-kb",
            source="test_suite",
            source_type="database",
            title=title,
            content_hash=f"hash-{external_id}",
            metadata_={"category": "test"},
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
                metadata_={"category": "test"},
            )
        )
    session.commit()
    session.close()


def test_retrieve_ranks_most_similar_chunk_first(seeded_chunks, isolated_session_factory, embedding_provider):
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("Do you have Argentina jerseys?", "test-kb", top_k=3))

    assert len(results) == 3
    assert results[0].document_id == "test-jersey"
    assert results[0].similarity > results[-1].similarity


def test_top_k_limits_result_count(seeded_chunks, isolated_session_factory, embedding_provider):
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("football", "test-kb", top_k=1))
    assert len(results) == 1


def test_response_includes_required_fields(seeded_chunks, isolated_session_factory, embedding_provider):
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("jersey", "test-kb", top_k=1))
    result = results[0]

    assert result.content
    assert result.document_id
    assert result.source == "test_suite"
    assert isinstance(result.metadata, dict)
    assert 0.0 <= result.similarity <= 1.0
    assert result.distance == pytest.approx(1.0 - result.similarity)


def test_semantic_search_matches_without_keyword_overlap(seeded_chunks, isolated_session_factory, embedding_provider):
    """The query shares zero words with the jersey document's content, yet it
    should still rank as the most similar result — proving this isn't SQL LIKE."""
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)
    results = asyncio.run(retriever.retrieve("national team soccer uniform for supporters", "test-kb", top_k=3))

    assert results[0].document_id == "test-jersey"


def test_retrieval_never_mixes_knowledge_bases(seeded_chunks, isolated_session_factory, embedding_provider):
    """The seeded jersey chunk is a near-perfect semantic match for this
    query, but it belongs to knowledge_base_id='test-kb' — asking a
    different knowledge base the exact same question must return nothing
    from it. This is the acceptance criterion for Plan Set 21."""
    retriever = VectorRetriever(embedding_provider, session_factory=isolated_session_factory)

    results = asyncio.run(retriever.retrieve("Do you have Argentina jerseys?", "some-other-kb", top_k=5))

    assert results == []
