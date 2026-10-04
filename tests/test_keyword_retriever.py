import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.core.database import engine
from app.models import Chunk, Document
from app.retrieval.keyword_retriever import KeywordRetriever

SEED_ITEMS = [
    ("kw-exact-code", "Model XJ429Q running shoe with reinforced sole and breathable mesh.", "XJ429Q Shoe", "Shoes"),
    ("kw-similar", "Running shoes with breathable mesh for jogging, various colors available.", "Running Shoes", "Shoes"),
    ("kw-unrelated", "Stainless steel kitchen blender with 5 speed settings.", "Kitchen Blender", "Kitchen"),
]


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="RAG postgres not running")


@pytest.fixture
def isolated_session_factory():
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection)

    yield factory

    transaction.rollback()
    connection.close()


@pytest.fixture
def seeded_chunks(isolated_session_factory):
    session = isolated_session_factory()
    session.execute(text("DELETE FROM chunks"))
    session.execute(text("DELETE FROM documents"))

    for external_id, content, title, category in SEED_ITEMS:
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
                embedding=[0.0] * 384,
                metadata_={"category": category},
            )
        )
    session.commit()
    session.close()


def test_exact_token_match_is_found(seeded_chunks, isolated_session_factory):
    retriever = KeywordRetriever(isolated_session_factory)
    results = asyncio.run(retriever.retrieve("XJ429Q", "test-kb", top_k=5))

    assert len(results) == 1
    assert results[0].document_id == "kw-exact-code"


def test_no_lexical_overlap_returns_empty_list_unlike_vector_search(seeded_chunks, isolated_session_factory):
    """Vector search always returns its nearest neighbors, however weak. Full-text
    search has no such fallback: a query sharing no stemmed word with any chunk
    matches nothing at all — a real structural difference between the two."""
    retriever = KeywordRetriever(isolated_session_factory)
    results = asyncio.run(retriever.retrieve("zzyzx nonexistent gibberish term", "test-kb", top_k=5))

    assert results == []


def test_filters_apply_to_keyword_search_too(seeded_chunks, isolated_session_factory):
    retriever = KeywordRetriever(isolated_session_factory)
    results = asyncio.run(retriever.retrieve("running", "test-kb", top_k=5, filters={"category": "Kitchen"}))

    assert results == []


def test_ranks_more_relevant_match_higher(seeded_chunks, isolated_session_factory):
    retriever = KeywordRetriever(isolated_session_factory)
    results = asyncio.run(retriever.retrieve("running shoes breathable mesh", "test-kb", top_k=5))

    document_ids = [r.document_id for r in results]
    assert "kw-exact-code" in document_ids
    assert "kw-similar" in document_ids
    assert "kw-unrelated" not in document_ids
