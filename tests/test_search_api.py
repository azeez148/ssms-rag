import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.api.search import get_retriever
from app.core.config import Settings
from app.core.database import engine
from app.embeddings.factory import get_embedding_provider
from app.main import app
from app.models import Chunk, Document
from app.retrieval.vector_retriever import VectorRetriever

client = TestClient(app)

SEED_ITEMS = [
    ("api-test-jersey", "Argentina Home Jersey 2026, official football shirt for fans", "Argentina Jersey", "test-kb"),
    ("api-test-blender", "Stainless steel kitchen blender with 5 speed settings", "Kitchen Blender", "test-kb"),
    ("api-test-other-kb-book", "1984 by George Orwell, a dystopian novel about surveillance", "1984", "other-kb"),
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
def seeded_chunks():
    """Seeds inside a transaction that gets rolled back, and points the /api/search
    endpoint at that same connection via dependency override, so this test never
    sees or pollutes the real Sports Store data already ingested in this database."""
    connection = engine.connect()
    transaction = connection.begin()
    session_factory = sessionmaker(bind=connection)

    embedding_provider = get_embedding_provider(Settings())
    embeddings = asyncio.run(embedding_provider.embed_batch([item[1] for item in SEED_ITEMS]))

    session = session_factory()
    # Uncommitted within this transaction only — rolled back afterward, so the
    # real Sports Store data (committed by earlier plan sets) is untouched and
    # doesn't compete with these tiny synthetic test chunks for top-K ranking.
    session.execute(text("DELETE FROM chunks"))
    session.execute(text("DELETE FROM documents"))

    for (external_id, content, title, knowledge_base_id), embedding in zip(SEED_ITEMS, embeddings):
        document = Document(
            external_id=external_id,
            knowledge_base_id=knowledge_base_id,
            source="test_suite",
            source_type="database",
            title=title,
            content_hash=f"hash-{external_id}",
            metadata_={},
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
                metadata_={},
            )
        )
    session.commit()
    session.close()

    app.dependency_overrides[get_retriever] = lambda: VectorRetriever(
        embedding_provider, session_factory=session_factory
    )

    yield

    app.dependency_overrides.pop(get_retriever, None)
    transaction.rollback()
    connection.close()


def test_search_endpoint_returns_ranked_results(seeded_chunks):
    response = client.post(
        "/api/search", json={"knowledge_base": "test-kb", "query": "Argentina jersey", "top_k": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["document_id"] == "api-test-jersey"
    assert body["results"][0]["knowledge_base_id"] == "test-kb"
    assert "similarity" in body["results"][0]
    assert "distance" in body["results"][0]


def test_search_endpoint_defaults_top_k_to_five(seeded_chunks):
    response = client.post("/api/search", json={"knowledge_base": "test-kb", "query": "jersey"})
    assert response.status_code == 200


def test_search_endpoint_rejects_invalid_top_k():
    response = client.post("/api/search", json={"knowledge_base": "test-kb", "query": "jersey", "top_k": 0})
    assert response.status_code == 422


def test_search_endpoint_requires_knowledge_base():
    response = client.post("/api/search", json={"query": "jersey"})
    assert response.status_code == 422


def test_search_endpoint_never_mixes_knowledge_bases(seeded_chunks):
    """A book seeded under 'other-kb' is a plausible semantic match for this
    query, but searching within 'test-kb' must never surface it."""
    response = client.post(
        "/api/search", json={"knowledge_base": "test-kb", "query": "dystopian novel about surveillance", "top_k": 5}
    )

    body = response.json()
    returned_ids = {r["document_id"] for r in body["results"]}
    assert "api-test-other-kb-book" not in returned_ids
    assert all(r["knowledge_base_id"] == "test-kb" for r in body["results"])
