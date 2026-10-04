"""Integration test against a real pgvector Postgres instance.

Requires the container from docker-compose.yml to be running:
    docker-compose up -d
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.core.database import engine
from app.models import Chunk, Document


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="RAG postgres container not running")


def test_insert_document_with_chunk_and_embedding(db_session):
    document = Document(
        external_id="test-doc-1",
        knowledge_base_id="test-kb",
        source="unit_test",
        source_type="database",
        title="Test Document",
        content_hash="abc123",
        metadata_={"entity_type": "test"},
    )
    db_session.add(document)
    db_session.flush()

    dimensions = get_settings().embedding_dimensions
    chunk = Chunk(
        document_id=document.id,
        content="hello world",
        chunk_index=0,
        token_count=2,
        embedding=[0.0] * dimensions,
        metadata_={},
    )
    db_session.add(chunk)
    db_session.commit()

    stored = db_session.get(Chunk, chunk.id)
    assert stored is not None
    assert len(stored.embedding) == dimensions
    assert stored.document_id == document.id
