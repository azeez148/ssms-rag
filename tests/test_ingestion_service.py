import asyncio

import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.core.database import SessionLocal, engine
from app.embeddings.factory import get_embedding_provider
from app.ingestion.chunking import LangChainRecursiveChunker, RecursiveTokenChunker
from app.ingestion.config import ChunkingConfig
from app.models import Chunk, Document
from app.services.ingestion_service import IngestionService
from app.sources.config import ConnectionConfig, EntityConfig, PostgresSourceConfig
from app.sources.postgres_source import PostgresKnowledgeSource


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
def dummy_source_table(monkeypatch):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_ingestion_items"))
        conn.execute(
            text(
                """
                CREATE TABLE test_ingestion_items (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description VARCHAR
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO test_ingestion_items (name, description) VALUES "
                "('Item One', 'First test item'), "
                "('Item Two', 'Second test item')"
            )
        )

    monkeypatch.setenv("TEST_INGESTION_DATABASE_URL", engine.url.render_as_string(hide_password=False))

    yield

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_ingestion_items"))


@pytest.fixture
def cleanup_documents():
    yield
    session = SessionLocal()
    try:
        session.query(Document).filter(Document.external_id.like("items-%")).delete(synchronize_session=False)
        session.commit()
    finally:
        session.close()


def _make_source() -> PostgresKnowledgeSource:
    config = PostgresSourceConfig(
        type="postgres",
        name="test_ingestion",
        knowledge_base_id="test-ingestion",
        connection=ConnectionConfig(env="TEST_INGESTION_DATABASE_URL"),
        entities=[
            EntityConfig(
                name="items",
                query="SELECT id, name, description FROM test_ingestion_items ORDER BY id",
                id_column="id",
                title_column="name",
                content_template="{name}: {description}",
            )
        ],
    )
    return PostgresKnowledgeSource(config)


def _make_service(embedding_provider) -> IngestionService:
    return IngestionService(
        source=_make_source(),
        source_name="test_ingestion",
        chunker=RecursiveTokenChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50)),
        embedding_provider=embedding_provider,
    )


def test_ingestion_service_works_unchanged_with_langchain_chunker(dummy_source_table, cleanup_documents, embedding_provider):
    """IngestionService depends only on the ChunkingStrategy ABC, so swapping
    in Plan Set 22's LangChain-backed chunker requires no change to the
    service itself — the whole point of the strategy pattern from Plan Set 5."""
    service = IngestionService(
        source=_make_source(),
        source_name="test_ingestion",
        chunker=LangChainRecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50)),
        embedding_provider=embedding_provider,
    )
    session = SessionLocal()
    try:
        run, result = asyncio.run(service.run(session))
    finally:
        session.close()

    assert run.status == "completed"
    assert result.documents_created == 2
    assert result.chunks_created == 2


def test_first_run_creates_documents_and_chunks(dummy_source_table, cleanup_documents, embedding_provider):
    service = _make_service(embedding_provider)
    session = SessionLocal()
    try:
        run, result = asyncio.run(service.run(session))
    finally:
        session.close()

    assert run.status == "completed"
    assert result.documents_created == 2
    assert result.documents_updated == 0
    assert result.documents_unchanged == 0
    assert result.chunks_created == 2


def test_second_run_with_no_changes_skips_embedding(dummy_source_table, cleanup_documents, embedding_provider):
    service = _make_service(embedding_provider)

    session = SessionLocal()
    try:
        asyncio.run(service.run(session))
    finally:
        session.close()

    session = SessionLocal()
    try:
        run, result = asyncio.run(service.run(session))
    finally:
        session.close()

    assert run.status == "completed"
    assert result.documents_unchanged == 2
    assert result.documents_created == 0
    assert result.chunks_created == 0


def test_changed_content_replaces_chunks_and_regenerates_embeddings(
    dummy_source_table, cleanup_documents, embedding_provider
):
    service = _make_service(embedding_provider)

    session = SessionLocal()
    try:
        asyncio.run(service.run(session))
    finally:
        session.close()

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE test_ingestion_items SET description = 'Completely rewritten description' WHERE id = 1")
        )

    session = SessionLocal()
    try:
        run, result = asyncio.run(service.run(session))
    finally:
        session.close()

    assert result.documents_updated == 1
    assert result.documents_unchanged == 1
    assert result.chunks_created == 1

    session = SessionLocal()
    try:
        updated_doc = session.query(Document).filter_by(external_id="items-1").one()
        chunks = session.query(Chunk).filter_by(document_id=updated_doc.id).all()
        assert len(chunks) == 1
        assert "Completely rewritten" in chunks[0].content
    finally:
        session.close()


def test_new_row_is_created_without_touching_existing_documents(
    dummy_source_table, cleanup_documents, embedding_provider
):
    service = _make_service(embedding_provider)

    session = SessionLocal()
    try:
        asyncio.run(service.run(session))
    finally:
        session.close()

    with engine.begin() as conn:
        conn.execute(text("INSERT INTO test_ingestion_items (name, description) VALUES ('Item Three', 'A new item')"))

    session = SessionLocal()
    try:
        run, result = asyncio.run(service.run(session))
    finally:
        session.close()

    assert result.documents_created == 1
    assert result.documents_unchanged == 2
    assert result.chunks_created == 1
