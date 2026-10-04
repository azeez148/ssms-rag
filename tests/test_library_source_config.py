"""Plan Set 20 — proves the RAG core is database-agnostic by pointing the
existing, unmodified PostgresKnowledgeSource at a completely unrelated
domain (a book library, not a sports store). No line in app/sources/,
app/ingestion/, app/embeddings/, app/retrieval/, app/services/, or
app/generation/ was touched to make this work — only this config file and
one env var were added. Requires the real library_db Postgres instance;
skips if unavailable.
"""

import asyncio
import os

import pytest
from sqlalchemy import create_engine, text

from app.sources.config import load_postgres_source_config
from app.sources.postgres_source import PostgresKnowledgeSource

CONFIG_PATH = "config/library_source.yaml"
ENV_VAR = "LIBRARY_DATABASE_URL"
DEFAULT_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/library_db"


def _database_available() -> bool:
    url = os.environ.get(ENV_VAR, DEFAULT_URL)
    try:
        with create_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="library_db not reachable")


@pytest.fixture(autouse=True)
def _set_connection_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, os.environ.get(ENV_VAR, DEFAULT_URL))


def test_library_documents_are_normalized_using_the_unmodified_adapter():
    config = load_postgres_source_config(CONFIG_PATH)
    source = PostgresKnowledgeSource(config)

    documents = asyncio.run(source.load_documents())

    assert len(documents) == 20
    for document in documents:
        assert document.source == "library.books"
        assert document.source_type == "database"
        assert document.metadata["entity_type"] == "book"
        assert "genre" in document.metadata
        assert "author" in document.metadata
        assert document.title  # every book has a title


def test_a_specific_real_book_is_correctly_normalized():
    config = load_postgres_source_config(CONFIG_PATH)
    source = PostgresKnowledgeSource(config)

    documents = asyncio.run(source.load_documents())
    hobbit = next(d for d in documents if d.title == "The Hobbit")

    assert hobbit.metadata["author"] == "J.R.R. Tolkien"
    assert hobbit.metadata["genre"] == "Fantasy"
    assert "Bilbo Baggins" in hobbit.content


def test_document_schema_has_no_knowledge_of_books_or_sports_store():
    from app.schemas.document import Document

    assert set(Document.model_fields) == {"id", "content", "source", "source_type", "title", "metadata"}
