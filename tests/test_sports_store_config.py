"""Verifies the Sports Store source config produces valid, non-sensitive Documents
using the existing, unmodified PostgresKnowledgeSource. Requires the real ssms_db
Postgres instance; skips if unavailable.
"""

import asyncio
import os

import pytest
from sqlalchemy import create_engine, text

from app.sources.config import load_postgres_source_config
from app.sources.postgres_source import PostgresKnowledgeSource

CONFIG_PATH = "config/sports_store_source.yaml"
ENV_VAR = "SPORTS_STORE_DATABASE_URL"
DEFAULT_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/ssms_db"


def _database_available() -> bool:
    url = os.environ.get(ENV_VAR, DEFAULT_URL)
    try:
        with create_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="ssms_db not reachable")


@pytest.fixture(autouse=True)
def _set_connection_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, os.environ.get(ENV_VAR, DEFAULT_URL))


def test_sports_store_documents_are_normalized_and_safe():
    config = load_postgres_source_config(CONFIG_PATH)
    source = PostgresKnowledgeSource(config)

    documents = asyncio.run(source.load_documents())

    assert len(documents) > 0

    sensitive_terms = {"customer", "password", "email", "phone", "unit_price", "cost"}
    for document in documents:
        assert document.source == "sports_store.products"
        assert document.source_type == "database"
        assert document.metadata["entity_type"] == "product"
        assert "category" in document.metadata
        assert not sensitive_terms & set(document.content.lower().split())


def test_sports_store_documents_contain_no_sports_store_python_types():
    # The core Document schema stays generic: it has no knowledge of "Product".
    from app.schemas.document import Document

    assert "product" not in Document.model_fields
    assert set(Document.model_fields) == {"id", "content", "source", "source_type", "title", "metadata"}
