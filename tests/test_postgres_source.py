import asyncio

import pytest
from sqlalchemy import text

from app.core.database import engine
from app.sources.config import ConnectionConfig, EntityConfig, PostgresSourceConfig
from app.sources.postgres_source import PostgresKnowledgeSource


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="postgres not running")


@pytest.fixture
def dummy_table():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_source_items"))
        conn.execute(
            text(
                """
                CREATE TABLE test_source_items (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description VARCHAR,
                    category VARCHAR
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO test_source_items (name, description, category) VALUES "
                "('Widget A', 'A fine widget', 'Widgets'), "
                "('Widget B', 'Another widget', 'Widgets')"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_source_items"))


def _dummy_config() -> PostgresSourceConfig:
    return PostgresSourceConfig(
        type="postgres",
        name="dummy_source",
        knowledge_base_id="dummy",
        connection=ConnectionConfig(env="TEST_SOURCE_DATABASE_URL"),
        entities=[
            EntityConfig(
                name="items",
                query="SELECT id, name, description, category FROM test_source_items ORDER BY id",
                id_column="id",
                title_column="name",
                content_template="Name: {name}\nDescription: {description}",
                metadata_columns=["category"],
            )
        ],
    )


def test_load_documents_from_dummy_table(dummy_table, monkeypatch):
    monkeypatch.setenv("TEST_SOURCE_DATABASE_URL", engine.url.render_as_string(hide_password=False))

    source = PostgresKnowledgeSource(_dummy_config())
    documents = asyncio.run(source.load_documents())

    assert len(documents) == 2

    first = documents[0]
    assert first.id == "items-1"
    assert first.source == "dummy_source.items"
    assert first.source_type == "database"
    assert first.title == "Widget A"
    assert first.content == "Name: Widget A\nDescription: A fine widget"
    assert first.metadata == {"category": "Widgets", "knowledge_base_id": "dummy"}


def test_missing_content_template_column_raises_clear_error(dummy_table, monkeypatch):
    monkeypatch.setenv("TEST_SOURCE_DATABASE_URL", engine.url.render_as_string(hide_password=False))

    config = _dummy_config()
    config.entities[0].content_template = "Name: {name}\nSKU: {sku}"

    source = PostgresKnowledgeSource(config)

    with pytest.raises(ValueError, match="missing column"):
        asyncio.run(source.load_documents())


def test_config_rejects_empty_entities():
    with pytest.raises(ValueError, match="at least one entity"):
        PostgresSourceConfig(
            type="postgres",
            name="dummy_source",
            knowledge_base_id="dummy",
            connection=ConnectionConfig(env="TEST_SOURCE_DATABASE_URL"),
            entities=[],
        )


def test_knowledge_source_cannot_be_instantiated_directly():
    from app.sources.base import KnowledgeSource

    with pytest.raises(TypeError):
        KnowledgeSource()
