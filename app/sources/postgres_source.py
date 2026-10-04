import asyncio
import os
from typing import Any, Mapping

from sqlalchemy import create_engine, text

from app.schemas.document import Document
from app.sources.base import KnowledgeSource
from app.sources.config import EntityConfig, PostgresSourceConfig


class PostgresKnowledgeSource(KnowledgeSource):
    """Turns rows from a configured Postgres source into normalized Documents.

    This class must never learn about a specific domain (products, orders, ...).
    Everything domain-specific comes in through `config`.
    """

    def __init__(self, config: PostgresSourceConfig):
        self._config = config
        connection_string = os.environ[config.connection.env]
        self._engine = create_engine(connection_string, future=True)

    async def load_documents(self) -> list[Document]:
        return await asyncio.to_thread(self._load_documents_sync)

    def _load_documents_sync(self) -> list[Document]:
        documents: list[Document] = []
        with self._engine.connect() as conn:
            for entity in self._config.entities:
                rows = conn.execute(text(entity.query)).mappings().all()
                documents.extend(self._row_to_document(entity, row) for row in rows)
        return documents

    def _row_to_document(self, entity: EntityConfig, row: Mapping[str, Any]) -> Document:
        row_dict = dict(row)

        try:
            content = entity.content_template.format(**row_dict)
        except KeyError as exc:
            raise ValueError(
                f"content_template for entity '{entity.name}' references missing column {exc}"
            ) from exc

        title = str(row_dict[entity.title_column]) if entity.title_column else None

        metadata = dict(entity.static_metadata)
        for column in entity.metadata_columns:
            if column not in row_dict:
                raise ValueError(
                    f"metadata_column '{column}' not found in query result for entity '{entity.name}'"
                )
            metadata[column] = row_dict[column]
        # Every document carries which knowledge base it belongs to — this is
        # what IngestionService promotes into the Document table's dedicated
        # knowledge_base_id column, and what makes isolation enforceable.
        metadata["knowledge_base_id"] = self._config.knowledge_base_id

        return Document(
            id=f"{entity.name}-{row_dict[entity.id_column]}",
            content=content.strip(),
            source=f"{self._config.name}.{entity.name}",
            source_type="database",
            title=title,
            metadata=metadata,
        )
