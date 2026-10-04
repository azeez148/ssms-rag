import asyncio
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from app.core.database import SessionLocal
from app.models import Chunk, Document
from app.retrieval.base import Retriever
from app.retrieval.filters import apply_metadata_filters
from app.schemas.retrieval import RetrievedChunk


class KeywordRetriever(Retriever):
    """Postgres full-text (lexical) search against `chunks.content`, using
    `to_tsvector`/`plainto_tsquery`. Matches on stemmed word tokens, not
    meaning — catches exact names/SKUs that an embedding might blur past.

    No index is built for this (e.g. a persisted tsvector column + GIN index);
    `to_tsvector` runs per-row at query time. Fine at this corpus's scale
    (~1000 rows) and keeps the implementation simple; a production system
    would add a generated column + GIN index for performance.
    """

    def __init__(self, session_factory: sessionmaker = SessionLocal):
        self._session_factory = session_factory

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._retrieve_sync, query, knowledge_base_id, top_k, filters)

    def _retrieve_sync(
        self, query: str, knowledge_base_id: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        session = self._session_factory()
        try:
            tsquery = func.plainto_tsquery("english", query)
            tsvector = func.to_tsvector("english", Chunk.content)
            rank = func.ts_rank(tsvector, tsquery).label("rank")

            db_query = (
                session.query(Chunk, Document, rank)
                .join(Document, Chunk.document_id == Document.id)
                .filter(Document.knowledge_base_id == knowledge_base_id)
                .filter(tsvector.op("@@")(tsquery))
            )
            db_query = apply_metadata_filters(db_query, filters)
            rows = db_query.order_by(rank.desc()).limit(top_k).all()
        finally:
            session.close()

        return [
            RetrievedChunk(
                content=chunk.content,
                document_id=document.external_id,
                chunk_index=chunk.chunk_index,
                document_title=document.title,
                source=document.source,
                knowledge_base_id=document.knowledge_base_id,
                metadata=chunk.metadata_,
                # ts_rank is an unbounded relevance score, not a distance metric —
                # different scale than cosine similarity, not directly comparable.
                similarity=float(rank_value),
                distance=-float(rank_value),
            )
            for chunk, document, rank_value in rows
        ]
