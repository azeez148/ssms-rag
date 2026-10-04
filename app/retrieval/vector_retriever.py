from typing import Any

from sqlalchemy.orm import sessionmaker

from app.core.database import SessionLocal
from app.embeddings.base import EmbeddingProvider
from app.models import Chunk, Document
from app.retrieval.base import Retriever
from app.retrieval.filters import apply_metadata_filters
from app.schemas.retrieval import RetrievedChunk


class VectorRetriever(Retriever):
    """Embeds the query and runs a pgvector cosine-distance nearest-neighbor
    search against `chunks`. No keyword matching, no LLM call.

    `knowledge_base_id` is always applied as a hard SQL WHERE clause on the
    real, indexed Document.knowledge_base_id column — before ranking, before
    top_k — so a query against one knowledge base structurally cannot surface
    a chunk from another one.

    Optional `filters` (e.g. {"category": "Jersey"}) are a separate, generic
    mechanism applied as SQL WHERE clauses on the chunk's JSON metadata *before*
    the ORDER BY/LIMIT — pre-filtering, not filtering the top_k results
    afterward. The retriever has no idea what "category" means.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, session_factory: sessionmaker = SessionLocal):
        self._embedding_provider = embedding_provider
        self._session_factory = session_factory

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        query_vector = await self._embedding_provider.embed_text(query)

        session = self._session_factory()
        try:
            distance = Chunk.embedding.cosine_distance(query_vector)
            db_query = (
                session.query(Chunk, Document, distance.label("distance"))
                .join(Document, Chunk.document_id == Document.id)
                .filter(Document.knowledge_base_id == knowledge_base_id)
            )
            db_query = apply_metadata_filters(db_query, filters)
            rows = db_query.order_by(distance).limit(top_k).all()
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
                distance=float(distance_value),
                similarity=1.0 - float(distance_value),
            )
            for chunk, document, distance_value in rows
        ]
