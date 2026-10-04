import math
from typing import Any

from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import Retriever
from app.schemas.chunk import ChunkData
from app.schemas.document import Document
from app.schemas.retrieval import RetrievedChunk


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


class InMemoryRetriever(Retriever):
    """A pgvector-free Retriever for offline experiments (e.g. comparing chunk
    configurations) without writing to the real ingested index."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        indexed_chunks: list[tuple[Document, ChunkData, list[float]]],
    ):
        self._embedding_provider = embedding_provider
        self._indexed_chunks = indexed_chunks

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        query_vector = await self._embedding_provider.embed_text(query)

        # Ephemeral/in-memory, so there's no real Document.knowledge_base_id
        # column to filter on — the source adapter already stamped it into
        # each chunk's metadata, so filtering there gives the same isolation
        # guarantee as the DB-backed retrievers, just via a different mechanism.
        candidates = [
            (document, chunk_data, embedding)
            for document, chunk_data, embedding in self._indexed_chunks
            if chunk_data.metadata.get("knowledge_base_id") == knowledge_base_id
        ]
        if filters:
            candidates = [
                (document, chunk_data, embedding)
                for document, chunk_data, embedding in candidates
                if all(str(chunk_data.metadata.get(key)) == str(value) for key, value in filters.items())
            ]

        scored = [
            (_cosine_similarity(query_vector, embedding), document, chunk_data)
            for document, chunk_data, embedding in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(
                content=chunk_data.content,
                document_id=document.id,
                chunk_index=chunk_data.chunk_index,
                document_title=document.title,
                source=document.source,
                knowledge_base_id=chunk_data.metadata.get("knowledge_base_id", ""),
                metadata=chunk_data.metadata,
                similarity=similarity,
                distance=1.0 - similarity,
            )
            for similarity, document, chunk_data in scored[:top_k]
        ]
