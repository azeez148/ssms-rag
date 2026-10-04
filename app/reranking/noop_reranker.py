from app.reranking.base import Reranker
from app.schemas.retrieval import RetrievedChunk


class NoOpReranker(Reranker):
    """Passes through the retriever's own ranking, truncated to top_k. Used
    when reranking is disabled — lets callers request reranking unconditionally
    while the actual behavior stays config-driven."""

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]
