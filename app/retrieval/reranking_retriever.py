from typing import Any

from app.reranking.base import Reranker
from app.retrieval.base import Retriever
from app.schemas.retrieval import RetrievedChunk


class RerankingRetriever(Retriever):
    """Wraps a base retriever: fetches a wider candidate pool (top_k *
    candidate_multiplier), then reranks down to top_k. Composes any Retriever
    with any Reranker — neither knows the other exists.
    """

    def __init__(self, base_retriever: Retriever, reranker: Reranker, candidate_multiplier: int = 4):
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._candidate_multiplier = candidate_multiplier

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        candidates = await self._base_retriever.retrieve(
            query, knowledge_base_id, top_k=top_k * self._candidate_multiplier, filters=filters
        )
        return await self._reranker.rerank(query, candidates, top_k)
