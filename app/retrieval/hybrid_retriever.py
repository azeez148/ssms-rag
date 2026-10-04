from typing import Any

from app.retrieval.base import Retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.schemas.retrieval import RetrievedChunk


class HybridRetriever(Retriever):
    """Runs vector and keyword retrieval independently, then combines their
    rankings with Reciprocal Rank Fusion. Composes two existing Retrievers —
    it doesn't know or care whether they're backed by pgvector, Postgres
    full-text search, or anything else.
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ):
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # Pull more candidates than top_k from each side before fusing, since
        # the best fused result might rank outside top_k on either individual list.
        candidate_k = top_k * self._candidate_multiplier

        vector_results = await self._vector_retriever.retrieve(
            query, knowledge_base_id, top_k=candidate_k, filters=filters
        )
        keyword_results = await self._keyword_retriever.retrieve(
            query, knowledge_base_id, top_k=candidate_k, filters=filters
        )

        fused = reciprocal_rank_fusion([vector_results, keyword_results], k=self._rrf_k)
        return fused[:top_k]
