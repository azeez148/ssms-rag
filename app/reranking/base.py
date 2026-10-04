from abc import ABC, abstractmethod

from app.schemas.retrieval import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...
