from abc import ABC, abstractmethod
from typing import Any

from app.schemas.retrieval import RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """knowledge_base_id is required, not optional — retrieval must never
        mix chunks from different knowledge bases, and making this an
        unavoidable positional concept (not just another entry in `filters`)
        is what makes that invariant impossible to forget at a call site."""
        ...
