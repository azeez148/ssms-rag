from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    knowledge_base: str
    query: str
    top_k: int = Field(default=5, gt=0, le=50)
    filters: dict[str, str] | None = None
    mode: Literal["vector", "keyword", "hybrid"] = "vector"
    rerank: bool = False


class RetrievedChunk(BaseModel):
    content: str
    document_id: str
    chunk_index: int
    document_title: str | None
    source: str
    knowledge_base_id: str
    metadata: dict
    similarity: float
    distance: float


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]
