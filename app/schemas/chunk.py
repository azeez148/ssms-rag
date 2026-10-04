from pydantic import BaseModel, Field


class ChunkData(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: dict = Field(default_factory=dict)
