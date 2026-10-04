from pydantic import BaseModel, field_validator


class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_size(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value
