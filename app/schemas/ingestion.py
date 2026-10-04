from pydantic import BaseModel


class IngestRequest(BaseModel):
    config_path: str
    chunk_size: int = 500
    chunk_overlap: int = 50


class IngestResponse(BaseModel):
    run_id: int
    status: str
    documents_processed: int
    documents_created: int
    documents_updated: int
    documents_unchanged: int
    chunks_created: int
    error: str | None = None
