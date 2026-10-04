from app.core.config import get_settings
from app.models import Base, Chunk, Document, IngestionRun, QueryLog


def test_expected_tables_registered():
    assert set(Base.metadata.tables.keys()) == {"documents", "chunks", "ingestion_runs", "query_logs"}


def test_document_columns():
    columns = {c.name for c in Document.__table__.columns}
    assert columns == {
        "id",
        "external_id",
        "knowledge_base_id",
        "source",
        "source_type",
        "title",
        "content_hash",
        "metadata",
        "created_at",
        "updated_at",
    }


def test_chunk_has_vector_embedding_column():
    embedding_column = Chunk.__table__.columns["embedding"]
    assert embedding_column.type.dim == get_settings().embedding_dimensions


def test_ingestion_run_defaults():
    columns = {c.name for c in IngestionRun.__table__.columns}
    assert {"source", "status", "documents_processed", "chunks_created"} <= columns


def test_query_log_columns():
    columns = {c.name for c in QueryLog.__table__.columns}
    assert {
        "question",
        "knowledge_base_id",
        "retrieval_query",
        "outcome",
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "retrieved_chunks",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "estimated_cost_usd",
        "blocked_category",
        "error",
    } <= columns
