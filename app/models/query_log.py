from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class QueryLog(Base):
    """Durable, queryable record of every /api/chat call — the rag_query_logs
    table from the original architecture. Stores chunk IDs and similarity
    scores, never raw chunk content or anything from the source database."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    question: Mapped[str] = mapped_column(String)
    knowledge_base_id: Mapped[str] = mapped_column(String, index=True)
    retrieval_query: Mapped[str] = mapped_column(String)
    outcome: Mapped[str] = mapped_column(String)

    retrieval_latency_ms: Mapped[float] = mapped_column(Float)
    rerank_latency_ms: Mapped[float] = mapped_column(Float)
    generation_latency_ms: Mapped[float] = mapped_column(Float)
    total_latency_ms: Mapped[float] = mapped_column(Float)

    retrieved_chunks: Mapped[list] = mapped_column(JSON, default=list)

    embedding_model: Mapped[str] = mapped_column(String)
    llm_provider: Mapped[str] = mapped_column(String)
    llm_model: Mapped[str] = mapped_column(String)
    reranker_used: Mapped[bool] = mapped_column(Boolean, default=False)
    reranker_model: Mapped[str | None] = mapped_column(String, nullable=True)

    estimated_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    blocked_category: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
