from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RetrievedChunkLog:
    document_id: str
    chunk_index: int
    similarity: float


@dataclass
class ChatObservabilityEvent:
    """Everything needed to debug a single /api/chat call after the fact,
    without needing to reproduce it: what was asked, what was retrieved (IDs
    and scores, never raw chunk content), how long each stage took, which
    models were involved, and what it cost.

    Never includes: chunk content, credentials, or anything from the source
    database — only the question itself and derived metadata. The question is
    logged because it's the whole point of debugging a bad answer; a system
    over a more sensitive domain would need to redact it further.
    """

    question: str
    knowledge_base_id: str
    retrieval_query: str
    outcome: str  # "answered" | "blocked" | "insufficient_information" | "error"

    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    retrieved_chunks: list[RetrievedChunkLog]

    embedding_model: str
    llm_provider: str
    llm_model: str
    reranker_used: bool
    reranker_model: str | None = None

    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    blocked_category: str | None = None
    error: str | None = None

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
