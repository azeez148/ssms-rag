from typing import Any, TypedDict

from app.schemas.chat import ConversationTurn
from app.schemas.retrieval import RetrievedChunk


class AgenticState(TypedDict):
    """LangGraph state for Plan Set 23's agentic RAG flow. Every node reads
    some of this and returns a partial dict of the keys it changed — LangGraph
    merges those into the state that flows to the next node. Nothing here is
    conversation-turn history in the durable sense; it lives only for the
    lifetime of one `AgenticRAGService.answer()` call.
    """

    question: str
    knowledge_base_id: str
    top_k: int
    fetch_k: int
    filters: dict[str, Any] | None
    rerank: bool
    history: list[ConversationTurn]

    retrieval_query: str
    candidates: list[RetrievedChunk]
    retries: int
    max_retries: int
    enough_context: bool

    sources: list[RetrievedChunk]
    answer: str
    blocked: bool
    blocked_category: str | None
    validated: bool

    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
