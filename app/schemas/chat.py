from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    knowledge_base: str
    question: str
    top_k: int = Field(default=5, gt=0, le=50)
    filters: dict[str, str] | None = None
    rerank: bool = False
    history: list[ConversationTurn] = Field(default_factory=list)


class ChatSource(BaseModel):
    index: int
    document_id: str
    document_title: str | None
    source: str
    knowledge_base_id: str
    similarity: float
    metadata: dict
    cited: bool


class ChatAction(BaseModel):
    """A proposed write (e.g. recording a sale) waiting for the user to
    confirm via POST /api/actions/{id}/confirm or cancel it."""

    id: str
    title: str
    lines: list[str]
    confirm_label: str
    expires_at: float


class ActionOutcome(BaseModel):
    status: Literal["done", "cancelled"]
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    cited_source_count: int
    retrieval_query: str
    blocked: bool = False
    blocked_category: str | None = None
    # Set when the question was an action request that needs confirmation.
    action: ChatAction | None = None
    # Answered by the knowledge base's action handler (stock, sales...) rather
    # than RAG. Clients leave these turns out of `history`: they're tool
    # output, and feeding them to the query rewriter drags their products
    # into unrelated follow-up questions.
    from_action: bool = False
