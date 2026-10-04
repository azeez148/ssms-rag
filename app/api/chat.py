import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends

from app.actions.pending import pending_actions
from app.actions.registry import get_action_handler
from app.agentic.service import AgenticRAGService
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.generation.citations import extract_cited_indices
from app.generation.factory import get_llm_provider, resolve_llm_identity
from app.generation.query_rewriter import QueryRewriter
from app.models import QueryLog
from app.observability.events import ChatObservabilityEvent
from app.reranking.factory import get_reranker
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.chat import ChatAction, ChatRequest, ChatResponse, ChatSource
from app.services.rag_service import ChatResult, RAGService

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


def get_rag_service() -> RAGService:
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    retriever = VectorRetriever(embedding_provider)
    llm_provider = get_llm_provider(settings)
    reranker = get_reranker(settings)
    query_rewriter = QueryRewriter(llm_provider, max_history_turns=settings.max_history_turns)
    return RAGService(
        retriever,
        llm_provider,
        reranker,
        query_rewriter,
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        similarity_threshold=settings.guardrails_similarity_threshold,
        max_context_tokens=settings.max_context_tokens,
        embedding_model_name=settings.embedding_model,
        llm_provider_name=resolve_llm_identity(settings)[0],
        llm_model_name=resolve_llm_identity(settings)[1],
        reranker_model_name=settings.reranker_model,
    )


def get_agentic_rag_service() -> AgenticRAGService:
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    retriever = VectorRetriever(embedding_provider)
    llm_provider = get_llm_provider(settings)
    reranker = get_reranker(settings)
    query_rewriter = QueryRewriter(llm_provider, max_history_turns=settings.max_history_turns)
    return AgenticRAGService(
        retriever,
        llm_provider,
        reranker,
        query_rewriter,
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
        similarity_threshold=settings.guardrails_similarity_threshold,
        max_context_tokens=settings.max_context_tokens,
        max_retries=settings.agentic_max_retries,
        embedding_model_name=settings.embedding_model,
        llm_provider_name=resolve_llm_identity(settings)[0],
        llm_model_name=resolve_llm_identity(settings)[1],
        reranker_model_name=settings.reranker_model,
    )


def _persist_query_log(event: ChatObservabilityEvent) -> None:
    """Best-effort persistence to the queryable rag_query_logs table. The
    structured stdout log (already emitted by RAGService) is the durable
    source of truth — a DB write failure here must never break the chat
    response the user is waiting on.
    """
    session = SessionLocal()
    try:
        session.add(
            QueryLog(
                question=event.question,
                knowledge_base_id=event.knowledge_base_id,
                retrieval_query=event.retrieval_query,
                outcome=event.outcome,
                retrieval_latency_ms=event.retrieval_latency_ms,
                rerank_latency_ms=event.rerank_latency_ms,
                generation_latency_ms=event.generation_latency_ms,
                total_latency_ms=event.total_latency_ms,
                retrieved_chunks=[asdict(c) for c in event.retrieved_chunks],
                embedding_model=event.embedding_model,
                llm_provider=event.llm_provider,
                llm_model=event.llm_model,
                reranker_used=event.reranker_used,
                reranker_model=event.reranker_model,
                estimated_prompt_tokens=event.estimated_prompt_tokens,
                estimated_completion_tokens=event.estimated_completion_tokens,
                estimated_cost_usd=event.estimated_cost_usd,
                blocked_category=event.blocked_category,
                error=event.error,
            )
        )
        session.commit()
    except Exception:
        logger.exception("Failed to persist query log (chat response is unaffected)")
    finally:
        session.close()


def _build_chat_response(result: ChatResult) -> ChatResponse:
    if result.observability is not None:
        _persist_query_log(result.observability)

    # Citation indices match the [n] numbering build_context() assigned when
    # constructing the LLM's prompt, so a cited claim maps back to one exact chunk.
    cited_indices = extract_cited_indices(result.answer, num_sources=len(result.sources))

    sources = [
        ChatSource(
            index=index,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            source=chunk.source,
            knowledge_base_id=chunk.knowledge_base_id,
            similarity=chunk.similarity,
            metadata=chunk.metadata,
            cited=index in cited_indices,
        )
        for index, chunk in enumerate(result.sources, start=1)
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        cited_source_count=len(cited_indices),
        retrieval_query=result.retrieval_query,
        blocked=result.blocked,
        blocked_category=result.blocked_category,
    )


ACTION_UNAVAILABLE = "I couldn't reach the live store data just now, so I can't do that. Please try again in a moment."


async def _answer_as_action(request: ChatRequest) -> ChatResponse | None:
    """If the knowledge base declares actions and this question is one (a
    sale, a live stock check...), answer it with the action handler instead
    of RAG. Writes are only proposed here; they run on confirmation."""
    handler = get_action_handler(request.knowledge_base, Path(get_settings().source_config_dir))
    if handler is None:
        return None

    try:
        reply = await handler.handle(request.question, request.history)
    except Exception:
        logger.exception("action handler failed for knowledge base %s", request.knowledge_base)
        reply_text, action = ACTION_UNAVAILABLE, None
    else:
        if reply is None:
            return None
        reply_text, action = reply.answer, None
        if reply.proposal is not None:
            pending = pending_actions.put(request.knowledge_base, reply.proposal)
            action = ChatAction(
                id=pending.id,
                title=reply.proposal.title,
                lines=reply.proposal.lines,
                confirm_label=reply.proposal.confirm_label,
                expires_at=pending.expires_at,
            )

    return ChatResponse(
        answer=reply_text,
        sources=[],
        cited_source_count=0,
        retrieval_query=request.question,
        action=action,
        from_action=True,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    if (action_response := await _answer_as_action(request)) is not None:
        return action_response
    result = await service.answer(
        request.question,
        request.knowledge_base,
        top_k=request.top_k,
        filters=request.filters,
        rerank=request.rerank,
        history=request.history,
    )
    return _build_chat_response(result)


@router.post("/chat/agentic", response_model=ChatResponse)
async def chat_agentic(
    request: ChatRequest, service: AgenticRAGService = Depends(get_agentic_rag_service)
) -> ChatResponse:
    """Plan Set 23: the same request/response shape as /api/chat, answered by
    the LangGraph agentic pipeline (Analyze -> Retrieve -> Evaluate ->
    bounded rewrite-and-retry -> Generate -> Validate -> Respond) instead of
    RAGService's straight-line pipeline.
    """
    if (action_response := await _answer_as_action(request)) is not None:
        return action_response
    result = await service.answer(
        request.question,
        request.knowledge_base,
        top_k=request.top_k,
        filters=request.filters,
        rerank=request.rerank,
        history=request.history,
    )
    return _build_chat_response(result)
