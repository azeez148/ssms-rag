import time
from dataclasses import dataclass
from typing import Any

from app.generation.base import LLMProvider
from app.generation.context_builder import build_context
from app.generation.prompts import SYSTEM_PROMPT
from app.generation.query_rewriter import QueryRewriter
from app.guardrails.content_guard import filter_suspicious_chunks
from app.guardrails.context_budget import limit_to_token_budget
from app.guardrails.input_guard import InputGuard
from app.guardrails.retrieval_quality import retrieval_is_too_weak
from app.ingestion.chunking import count_tokens
from app.observability.cost import estimate_cost
from app.observability.events import ChatObservabilityEvent, RetrievedChunkLog
from app.observability.logger import log_chat_event
from app.reranking.base import Reranker
from app.retrieval.base import Retriever
from app.schemas.chat import ConversationTurn
from app.schemas.retrieval import RetrievedChunk

INSUFFICIENT_INFO_MESSAGE = "There is insufficient information in the knowledge base to answer that question."


@dataclass
class ChatResult:
    answer: str
    sources: list[RetrievedChunk]
    retrieval_query: str
    blocked: bool = False
    blocked_category: str | None = None
    observability: ChatObservabilityEvent | None = None


class RAGService:
    """Question -> input guard -> (optional query rewrite) -> retrieval ->
    weak-retrieval check -> content guard -> (optional rerank) -> context
    budget -> prompt -> LLM -> answer + sources.

    Every guardrail runs before the LLM is called wherever possible — a
    blocked question or a too-weak retrieval never reaches generation at all.
    Every return path emits a structured observability event so a bad answer
    can be traced back to the exact stage that produced it.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        reranker: Reranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        top_k: int = 5,
        rerank_candidate_multiplier: int = 4,
        input_guard: InputGuard | None = None,
        similarity_threshold: float = 0.35,
        max_context_tokens: int = 1500,
        embedding_model_name: str = "unknown",
        llm_provider_name: str = "unknown",
        llm_model_name: str = "unknown",
        reranker_model_name: str | None = None,
    ):
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._reranker = reranker
        self._query_rewriter = query_rewriter
        self._top_k = top_k
        self._rerank_candidate_multiplier = rerank_candidate_multiplier
        self._input_guard = input_guard or InputGuard()
        self._similarity_threshold = similarity_threshold
        self._max_context_tokens = max_context_tokens
        self._embedding_model_name = embedding_model_name
        self._llm_provider_name = llm_provider_name
        self._llm_model_name = llm_model_name
        self._reranker_model_name = reranker_model_name

    def _emit(
        self,
        *,
        question: str,
        knowledge_base_id: str,
        retrieval_query: str,
        outcome: str,
        total_start: float,
        retrieval_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
        generation_latency_ms: float = 0.0,
        sources: list[RetrievedChunk] = (),
        blocked_category: str | None = None,
        error: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> ChatObservabilityEvent:
        event = ChatObservabilityEvent(
            question=question,
            knowledge_base_id=knowledge_base_id,
            retrieval_query=retrieval_query,
            outcome=outcome,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            rerank_latency_ms=round(rerank_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            retrieved_chunks=[
                RetrievedChunkLog(document_id=c.document_id, chunk_index=c.chunk_index, similarity=c.similarity)
                for c in sources
            ],
            embedding_model=self._embedding_model_name,
            llm_provider=self._llm_provider_name,
            llm_model=self._llm_model_name,
            reranker_used=rerank_latency_ms > 0,
            reranker_model=self._reranker_model_name if rerank_latency_ms > 0 else None,
            estimated_prompt_tokens=prompt_tokens,
            estimated_completion_tokens=completion_tokens,
            estimated_cost_usd=estimate_cost(self._llm_provider_name, self._llm_model_name, prompt_tokens, completion_tokens),
            blocked_category=blocked_category,
            error=error,
        )
        log_chat_event(event)
        return event

    async def answer(
        self,
        question: str,
        knowledge_base_id: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = False,
        history: list[ConversationTurn] | None = None,
    ) -> ChatResult:
        total_start = time.perf_counter()

        guard_result = self._input_guard.check(question)
        if guard_result.blocked:
            event = self._emit(
                question=question,
                knowledge_base_id=knowledge_base_id,
                retrieval_query=question,
                outcome="blocked",
                total_start=total_start,
                blocked_category=guard_result.category,
            )
            return ChatResult(
                answer=guard_result.reason or INSUFFICIENT_INFO_MESSAGE,
                sources=[],
                retrieval_query=question,
                blocked=True,
                blocked_category=guard_result.category,
                observability=event,
            )

        effective_top_k = top_k or self._top_k
        history = history or []

        if self._query_rewriter is not None and history:
            retrieval_query = await self._query_rewriter.rewrite(question, history)
        else:
            retrieval_query = question

        should_rerank = rerank and self._reranker is not None
        fetch_k = effective_top_k * self._rerank_candidate_multiplier if should_rerank else effective_top_k

        retrieval_start = time.perf_counter()
        candidates = await self._retriever.retrieve(retrieval_query, knowledge_base_id, top_k=fetch_k, filters=filters)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        if retrieval_is_too_weak(candidates, self._similarity_threshold):
            event = self._emit(
                question=question,
                knowledge_base_id=knowledge_base_id,
                retrieval_query=retrieval_query,
                outcome="insufficient_information",
                total_start=total_start,
                retrieval_latency_ms=retrieval_latency_ms,
                sources=candidates,
            )
            return ChatResult(
                answer=INSUFFICIENT_INFO_MESSAGE, sources=[], retrieval_query=retrieval_query, observability=event
            )

        candidates = filter_suspicious_chunks(candidates)
        if not candidates:
            event = self._emit(
                question=question,
                knowledge_base_id=knowledge_base_id,
                retrieval_query=retrieval_query,
                outcome="insufficient_information",
                total_start=total_start,
                retrieval_latency_ms=retrieval_latency_ms,
            )
            return ChatResult(
                answer=INSUFFICIENT_INFO_MESSAGE, sources=[], retrieval_query=retrieval_query, observability=event
            )

        rerank_latency_ms = 0.0
        if should_rerank:
            rerank_start = time.perf_counter()
            sources = await self._reranker.rerank(retrieval_query, candidates, effective_top_k)
            rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000
        else:
            sources = candidates[:effective_top_k]

        sources = limit_to_token_budget(sources, self._max_context_tokens)
        context = build_context(sources)
        prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(question) + count_tokens(context)

        generation_start = time.perf_counter()
        try:
            # The LLM still sees the user's actual latest question as user_prompt —
            # only the retrieval step uses the rewritten, context-resolved query.
            answer = await self._llm_provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=question,
                context=context,
            )
        except Exception as exc:
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000
            self._emit(
                question=question,
                knowledge_base_id=knowledge_base_id,
                retrieval_query=retrieval_query,
                outcome="error",
                total_start=total_start,
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                generation_latency_ms=generation_latency_ms,
                sources=sources,
                error=str(exc),
                prompt_tokens=prompt_tokens,
            )
            raise
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        completion_tokens = count_tokens(answer)
        event = self._emit(
            question=question,
            knowledge_base_id=knowledge_base_id,
            retrieval_query=retrieval_query,
            outcome="answered",
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            generation_latency_ms=generation_latency_ms,
            sources=sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return ChatResult(answer=answer, sources=sources, retrieval_query=retrieval_query, observability=event)
