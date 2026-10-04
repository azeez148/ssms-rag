import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agentic.query_broadener import QueryBroadener
from app.agentic.state import AgenticState
from app.generation.base import LLMProvider
from app.generation.context_builder import build_context
from app.generation.prompts import SYSTEM_PROMPT
from app.generation.query_rewriter import QueryRewriter
from app.guardrails.content_guard import filter_suspicious_chunks
from app.guardrails.context_budget import limit_to_token_budget
from app.guardrails.input_guard import REFUSAL_MESSAGE, InputGuard
from app.guardrails.patterns import INJECTION_PATTERNS, SENSITIVE_DATA_PATTERNS, matches_any
from app.guardrails.retrieval_quality import retrieval_is_too_weak
from app.ingestion.chunking import count_tokens
from app.observability.cost import estimate_cost
from app.observability.events import ChatObservabilityEvent, RetrievedChunkLog
from app.observability.logger import log_chat_event
from app.reranking.base import Reranker
from app.retrieval.base import Retriever
from app.schemas.chat import ConversationTurn
from app.schemas.retrieval import RetrievedChunk
from app.services.rag_service import INSUFFICIENT_INFO_MESSAGE, ChatResult

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Question -> Analyze -> Retrieve -> Evaluate retrieval -> enough
    context? -> (No: Rewrite query -> Retrieve again, bounded) -> Generate ->
    Validate answer -> Respond.

    Plan Set 23's agentic alternative to RAGService's straight-line pipeline:
    the same building blocks (retriever, reranker, LLM, guardrails), wired
    into a LangGraph state machine so a weak first retrieval can trigger a
    bounded number of query-rewrite-and-retry cycles instead of immediately
    giving up with "insufficient information."

    Returns the same `ChatResult` RAGService does, so callers (the /api/chat
    routes) don't need to know which pipeline answered a given request.
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
        max_retries: int = 2,
        embedding_model_name: str = "unknown",
        llm_provider_name: str = "unknown",
        llm_model_name: str = "unknown",
        reranker_model_name: str | None = None,
    ):
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._reranker = reranker
        self._query_rewriter = query_rewriter
        self._query_broadener = QueryBroadener(llm_provider)
        self._top_k = top_k
        self._rerank_candidate_multiplier = rerank_candidate_multiplier
        self._input_guard = input_guard or InputGuard()
        self._similarity_threshold = similarity_threshold
        self._max_context_tokens = max_context_tokens
        self._max_retries = max_retries
        self._embedding_model_name = embedding_model_name
        self._llm_provider_name = llm_provider_name
        self._llm_model_name = llm_model_name
        self._reranker_model_name = reranker_model_name
        self._graph = self._build_graph()

    # ---- graph construction -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgenticState)

        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("evaluate_retrieval", self._evaluate_retrieval)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate", self._generate)
        graph.add_node("validate_answer", self._validate_answer)
        graph.add_node("respond", self._respond)

        graph.add_edge(START, "analyze")
        graph.add_conditional_edges(
            "analyze", self._route_after_analyze, {"retrieve": "retrieve", "respond": "respond"}
        )
        graph.add_edge("retrieve", "evaluate_retrieval")
        graph.add_conditional_edges(
            "evaluate_retrieval",
            self._route_after_evaluate,
            {"generate": "generate", "rewrite_query": "rewrite_query", "respond": "respond"},
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", "validate_answer")
        graph.add_edge("validate_answer", "respond")
        graph.add_edge("respond", END)

        return graph.compile()

    # ---- nodes ---------------------------------------------------------------

    async def _analyze(self, state: AgenticState) -> dict[str, Any]:
        guard_result = self._input_guard.check(state["question"])
        if guard_result.blocked:
            return {
                "blocked": True,
                "blocked_category": guard_result.category,
                "answer": guard_result.reason or INSUFFICIENT_INFO_MESSAGE,
            }

        if self._query_rewriter is not None and state["history"]:
            retrieval_query = await self._query_rewriter.rewrite(state["question"], state["history"])
        else:
            retrieval_query = state["question"]
        return {"retrieval_query": retrieval_query}

    @staticmethod
    def _route_after_analyze(state: AgenticState) -> str:
        return "respond" if state["blocked"] else "retrieve"

    async def _retrieve(self, state: AgenticState) -> dict[str, Any]:
        start = time.perf_counter()
        candidates = await self._retriever.retrieve(
            state["retrieval_query"], state["knowledge_base_id"], top_k=state["fetch_k"], filters=state["filters"]
        )
        latency_ms = (time.perf_counter() - start) * 1000
        candidates = filter_suspicious_chunks(candidates)
        return {
            "candidates": candidates,
            "retrieval_latency_ms": state["retrieval_latency_ms"] + latency_ms,
        }

    async def _evaluate_retrieval(self, state: AgenticState) -> dict[str, Any]:
        candidates = state["candidates"]
        enough_context = bool(candidates) and not retrieval_is_too_weak(candidates, self._similarity_threshold)
        return {"enough_context": enough_context}

    def _route_after_evaluate(self, state: AgenticState) -> str:
        if state["enough_context"]:
            return "generate"
        if state["retries"] < state["max_retries"]:
            return "rewrite_query"
        # Retries exhausted — this is the loop-prevention exit the plan set
        # asked for: give up honestly rather than retry forever.
        return "respond"

    async def _rewrite_query(self, state: AgenticState) -> dict[str, Any]:
        attempt = state["retries"] + 1
        broadened = await self._query_broadener.broaden(state["question"], state["retrieval_query"], attempt)
        logger.info(
            "agentic retry %d/%d: %r -> %r", attempt, state["max_retries"], state["retrieval_query"], broadened
        )
        return {"retrieval_query": broadened, "retries": attempt}

    async def _generate(self, state: AgenticState) -> dict[str, Any]:
        candidates = state["candidates"]
        should_rerank = state["rerank"] and self._reranker is not None

        rerank_latency_ms = 0.0
        if should_rerank:
            start = time.perf_counter()
            sources = await self._reranker.rerank(state["retrieval_query"], candidates, state["top_k"])
            rerank_latency_ms = (time.perf_counter() - start) * 1000
        else:
            sources = candidates[: state["top_k"]]

        sources = limit_to_token_budget(sources, self._max_context_tokens)
        context = build_context(sources)
        prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(state["question"]) + count_tokens(context)

        start = time.perf_counter()
        # The LLM still sees the user's actual latest question, not the
        # rewritten/broadened retrieval query — same split as RAGService.
        answer = await self._llm_provider.generate(
            system_prompt=SYSTEM_PROMPT, user_prompt=state["question"], context=context
        )
        generation_latency_ms = (time.perf_counter() - start) * 1000

        return {
            "sources": sources,
            "answer": answer,
            "rerank_latency_ms": rerank_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": count_tokens(answer),
        }

    async def _validate_answer(self, state: AgenticState) -> dict[str, Any]:
        """Output-guard pass: a defense-in-depth complement to the input
        guard (Plan Set 17) and content guard (also Plan Set 17) — this time
        checking what the LLM actually produced, in case adversarial content
        that slipped past the content guard got the model to repeat it."""
        answer = state["answer"]
        if not answer or not answer.strip():
            return {"answer": INSUFFICIENT_INFO_MESSAGE, "validated": False}

        if matches_any(answer, INJECTION_PATTERNS) or matches_any(answer, SENSITIVE_DATA_PATTERNS):
            return {
                "answer": REFUSAL_MESSAGE,
                "blocked": True,
                "blocked_category": "output_guard",
                "validated": False,
            }

        return {"validated": True}

    async def _respond(self, state: AgenticState) -> dict[str, Any]:
        return {}

    # ---- public entry point ---------------------------------------------------

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
        effective_top_k = top_k or self._top_k
        should_rerank = rerank and self._reranker is not None
        fetch_k = effective_top_k * self._rerank_candidate_multiplier if should_rerank else effective_top_k

        initial_state: AgenticState = {
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "top_k": effective_top_k,
            "fetch_k": fetch_k,
            "filters": filters,
            "rerank": rerank,
            "history": history or [],
            "retrieval_query": question,
            "candidates": [],
            "retries": 0,
            "max_retries": self._max_retries,
            "enough_context": False,
            "sources": [],
            "answer": "",
            "blocked": False,
            "blocked_category": None,
            "validated": False,
            "retrieval_latency_ms": 0.0,
            "rerank_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        final_state = await self._graph.ainvoke(initial_state)

        if final_state["blocked"]:
            outcome = "blocked"
            answer_text = final_state["answer"] or INSUFFICIENT_INFO_MESSAGE
            sources: list[RetrievedChunk] = []
        elif not final_state["answer"]:
            outcome = "insufficient_information"
            answer_text = INSUFFICIENT_INFO_MESSAGE
            sources = []
        else:
            outcome = "answered"
            answer_text = final_state["answer"]
            sources = final_state["sources"]

        event = self._emit(
            question=question,
            knowledge_base_id=knowledge_base_id,
            retrieval_query=final_state["retrieval_query"],
            outcome=outcome,
            total_start=total_start,
            retrieval_latency_ms=final_state["retrieval_latency_ms"],
            rerank_latency_ms=final_state["rerank_latency_ms"],
            generation_latency_ms=final_state["generation_latency_ms"],
            sources=sources,
            blocked_category=final_state["blocked_category"],
            prompt_tokens=final_state["prompt_tokens"],
            completion_tokens=final_state["completion_tokens"],
        )

        return ChatResult(
            answer=answer_text,
            sources=sources,
            retrieval_query=final_state["retrieval_query"],
            blocked=final_state["blocked"],
            blocked_category=final_state["blocked_category"],
            observability=event,
        )
