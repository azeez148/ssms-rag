from app.generation.context_builder import build_context
from app.ingestion.chunking import count_tokens
from app.schemas.retrieval import RetrievedChunk


def limit_to_token_budget(chunks: list[RetrievedChunk], max_tokens: int) -> list[RetrievedChunk]:
    """Greedily keeps chunks, in their existing rank order, until adding the
    next one would push the rendered context over max_tokens — measured the
    same way build_context() will actually render them, not just by summing
    each chunk's own token_count in isolation.

    Always keeps at least one chunk, even if it alone exceeds the budget —
    an over-budget answer is better than no context at all. Guards against an
    unusually large top_k or chunk_size silently blowing the LLM's context
    window or per-call cost.
    """
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        candidate = kept + [chunk]
        if kept and count_tokens(build_context(candidate)) > max_tokens:
            break
        kept = candidate
    return kept
