from app.guardrails.patterns import INJECTION_PATTERNS, matches_any
from app.schemas.retrieval import RetrievedChunk


def filter_suspicious_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drops retrieved chunks whose *content* (not the user's question)
    contains injection-like patterns. Retrieved text came from a data source,
    not the operator — it's untrusted the same way user input is, and gets
    the same scrutiny before being placed in the LLM's context.
    """
    return [chunk for chunk in chunks if matches_any(chunk.content, INJECTION_PATTERNS) is None]
