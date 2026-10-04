import re

from app.generation.base import LLMProvider

QUERY_BROADEN_SYSTEM_PROMPT = """The previous search of the knowledge base for this question returned no \
closely-matching results. Rewrite the question using different, broader, or more general terms that might \
match how the knowledge base actually describes this topic, while preserving the original intent. Return \
ONLY the rewritten question and nothing else."""

_LEADING_INTERROGATIVE = re.compile(r"^(do|does|is|are|can|could|would|will)\s+(you\s+)?", re.IGNORECASE)


class QueryBroadener:
    """Used only on the agentic retry loop's NO branch (Plan Set 23): when
    retrieval comes back too weak, rewrites the query with broader/alternate
    terms rather than the conversation-history resolution QueryRewriter does
    (Plan Set 16) — a different problem (the wording didn't match the corpus)
    needing a different rewrite strategy.

    Falls back to a simple heuristic (stripping a leading interrogative like
    "Do you have") if the LLM doesn't produce a usable rewrite, or produces
    the same query unchanged — same documented, tested fallback philosophy as
    QueryRewriter, and what keeps a retry loop from spinning on an LLM that
    just echoes its input back.
    """

    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    async def broaden(self, question: str, previous_query: str, attempt: int) -> str:
        rewritten = await self._llm_provider.generate(
            system_prompt=QUERY_BROADEN_SYSTEM_PROMPT,
            user_prompt=question,
            context=f"Previous search query (attempt {attempt}): {previous_query}",
        )
        rewritten = rewritten.strip()

        if self._looks_like_valid_rewrite(rewritten, previous_query):
            return rewritten
        return self._fallback_broaden(previous_query)

    @staticmethod
    def _looks_like_valid_rewrite(text: str, previous_query: str) -> bool:
        return bool(text) and "[stub-llm]" not in text and text.lower() != previous_query.lower()

    @staticmethod
    def _fallback_broaden(previous_query: str) -> str:
        stripped = _LEADING_INTERROGATIVE.sub("", previous_query).rstrip("?").strip()
        return stripped or previous_query
