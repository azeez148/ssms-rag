from app.generation.base import LLMProvider


class StubLLMProvider(LLMProvider):
    """A deterministic, no-API-key placeholder LLM.

    No account, key, or local model install is required, so the rest of the
    RAG pipeline (prompting, context building, chat wiring in later plan sets)
    can be built and tested now. It does not "understand" anything — it just
    proves the prompt/context reached the provider. It always cites [1] (the
    top retrieved chunk) to keep the citation pipeline (Plan Set 12)
    exercisable end-to-end. Swap for a real Anthropic/OpenAI/Ollama provider
    later by implementing LLMProvider; no other code changes.
    """

    def __init__(self, model_name: str = "stub-echo-v1"):
        self._model_name = model_name

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        if not context.strip():
            return "[stub-llm] No context was provided, so there is insufficient information to answer."

        # Only preview the first context block ([1]) — naively truncating the
        # whole flattened context could bleed into a later block's own "[2]"
        # header, making it look falsely cited.
        first_block = context.strip().split("\n\n")[0]
        preview = " ".join(first_block.split())[:200]
        return (
            "[stub-llm] This is a placeholder response — no real LLM is configured yet. "
            f"Based on [1]: {preview}"
        )
