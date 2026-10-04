from abc import ABC, abstractmethod


class LLMRefusedError(RuntimeError):
    """The model declined to answer (a refusal or content-filter stop)."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        ...


def fence_context(user_prompt: str, context: str) -> str:
    """The retrieved context is untrusted data, so it's fenced off from the
    question instead of being concatenated into the instructions."""
    if context.strip():
        return f"<context>\n{context}\n</context>\n\n{user_prompt}"
    return user_prompt
