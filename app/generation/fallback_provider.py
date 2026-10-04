import logging

from app.generation.base import LLMProvider

logger = logging.getLogger(__name__)


class FallbackLLMProvider(LLMProvider):
    """Tries `primary`; if it raises for any reason (network, auth, rate limit,
    refusal, empty output), logs why and answers with `fallback` instead, so a
    hosted-model outage degrades the chat rather than breaking it."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self._primary = primary
        self._fallback = fallback

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        try:
            return await self._primary.generate(system_prompt, user_prompt, context)
        except Exception as exc:
            logger.warning(
                "primary LLM failed (%s: %s); answering with the fallback provider",
                type(exc).__name__,
                exc,
            )
            return await self._fallback.generate(system_prompt, user_prompt, context)
