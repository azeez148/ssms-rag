import logging
from functools import lru_cache

import anthropic

from app.generation.base import LLMProvider, LLMRefusedError, fence_context

logger = logging.getLogger(__name__)

@lru_cache(maxsize=4)
def _shared_client(api_key: str) -> anthropic.AsyncAnthropic:
    # get_llm_provider() runs on every request; one client per key keeps the
    # HTTP connection pool alive instead of rebuilding it per chat message.
    return anthropic.AsyncAnthropic(api_key=api_key)


class AnthropicLLMProvider(LLMProvider):
    """Answers with a Claude model through the official Anthropic SDK.

    Thinking is left at the model's default (adaptive on Claude Opus 5) rather
    than disabled — with thinking off, Opus 5 can emit malformed output. Cost
    and latency are controlled with `effort: low` instead, which is plenty for
    answering from a handful of retrieved passages. Thinking tokens count
    against max_tokens, so it's kept generous relative to the answer length.

    Errors (network, auth, rate limit, refusal) propagate; wrap this in a
    FallbackLLMProvider if the chat should degrade instead of failing.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self._client = client or _shared_client(api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        content = fence_context(user_prompt, context)

        kwargs: dict = {}
        if self._supports_effort(self._model):
            kwargs["output_config"] = {"effort": "low"}

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )

        if response.stop_reason == "refusal":
            raise LLMRefusedError("the model declined to answer")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise RuntimeError(f"the model returned no text (stop_reason={response.stop_reason})")
        return text

    @staticmethod
    def _supports_effort(model: str) -> bool:
        # `effort` errors on Haiku 4.5 and older; every current larger model accepts it.
        return "haiku" not in model
