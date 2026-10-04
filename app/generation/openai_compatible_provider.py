from functools import lru_cache

import httpx

from app.generation.base import LLMProvider, LLMRefusedError, fence_context


@lru_cache(maxsize=8)
def _shared_client(timeout_seconds: float) -> httpx.AsyncClient:
    # get_llm_provider() runs on every request; reusing one client keeps the
    # HTTP connection pool alive instead of rebuilding it per chat message.
    return httpx.AsyncClient(timeout=timeout_seconds)


class OpenAICompatibleLLMProvider(LLMProvider):
    """Answers through any `POST {base_url}/chat/completions` endpoint that
    speaks the OpenAI Chat Completions format: OpenAI itself, Google Gemini's
    OpenAI-compatible endpoint, Qwen on Alibaba DashScope, Groq, OpenRouter,
    DeepSeek, Together, vLLM, LM Studio, and Ollama's `/v1`.

    Talks plain HTTP (httpx) rather than a vendor SDK, so adding another
    compatible vendor is a base URL + model name, not a new dependency.

    Errors (network, auth, rate limit, content filter, empty output) propagate;
    wrap this in a FallbackLLMProvider if the chat should degrade instead.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client or _shared_client(timeout_seconds)

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = await self._client.post(
            self._url,
            headers=headers,
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fence_context(user_prompt, context)},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )
        response.raise_for_status()

        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMRefusedError("the model's output was blocked by a content filter")

        text = (choice.get("message", {}).get("content") or "").strip()
        if not text:
            raise RuntimeError(f"the model returned no text (finish_reason={choice.get('finish_reason')})")
        return text
