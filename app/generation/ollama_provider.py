import httpx

from app.generation.base import LLMProvider, fence_context
from app.generation.openai_compatible_provider import _shared_client


class OllamaLLMProvider(LLMProvider):
    """Answers with a locally served Ollama model (e.g. `qwen2.5:1.5b`) via
    Ollama's native `/api/chat` — free, offline, no API key.

    The native API is used instead of Ollama's OpenAI-compatible `/v1` because
    only it accepts `num_ctx`. Ollama's default context window can be smaller
    than system prompt + MAX_CONTEXT_TOKENS of retrieved passages + answer, and
    it silently truncates the prompt's start (the instructions) when that
    happens.

    Errors (Ollama not running, model not pulled, empty output) propagate;
    wrap this in a FallbackLLMProvider if the chat should degrade instead.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_ctx: int = 4096,
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._options = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
        self._client = client or _shared_client(timeout_seconds)

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        response = await self._client.post(
            self._url,
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fence_context(user_prompt, context)},
                ],
                "stream": False,
                "options": self._options,
            },
        )
        if response.status_code == 404:
            raise RuntimeError(f"Ollama has no model {self._model!r}; run `ollama pull {self._model}`")
        response.raise_for_status()

        body = response.json()
        text = (body.get("message", {}).get("content") or "").strip()
        if not text:
            raise RuntimeError(f"the model returned no text (done_reason={body.get('done_reason')})")
        return text
