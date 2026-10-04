import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.generation.base import LLMRefusedError
from app.generation.factory import get_llm_provider, resolve_llm_identity
from app.generation.fallback_provider import FallbackLLMProvider
from app.generation.ollama_provider import OllamaLLMProvider
from app.generation.openai_compatible_provider import OpenAICompatibleLLMProvider

# Pin every LLM setting so the host's .env can't leak into these tests.
_BASE = dict(
    llm_model="",
    llm_base_url="",
    anthropic_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    qwen_api_key=None,
    llm_api_key=None,
)


def _settings(**overrides) -> Settings:
    return Settings(**{**_BASE, **overrides})


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(provider, context="[1] Jersey\nPrice: 275"):
    return asyncio.run(provider.generate("SYS", "price?", context))


# --- OpenAI-compatible ------------------------------------------------------


def test_openai_compatible_sends_chat_completions_request_with_fenced_context():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " It costs 275 [1]. "}, "finish_reason": "stop"}]})

    provider = OpenAICompatibleLLMProvider("https://api.example.com/v1/", "m-1", api_key="k", client=_client(handler))

    assert _run(provider) == "It costs 275 [1]."
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m-1"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["messages"][1]["content"].startswith("<context>\n[1] Jersey")


def test_openai_compatible_omits_auth_header_without_a_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _run(OpenAICompatibleLLMProvider("http://local/v1", "m", client=_client(handler))) == "ok"


def test_openai_compatible_raises_on_content_filter_and_empty_and_http_error():
    filtered = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}))
    empty = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]}))
    unauthorized = _client(lambda r: httpx.Response(401, json={"error": "bad key"}))

    with pytest.raises(LLMRefusedError):
        _run(OpenAICompatibleLLMProvider("http://x", "m", client=filtered))
    with pytest.raises(RuntimeError):
        _run(OpenAICompatibleLLMProvider("http://x", "m", client=empty))
    with pytest.raises(httpx.HTTPStatusError):
        _run(OpenAICompatibleLLMProvider("http://x", "m", client=unauthorized))


# --- Ollama -----------------------------------------------------------------


def test_ollama_uses_native_chat_api_with_context_window():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "275 [1]"}, "done_reason": "stop"})

    provider = OllamaLLMProvider("http://localhost:11434", "qwen2.5:1.5b", num_ctx=8192, client=_client(handler))

    assert _run(provider) == "275 [1]"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["model"] == "qwen2.5:1.5b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_ctx"] == 8192


def test_ollama_explains_a_missing_model():
    provider = OllamaLLMProvider("http://o", "qwen2.5:1.5b", client=_client(lambda r: httpx.Response(404, json={"error": "not found"})))

    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:1.5b"):
        _run(provider)


# --- factory ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "key_setting", "model"),
    [
        ("openai", "openai_api_key", "gpt-4o-mini"),
        ("gemini", "gemini_api_key", "gemini-2.5-flash"),
        ("qwen", "qwen_api_key", "qwen-plus"),
    ],
)
def test_hosted_vendors_use_default_model_and_need_their_key(provider, key_setting, model):
    assert resolve_llm_identity(_settings(llm_provider=provider)) == (provider, model)
    assert isinstance(get_llm_provider(_settings(llm_provider=provider, **{key_setting: "k"})), FallbackLLMProvider)
    with pytest.raises(ValueError, match=key_setting.upper()):
        get_llm_provider(_settings(llm_provider=provider))


def test_ollama_needs_no_key_and_defaults_to_qwen():
    settings = _settings(llm_provider="ollama")

    assert resolve_llm_identity(settings) == ("ollama", "qwen2.5:1.5b")
    assert isinstance(get_llm_provider(settings), FallbackLLMProvider)


def test_explicit_model_overrides_the_default():
    assert resolve_llm_identity(_settings(llm_provider="ollama", llm_model="llama3.2:3b")) == ("ollama", "llama3.2:3b")


def test_openai_compatible_requires_base_url_and_model():
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        get_llm_provider(_settings(llm_provider="openai_compatible", llm_model="m"))
    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_llm_provider(_settings(llm_provider="openai_compatible", llm_base_url="http://x/v1"))
    provider = get_llm_provider(_settings(llm_provider="openai_compatible", llm_base_url="http://x/v1", llm_model="m"))
    assert isinstance(provider, FallbackLLMProvider)


def test_an_unreachable_model_degrades_to_the_extractive_answer():
    failing = OllamaLLMProvider("http://o", "m", client=_client(lambda r: httpx.Response(500)))
    from app.generation.extractive_provider import ExtractiveLLMProvider
    from app.generation.prompts import SYSTEM_PROMPT

    answer = asyncio.run(FallbackLLMProvider(failing, ExtractiveLLMProvider()).generate(SYSTEM_PROMPT, "price?", "[1] Jersey\nPrice: 275"))

    assert "**Jersey** [1]" in answer


def test_new_api_keys_are_never_shown_in_settings_repr():
    text = repr(_settings(openai_api_key="sk-a", gemini_api_key="sk-b", qwen_api_key="sk-c", llm_api_key="sk-d"))
    assert not any(secret in text for secret in ("sk-a", "sk-b", "sk-c", "sk-d"))
