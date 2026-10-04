from dataclasses import dataclass

from pydantic import SecretStr

from app.core.config import Settings
from app.generation.anthropic_provider import AnthropicLLMProvider
from app.generation.base import LLMProvider
from app.generation.extractive_provider import ExtractiveLLMProvider
from app.generation.fallback_provider import FallbackLLMProvider
from app.generation.ollama_provider import OllamaLLMProvider
from app.generation.openai_compatible_provider import OpenAICompatibleLLMProvider
from app.generation.stub_provider import StubLLMProvider

EXTRACTIVE_MODEL_NAME = "extractive-v1"
STUB_MODEL_NAME = "stub-echo-v1"


@dataclass(frozen=True)
class _OpenAICompatibleVendor:
    base_url: str  # "" = LLM_BASE_URL must be set
    default_model: str  # "" = LLM_MODEL must be set
    key_setting: str  # Settings attribute holding the vendor's API key
    key_required: bool = True


# Hosted vendors reached through OpenAI-compatible Chat Completions. Another
# compatible vendor (Groq, OpenRouter, DeepSeek, vLLM, ...) needs no code:
# LLM_PROVIDER=openai_compatible + LLM_BASE_URL + LLM_MODEL (+ LLM_API_KEY).
_OPENAI_COMPATIBLE_VENDORS: dict[str, _OpenAICompatibleVendor] = {
    "openai": _OpenAICompatibleVendor("https://api.openai.com/v1", "gpt-4o-mini", "openai_api_key"),
    "gemini": _OpenAICompatibleVendor(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", "gemini_api_key"
    ),
    "qwen": _OpenAICompatibleVendor(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus", "qwen_api_key"
    ),
    "openai_compatible": _OpenAICompatibleVendor("", "", "llm_api_key", key_required=False),
}

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "ollama": "qwen2.5:1.5b",
    **{name: vendor.default_model for name, vendor in _OPENAI_COMPATIBLE_VENDORS.items()},
}

SUPPORTED_PROVIDERS = ("auto", "anthropic", *_OPENAI_COMPATIBLE_VENDORS, "ollama", "extractive", "stub")


def _secret(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    return value.get_secret_value().strip() or None


def _resolve_provider(settings: Settings) -> str:
    provider = settings.llm_provider.strip().lower()
    if provider == "auto":
        return "anthropic" if _secret(settings.anthropic_api_key) else "extractive"
    return provider


def resolve_llm_identity(settings: Settings) -> tuple[str, str]:
    """(provider, model) that will actually answer — what gets logged and shown
    to users. `auto` means Claude when ANTHROPIC_API_KEY is set, otherwise the
    offline extractive provider. An empty LLM_MODEL means the provider's default."""
    provider = _resolve_provider(settings)
    if provider == "extractive":
        return "extractive", EXTRACTIVE_MODEL_NAME
    if provider == "stub":
        return "stub", settings.llm_model or STUB_MODEL_NAME
    return provider, settings.llm_model or _DEFAULT_MODELS.get(provider, "")


def _answering_provider(settings: Settings, provider: str, model: str) -> LLMProvider:
    if provider == "anthropic":
        api_key = _secret(settings.anthropic_api_key)
        if api_key is None:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")
        return AnthropicLLMProvider(api_key, model)

    if provider == "ollama":
        return OllamaLLMProvider(
            settings.llm_base_url or settings.ollama_base_url,
            model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            num_ctx=settings.ollama_num_ctx,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    vendor = _OPENAI_COMPATIBLE_VENDORS.get(provider)
    if vendor is None:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider!r}. Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    base_url = settings.llm_base_url or vendor.base_url
    if not base_url:
        raise ValueError(f"LLM_PROVIDER={provider} requires LLM_BASE_URL to be set.")
    if not model:
        raise ValueError(f"LLM_PROVIDER={provider} requires LLM_MODEL to be set.")
    api_key = _secret(getattr(settings, vendor.key_setting))
    if api_key is None and vendor.key_required:
        raise ValueError(f"LLM_PROVIDER={provider} requires {vendor.key_setting.upper()} to be set.")
    return OpenAICompatibleLLMProvider(
        base_url,
        model,
        api_key=api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def get_llm_provider(settings: Settings) -> LLMProvider:
    provider, model = resolve_llm_identity(settings)

    if provider == "stub":
        return StubLLMProvider(model)

    if provider == "extractive":
        return ExtractiveLLMProvider()

    # If the model is unreachable/rate-limited/refuses, answer from the
    # retrieved text rather than failing the whole chat request.
    return FallbackLLMProvider(_answering_provider(settings, provider, model), ExtractiveLLMProvider())
