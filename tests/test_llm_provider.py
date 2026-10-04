import asyncio

import pytest

from app.core.config import Settings
from app.generation.base import LLMProvider
from app.generation.factory import get_llm_provider
from app.generation.stub_provider import StubLLMProvider


def test_stub_provider_echoes_context_preview():
    provider = StubLLMProvider()
    response = asyncio.run(
        provider.generate(
            system_prompt="Answer using only the context.",
            user_prompt="Do you have Argentina jerseys?",
            context="Argentina Home Jersey 2026 is available in sizes S, M, L.",
        )
    )

    assert "Argentina Home Jersey 2026" in response
    assert "stub" in response.lower()
    assert "[1]" in response


def test_stub_provider_flags_missing_context():
    provider = StubLLMProvider()
    response = asyncio.run(provider.generate("system", "user", context=""))

    assert "insufficient information" in response.lower()


def test_stub_provider_is_deterministic():
    provider = StubLLMProvider()
    first = asyncio.run(provider.generate("s", "u", "Some context line."))
    second = asyncio.run(provider.generate("s", "u", "Some context line."))

    assert first == second


def test_factory_returns_stub_provider_for_stub_config():
    provider = get_llm_provider(Settings(llm_provider="stub"))
    assert isinstance(provider, StubLLMProvider)
    assert isinstance(provider, LLMProvider)


def test_factory_raises_clear_error_for_unimplemented_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider(Settings(llm_provider="no-such-llm"))
