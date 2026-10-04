import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.generation.anthropic_provider import AnthropicLLMProvider, LLMRefusedError
from app.generation.base import LLMProvider
from app.generation.context_builder import build_context
from app.generation.extractive_provider import INSUFFICIENT_INFO, ExtractiveLLMProvider
from app.generation.factory import get_llm_provider, resolve_llm_identity
from app.generation.fallback_provider import FallbackLLMProvider
from app.generation.prompts import SYSTEM_PROMPT
from app.generation.query_rewriter import QueryRewriter
from app.schemas.chat import ConversationTurn
from app.schemas.retrieval import RetrievedChunk

JERSEY = (
    "Product: Argentina Home Jersey\nCategory: Jersey\nDescription: Classic sky-blue stripes.\n"
    "Price: 275\nAvailable sizes: M (3 in stock), L (1 in stock)"
)
SHORTS = "Product: Arsenal Shorts\nCategory: Shorts\nDescription: White training shorts.\nPrice: 90"


def _chunk(title: str, content: str, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        document_id=f"products-{title}",
        chunk_index=index,
        document_title=title,
        source="test",
        knowledge_base_id="kb",
        metadata={},
        similarity=0.9,
        distance=0.1,
    )


def _context() -> str:
    return build_context([_chunk("Argentina Home Jersey", JERSEY), _chunk("Arsenal Shorts", SHORTS)])


def _answer(question: str, context: str) -> str:
    return asyncio.run(ExtractiveLLMProvider().generate(SYSTEM_PROMPT, question, context))


# --- extractive provider ---------------------------------------------------


def test_extractive_lists_each_passage_with_its_citation():
    answer = _answer("Do you have jerseys or shorts?", _context())

    assert "**Argentina Home Jersey** [1]" in answer
    assert "**Arsenal Shorts** [2]" in answer
    assert "- Price: 275" in answer
    assert "- Category: Jersey" in answer


def test_extractive_shows_only_the_fields_the_question_asks_about():
    answer = _answer("Which sizes are available?", _context())

    assert "Available sizes: M (3 in stock), L (1 in stock)" in answer
    assert "Price" not in answer
    assert "Description" not in answer


def test_extractive_does_not_repeat_the_title_as_a_field():
    answer = _answer("Tell me about it", _context())

    assert "Product: Argentina Home Jersey" not in answer


def test_extractive_caps_the_number_of_items_and_says_so():
    chunks = [_chunk(f"Item {i}", f"Name: Item {i}\nPrice: {i}") for i in range(6)]

    answer = _answer("anything", build_context(chunks))

    assert answer.count("**Item") == 4
    assert "Showing the top 4 of 6 matches" in answer


def test_extractive_says_insufficient_information_without_context():
    assert _answer("anything", "") == INSUFFICIENT_INFO


def test_extractive_truncates_very_long_values():
    context = build_context([_chunk("Book", "Description: " + "word " * 200)])

    answer = _answer("Tell me about it", context)

    assert "…" in answer
    assert len(answer) < 400


def test_extractive_returns_nothing_for_non_answer_prompts_so_rewriter_fallbacks_apply():
    provider = ExtractiveLLMProvider()

    assert asyncio.run(provider.generate("Rewrite the question", "Which sizes?", "User: hi")) == ""

    rewriter = QueryRewriter(provider)
    history = [ConversationTurn(question="Do you have Argentina jerseys?", answer="Yes")]
    rewritten = asyncio.run(rewriter.rewrite("Which sizes are available?", history))
    assert rewritten == "Do you have Argentina jerseys? Which sizes are available?"


# --- anthropic provider ----------------------------------------------------


def _fake_client(text: str = "Hello [1]", stop_reason: str = "end_turn") -> SimpleNamespace:
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


def test_anthropic_provider_sends_context_fenced_and_returns_the_text():
    client = _fake_client("  Answer [1]  ")
    provider = AnthropicLLMProvider("key", "claude-opus-5", client=client)

    text = asyncio.run(provider.generate("SYS", "What is it?", "[1] Title\nBody"))

    assert text == "Answer [1]"
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["system"] == "SYS"
    assert kwargs["messages"] == [{"role": "user", "content": "<context>\n[1] Title\nBody\n</context>\n\nWhat is it?"}]
    assert kwargs["output_config"] == {"effort": "low"}
    assert "thinking" not in kwargs and "temperature" not in kwargs


def test_anthropic_provider_omits_context_wrapper_when_there_is_none():
    client = _fake_client()
    provider = AnthropicLLMProvider("key", "claude-opus-5", client=client)

    asyncio.run(provider.generate("SYS", "Hi", ""))

    assert client.messages.create.await_args.kwargs["messages"] == [{"role": "user", "content": "Hi"}]


def test_anthropic_provider_skips_effort_for_haiku():
    client = _fake_client()
    provider = AnthropicLLMProvider("key", "claude-haiku-4-5", client=client)

    asyncio.run(provider.generate("SYS", "Hi", ""))

    assert "output_config" not in client.messages.create.await_args.kwargs


def test_anthropic_provider_raises_on_refusal():
    provider = AnthropicLLMProvider("key", "claude-opus-5", client=_fake_client(stop_reason="refusal"))

    with pytest.raises(LLMRefusedError):
        asyncio.run(provider.generate("SYS", "Hi", ""))


def test_anthropic_provider_raises_when_no_text_comes_back():
    provider = AnthropicLLMProvider("key", "claude-opus-5", client=_fake_client(text="   "))

    with pytest.raises(RuntimeError):
        asyncio.run(provider.generate("SYS", "Hi", ""))


# --- fallback + factory ----------------------------------------------------


class _Boom(LLMProvider):
    async def generate(self, system_prompt, user_prompt, context):
        raise ConnectionError("network down")


def test_fallback_answers_from_the_backup_when_the_primary_fails():
    provider = FallbackLLMProvider(_Boom(), ExtractiveLLMProvider())

    answer = asyncio.run(provider.generate(SYSTEM_PROMPT, "price?", _context()))

    assert "**Argentina Home Jersey** [1]" in answer


def test_fallback_uses_the_primary_when_it_works():
    primary = SimpleNamespace(generate=AsyncMock(return_value="from claude"))
    provider = FallbackLLMProvider(primary, ExtractiveLLMProvider())

    assert asyncio.run(provider.generate("s", "u", "c")) == "from claude"


def test_auto_without_a_key_is_the_offline_provider():
    settings = Settings(llm_provider="auto", anthropic_api_key=None)

    assert resolve_llm_identity(settings) == ("extractive", "extractive-v1")
    assert isinstance(get_llm_provider(settings), ExtractiveLLMProvider)


def test_auto_with_a_key_is_claude_with_offline_fallback():
    settings = Settings(llm_provider="auto", anthropic_api_key="sk-test", llm_model="claude-sonnet-5")

    assert resolve_llm_identity(settings) == ("anthropic", "claude-sonnet-5")
    assert isinstance(get_llm_provider(settings), FallbackLLMProvider)


def test_explicit_anthropic_without_a_key_fails_loudly():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm_provider(Settings(llm_provider="anthropic", anthropic_api_key=None))


def test_the_api_key_is_never_shown_in_settings_repr():
    assert "sk-secret" not in repr(Settings(anthropic_api_key="sk-secret"))


# --- content folded onto one line (how YAML `>` templates are actually stored) ---

FOLDED_PRODUCT = (
    "Product: Adidas 3Line Black Category: Track Pants - Normal Description: Adidas 3Line Black "
    "Price: 600 Available sizes: M (2 in stock), L (1 in stock)"
)
FOLDED_BOOK = (
    "Title: 1984 Author: George Orwell Genre: Dystopian Published: 1949 "
    "Description: A totalitarian superstate uses surveillance to control its citizens."
)


def test_extractive_splits_fields_that_are_folded_onto_one_line():
    context = build_context([_chunk("Adidas 3Line Black", FOLDED_PRODUCT)])

    answer = _answer("Tell me about it", context)

    assert "- Category: Track Pants - Normal" in answer
    assert "- Price: 600" in answer
    assert "- Available sizes: M (2 in stock), L (1 in stock)" in answer
    assert "Product:" not in answer  # duplicate of the title


def test_extractive_answers_a_field_question_from_folded_content():
    context = build_context([_chunk("Adidas 3Line Black", FOLDED_PRODUCT)])

    answer = _answer("Which sizes are available?", context)

    assert answer.splitlines()[-1] == "- Available sizes: M (2 in stock), L (1 in stock)"
    assert "Price" not in answer


def test_extractive_handles_a_book_record_with_a_repeated_word_in_the_value():
    context = build_context([_chunk("1984", FOLDED_BOOK)])

    answer = _answer("Who is the author?", context)

    assert "- Author: George Orwell" in answer
    assert "Genre" not in answer


def test_extractive_does_not_mistake_a_capitalised_value_word_for_a_label():
    # "Normal Description:" must yield label "Description", value ending in "Normal"
    context = build_context([_chunk("X", FOLDED_PRODUCT)])

    answer = _answer("Tell me about it", context)

    assert "Track Pants - Normal" in answer
    assert "Normal Description" not in answer
