import asyncio

from app.generation.base import LLMProvider
from app.generation.query_rewriter import QueryRewriter
from app.generation.stub_provider import StubLLMProvider
from app.schemas.chat import ConversationTurn


class FakeLLMProvider(LLMProvider):
    """Returns a fixed, hand-crafted response so tests can control exactly
    what the rewriter receives, independent of any real LLM's behavior."""

    def __init__(self, response: str):
        self._response = response
        self.last_context = None
        self.last_user_prompt = None

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        self.last_context = context
        self.last_user_prompt = user_prompt
        return self._response


def test_no_history_returns_question_unchanged():
    rewriter = QueryRewriter(FakeLLMProvider("irrelevant"))
    result = asyncio.run(rewriter.rewrite("Do you have Argentina jerseys?", history=[]))

    assert result == "Do you have Argentina jerseys?"


def test_valid_llm_rewrite_is_used_as_is():
    llm = FakeLLMProvider("Which sizes are available for Argentina jerseys?")
    rewriter = QueryRewriter(llm)
    history = [ConversationTurn(question="Do you have Argentina jerseys?", answer="Yes, several styles.")]

    result = asyncio.run(rewriter.rewrite("Which sizes are available?", history))

    assert result == "Which sizes are available for Argentina jerseys?"
    assert llm.last_user_prompt == "Which sizes are available?"


def test_stub_llm_falls_back_to_concatenation():
    """StubLLMProvider has no real language understanding — its placeholder
    output is detected as unusable and the rewriter falls back instead of
    silently returning garbage as the retrieval query."""
    rewriter = QueryRewriter(StubLLMProvider())
    history = [ConversationTurn(question="Do you have Argentina jerseys?", answer="Yes.")]

    result = asyncio.run(rewriter.rewrite("Which sizes are available?", history))

    assert result == "Do you have Argentina jerseys? Which sizes are available?"


def test_empty_llm_response_falls_back():
    rewriter = QueryRewriter(FakeLLMProvider("   "))
    history = [ConversationTurn(question="Do you have Argentina jerseys?", answer="Yes.")]

    result = asyncio.run(rewriter.rewrite("Which sizes are available?", history))

    assert result == "Do you have Argentina jerseys? Which sizes are available?"


def test_history_is_capped_to_max_history_turns():
    llm = FakeLLMProvider("standalone question")
    rewriter = QueryRewriter(llm, max_history_turns=2)
    history = [
        ConversationTurn(question="q1", answer="a1"),
        ConversationTurn(question="q2", answer="a2"),
        ConversationTurn(question="q3", answer="a3"),
    ]

    asyncio.run(rewriter.rewrite("follow-up", history))

    assert "q1" not in llm.last_context
    assert "q2" in llm.last_context
    assert "q3" in llm.last_context


def test_fallback_uses_only_the_most_recent_turn():
    rewriter = QueryRewriter(StubLLMProvider())
    history = [
        ConversationTurn(question="q1", answer="a1"),
        ConversationTurn(question="q2", answer="a2"),
    ]

    result = asyncio.run(rewriter.rewrite("follow-up", history))

    assert result == "q2 follow-up"


def test_a_rewrite_that_drops_the_question_is_rejected():
    """Seen with qwen2.5:1.5b: the follow-up was replaced by a question about the
    product in the history, which then retrieved that product."""
    llm = FakeLLMProvider("What size of ARGENTINA 2018 AWAY MESSI are you interested in?")
    history = [ConversationTurn(question="Show me Argentina jerseys", answer="ARGENTINA 2018 AWAY MESSI [1]")]

    result = asyncio.run(QueryRewriter(llm).rewrite("last order?", history))

    assert result == "Show me Argentina jerseys last order?"


def test_a_rewrite_that_resolves_a_pronoun_is_kept():
    llm = FakeLLMProvider("How much is the Argentina Home Jersey?")
    history = [ConversationTurn(question="Do you have Argentina jerseys?", answer="Argentina Home Jersey [1]")]

    assert asyncio.run(QueryRewriter(llm).rewrite("How much is it?", history)) == "How much is the Argentina Home Jersey?"


def test_small_talk_is_not_rewritten_from_the_history():
    """Seen with qwen2.5:1.5b: "hi" became "What size are the available options for the product?"."""
    llm = FakeLLMProvider("What size are the available options for the product?")
    history = [ConversationTurn(question="Show me Argentina jerseys", answer="ARGENTINA 2018 AWAY MESSI [1]")]

    assert asyncio.run(QueryRewriter(llm).rewrite("hi", history)) == "hi"
    assert llm.last_user_prompt is None  # the LLM wasn't even asked
