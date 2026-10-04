import re

from app.generation.base import LLMProvider
from app.schemas.chat import ConversationTurn

QUERY_REWRITE_SYSTEM_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up into a single standalone question that includes any context needed to understand it without \
the history (resolve pronouns like "it"/"those" and implicit subjects). Return ONLY the rewritten \
question and nothing else. If the follow-up is already standalone, return it unchanged."""


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the and are was were for with you your have has had does did not any all can could would should will "
    "what which who whom whose when where why how this that these those there their them they its it's "
    "from into about than then also just only some more most much many very show tell give please".split()
)


# Words that point back into the conversation ("how much is it?"), so a
# question made only of them still needs the history to be understood.
_REFERRING = frozenset("it its they them those these that this one ones he she him her there same".split())


def _needs_history(question: str) -> bool:
    words = _WORD.findall(question.lower())
    return bool(_content_words(question)) or any(word in _REFERRING for word in words)


def _content_words(text: str) -> set[str]:
    words = (word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS)
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words}


class QueryRewriter:
    """Turns a follow-up question plus recent conversation history into a
    standalone retrieval query — e.g. "Which sizes are available?" after a
    question about Argentina jerseys becomes "Which sizes are available for
    Argentina jerseys?".

    History is capped to the last `max_history_turns` turns before being sent
    anywhere — conversations aren't allowed to grow the prompt unboundedly.

    Falls back to naive concatenation of the previous question with the new
    one if the configured LLM doesn't produce a usable rewrite (this always
    happens with StubLLMProvider, which has no real language understanding —
    documented and tested explicitly, not silently masked).
    """

    def __init__(self, llm_provider: LLMProvider, max_history_turns: int = 3):
        self._llm_provider = llm_provider
        self._max_history_turns = max_history_turns

    async def rewrite(self, question: str, history: list[ConversationTurn]) -> str:
        # "hi", "ok", "thanks": nothing to resolve, and a small model will
        # happily turn them into a question about the last product discussed.
        if not history or not _needs_history(question):
            return question

        recent_history = history[-self._max_history_turns :]
        history_text = "\n".join(f"User: {turn.question}\nAssistant: {turn.answer}" for turn in recent_history)

        rewritten = await self._llm_provider.generate(
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            user_prompt=question,
            context=history_text,
        )
        rewritten = rewritten.strip()

        if self._looks_like_valid_rewrite(rewritten) and self._keeps_the_question(question, rewritten):
            return rewritten
        return self._fallback_concatenation(question, recent_history)

    @staticmethod
    def _looks_like_valid_rewrite(text: str) -> bool:
        return bool(text) and "[stub-llm]" not in text

    @staticmethod
    def _keeps_the_question(question: str, rewritten: str) -> bool:
        """A rewrite should add context to the question, not replace it. Small
        models sometimes answer or paraphrase the history instead ("last
        order?" -> "What size of <product> are you interested in?"), which
        then retrieves that product for an unrelated question. At least half
        of the question's own content words must survive."""
        wanted = _content_words(question)
        if not wanted:
            return True
        return len(wanted & _content_words(rewritten)) * 2 >= len(wanted)

    @staticmethod
    def _fallback_concatenation(question: str, history: list[ConversationTurn]) -> str:
        return f"{history[-1].question} {question}"
