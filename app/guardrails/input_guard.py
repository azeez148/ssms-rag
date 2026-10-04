from dataclasses import dataclass

from app.guardrails.patterns import INJECTION_PATTERNS, SENSITIVE_DATA_PATTERNS, matches_any

REFUSAL_MESSAGE = (
    "I can't help with that request. I can only answer questions about products in this "
    "catalog using the store's own knowledge base — I don't have access to credentials, "
    "internal systems, or the ability to change my instructions."
)


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str | None = None
    category: str | None = None


class InputGuard:
    """Screens the user's raw question before it reaches retrieval or the LLM.

    Catches prompt injection attempts and requests for sensitive data
    (credentials, schemas, raw SQL) with simple pattern matching. This is not
    a substitute for a real safety-tuned model — it's a cheap, deterministic
    first line of defense that costs nothing and runs before any retrieval or
    generation expense is spent.
    """

    def check(self, question: str) -> GuardrailResult:
        if matches_any(question, INJECTION_PATTERNS):
            return GuardrailResult(blocked=True, reason=REFUSAL_MESSAGE, category="prompt_injection")

        if matches_any(question, SENSITIVE_DATA_PATTERNS):
            return GuardrailResult(blocked=True, reason=REFUSAL_MESSAGE, category="sensitive_data_request")

        return GuardrailResult(blocked=False)
