# Illustrative per-1K-token rates (USD) for common hosted providers — for
# demonstrating cost estimation only, not live pricing; check the provider's
# current pricing page before using this for anything real. Local/free
# providers (stub LLM, sentence-transformers embeddings, the cross-encoder
# reranker) aren't in this table and correctly estimate to $0.
_RATES_PER_1K_TOKENS: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-4o-mini"): (0.00015, 0.0006),
    ("anthropic", "claude-3-5-haiku"): (0.0008, 0.004),
    ("anthropic", "claude-opus-5"): (0.005, 0.025),
    ("anthropic", "claude-sonnet-5"): (0.002, 0.010),
    ("anthropic", "claude-haiku-4-5"): (0.001, 0.005),
}


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _RATES_PER_1K_TOKENS.get((provider, model))
    if rates is None:
        return 0.0

    input_rate, output_rate = rates
    return (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate
