import pytest

from app.observability.cost import estimate_cost


def test_unknown_provider_and_model_costs_zero():
    assert estimate_cost("stub", "stub-echo-v1", prompt_tokens=1000, completion_tokens=500) == 0.0


def test_local_free_providers_cost_zero_regardless_of_token_count():
    assert estimate_cost("sentence_transformers", "all-MiniLM-L6-v2", prompt_tokens=1_000_000, completion_tokens=0) == 0.0


def test_known_provider_and_model_computes_nonzero_cost():
    cost = estimate_cost("openai", "gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert cost > 0.0


def test_cost_scales_linearly_with_token_count():
    small = estimate_cost("openai", "gpt-4o-mini", prompt_tokens=100, completion_tokens=100)
    large = estimate_cost("openai", "gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert large == pytest.approx(small * 10)
