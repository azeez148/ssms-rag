import asyncio
import math

import pytest

from app.embeddings.factory import get_embedding_provider
from app.core.config import Settings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def provider():
    settings = Settings()
    return get_embedding_provider(settings)


def test_embed_text_returns_correct_dimensions(provider):
    vector = asyncio.run(provider.embed_text("Argentina Home Jersey"))
    assert len(vector) == provider.dimensions == 384


def test_embed_batch_returns_one_vector_per_input(provider):
    texts = ["Argentina Home Jersey", "Portugal Away Jersey", "Running Shoes"]
    vectors = asyncio.run(provider.embed_batch(texts))
    assert len(vectors) == len(texts)
    assert all(len(v) == provider.dimensions for v in vectors)


def test_same_text_produces_identical_embedding(provider):
    first = asyncio.run(provider.embed_text("Argentina Home Jersey"))
    second = asyncio.run(provider.embed_text("Argentina Home Jersey"))
    assert first == second


def test_semantically_related_text_is_more_similar_than_unrelated(provider):
    vectors = asyncio.run(
        provider.embed_batch(
            [
                "Argentina Home Jersey 2026, football kit for fans",
                "Argentina Away Jersey 2026, official football shirt",
                "Stainless steel kitchen blender with 5 speed settings",
            ]
        )
    )
    jersey_a, jersey_b, blender = vectors

    related_similarity = _cosine_similarity(jersey_a, jersey_b)
    unrelated_similarity = _cosine_similarity(jersey_a, blender)

    assert related_similarity > unrelated_similarity


def test_unsupported_provider_raises_clear_error():
    settings = Settings(embedding_provider="does_not_exist")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_provider(settings)


def test_factory_reuses_the_loaded_model_instead_of_reloading_per_request():
    from app.core.config import Settings
    from app.embeddings.factory import get_embedding_provider

    settings = Settings(embedding_provider="sentence_transformers")

    assert get_embedding_provider(settings) is get_embedding_provider(settings)
