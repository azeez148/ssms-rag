from functools import lru_cache

from app.core.config import Settings
from app.core.model_loading import MODEL_LOAD_LOCK
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformers_provider import SentenceTransformersEmbeddingProvider


@lru_cache(maxsize=4)
def _sentence_transformers(model_name: str) -> SentenceTransformersEmbeddingProvider:
    # Loading the model takes seconds; this factory runs on every request, so
    # the loaded model is kept and shared instead of being rebuilt each time.
    return SentenceTransformersEmbeddingProvider(model_name)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformers":
        with MODEL_LOAD_LOCK:
            return _sentence_transformers(settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
