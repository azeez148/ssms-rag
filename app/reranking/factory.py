from functools import lru_cache

from app.core.config import Settings
from app.core.model_loading import MODEL_LOAD_LOCK
from app.reranking.base import Reranker
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.reranking.noop_reranker import NoOpReranker


@lru_cache(maxsize=4)
def _cross_encoder(model_name: str) -> CrossEncoderReranker:
    # Same reasoning as the embedding factory: load the model once, reuse it.
    return CrossEncoderReranker(model_name)


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cross_encoder":
        with MODEL_LOAD_LOCK:
            return _cross_encoder(settings.reranker_model)
    if settings.reranker_provider == "none":
        return NoOpReranker()

    raise ValueError(f"Unsupported reranker provider: {settings.reranker_provider!r}")
