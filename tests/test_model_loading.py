import asyncio
import threading

from app.core.config import Settings
from app.embeddings import factory as embedding_factory
from app.embeddings.factory import get_embedding_provider
from app.reranking import factory as reranking_factory
from app.reranking.factory import get_reranker
from app.schemas.retrieval import RetrievedChunk


def test_models_loaded_concurrently_are_usable():
    """Regression: the startup warm-up and the first request used to load
    models at the same time, leaving one on the meta device ("Cannot copy out
    of meta tensor") and cached that way."""
    settings = Settings(embedding_provider="sentence_transformers", reranker_provider="cross_encoder")
    embedding_factory._sentence_transformers.cache_clear()
    reranking_factory._cross_encoder.cache_clear()

    loaded, errors = {}, []

    def load(name, fn):
        try:
            loaded[name] = fn(settings)
        except Exception as exc:  # pragma: no cover - the failure being guarded against
            errors.append(exc)

    threads = [
        threading.Thread(target=load, args=("embedding", get_embedding_provider)),
        threading.Thread(target=load, args=("reranker", get_reranker)),
        threading.Thread(target=load, args=("reranker2", get_reranker)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert loaded["reranker"] is loaded["reranker2"]
    assert len(asyncio.run(loaded["embedding"].embed_text("jersey"))) == 384
    chunk = RetrievedChunk(
        content="Argentina jersey", document_id="d", chunk_index=0, document_title="t", source="s",
        knowledge_base_id="kb", similarity=0.5, distance=0.5, metadata={},
    )
    assert len(asyncio.run(loaded["reranker"].rerank("argentina", [chunk], 1))) == 1
