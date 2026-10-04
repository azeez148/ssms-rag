import asyncio

from sentence_transformers import CrossEncoder

from app.reranking.base import Reranker
from app.schemas.retrieval import RetrievedChunk


class CrossEncoderReranker(Reranker):
    """Scores each (query, chunk) pair jointly through a cross-encoder model —
    slower per-pair than the bi-encoder used for initial retrieval, but far
    more accurate, since the model reads the query and document together
    instead of comparing two independently-computed vectors. Only practical
    over a small candidate set, never the whole corpus.

    Free and local (no API key), via the same sentence-transformers package
    already used for embeddings (Plan Set 6).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._rerank_sync, query, chunks, top_k)

    def _rerank_sync(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True)

        # The cross-encoder's score replaces the retriever's similarity/distance
        # entirely — they're on incomparable scales (cosine distance vs. a
        # cross-encoder logit), so keeping the stale retrieval score around
        # after reordering would be misleading.
        return [
            chunk.model_copy(update={"similarity": float(score), "distance": -float(score)})
            for score, chunk in ranked[:top_k]
        ]
