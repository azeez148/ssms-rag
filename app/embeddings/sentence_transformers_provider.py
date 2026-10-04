import asyncio

from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Runs embeddings locally via sentence-transformers. No API key, no per-call cost."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
