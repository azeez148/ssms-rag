import asyncio

from app.reranking.base import Reranker
from app.retrieval.base import Retriever
from app.retrieval.reranking_retriever import RerankingRetriever
from app.schemas.retrieval import RetrievedChunk


def _chunk(document_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content for {document_id}",
        document_id=document_id,
        chunk_index=0,
        document_title=document_id,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=0.5,
        distance=0.5,
    )


class FakeRetriever(Retriever):
    def __init__(self, canned: list[RetrievedChunk]):
        self._canned = canned
        self.last_top_k = None
        self.last_filters = None
        self.last_knowledge_base_id = None

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        self.last_filters = filters
        self.last_knowledge_base_id = knowledge_base_id
        return self._canned[:top_k]


class FakeReranker(Reranker):
    def __init__(self):
        self.last_top_k = None
        self.num_chunks_received = None

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        self.num_chunks_received = len(chunks)
        return list(reversed(chunks))[:top_k]


def test_fetches_candidate_multiplier_times_top_k_from_base_retriever():
    base_retriever = FakeRetriever([_chunk(f"c{i}") for i in range(25)])
    reranker = FakeReranker()
    retriever = RerankingRetriever(base_retriever, reranker, candidate_multiplier=4)

    asyncio.run(retriever.retrieve("query", "test-kb", top_k=5))

    assert base_retriever.last_top_k == 20
    assert reranker.num_chunks_received == 20
    assert reranker.last_top_k == 5


def test_forwards_filters_to_base_retriever():
    base_retriever = FakeRetriever([])
    retriever = RerankingRetriever(base_retriever, FakeReranker())

    asyncio.run(retriever.retrieve("query", "test-kb", top_k=5, filters={"category": "Jersey"}))

    assert base_retriever.last_filters == {"category": "Jersey"}


def test_forwards_knowledge_base_id_to_base_retriever():
    base_retriever = FakeRetriever([])
    retriever = RerankingRetriever(base_retriever, FakeReranker())

    asyncio.run(retriever.retrieve("query", "sports-store", top_k=5))

    assert base_retriever.last_knowledge_base_id == "sports-store"


def test_returns_reranked_order_not_base_retriever_order():
    base_retriever = FakeRetriever([_chunk("a"), _chunk("b"), _chunk("c")])
    retriever = RerankingRetriever(base_retriever, FakeReranker())

    results = asyncio.run(retriever.retrieve("query", "test-kb", top_k=3))

    assert [r.document_id for r in results] == ["c", "b", "a"]
