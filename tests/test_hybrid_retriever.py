import asyncio

from app.retrieval.base import Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.retrieval import RetrievedChunk


def _chunk(document_id: str, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content for {document_id}",
        document_id=document_id,
        chunk_index=0,
        document_title=document_id,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=similarity,
        distance=1.0 - similarity,
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


def test_hybrid_combines_vector_and_keyword_rankings():
    # "shared" is a strong vector match and a weak keyword match; it should
    # outrank items that only appear in one list.
    vector_results = [_chunk("shared", 0.9), _chunk("vector-only", 0.8)]
    keyword_results = [_chunk("keyword-only", 0.5), _chunk("shared", 0.3)]

    vector_retriever = FakeRetriever(vector_results)
    keyword_retriever = FakeRetriever(keyword_results)
    hybrid = HybridRetriever(vector_retriever, keyword_retriever)

    results = asyncio.run(hybrid.retrieve("query", "test-kb", top_k=3))

    assert results[0].document_id == "shared"
    assert {r.document_id for r in results} == {"shared", "vector-only", "keyword-only"}


def test_hybrid_requests_more_candidates_than_top_k_from_each_side():
    vector_retriever = FakeRetriever([_chunk(f"v{i}", 0.5) for i in range(20)])
    keyword_retriever = FakeRetriever([_chunk(f"k{i}", 0.5) for i in range(20)])
    hybrid = HybridRetriever(vector_retriever, keyword_retriever, candidate_multiplier=4)

    asyncio.run(hybrid.retrieve("query", "test-kb", top_k=5))

    assert vector_retriever.last_top_k == 20
    assert keyword_retriever.last_top_k == 20


def test_hybrid_forwards_filters_to_both_retrievers():
    vector_retriever = FakeRetriever([])
    keyword_retriever = FakeRetriever([])
    hybrid = HybridRetriever(vector_retriever, keyword_retriever)

    asyncio.run(hybrid.retrieve("query", "test-kb", top_k=5, filters={"category": "Jersey"}))

    assert vector_retriever.last_filters == {"category": "Jersey"}
    assert keyword_retriever.last_filters == {"category": "Jersey"}


def test_hybrid_forwards_knowledge_base_id_to_both_retrievers():
    vector_retriever = FakeRetriever([])
    keyword_retriever = FakeRetriever([])
    hybrid = HybridRetriever(vector_retriever, keyword_retriever)

    asyncio.run(hybrid.retrieve("query", "sports-store", top_k=5))

    assert vector_retriever.last_knowledge_base_id == "sports-store"
    assert keyword_retriever.last_knowledge_base_id == "sports-store"


def test_hybrid_result_count_is_capped_at_top_k():
    vector_retriever = FakeRetriever([_chunk(f"v{i}", 0.9 - i * 0.01) for i in range(10)])
    keyword_retriever = FakeRetriever([])
    hybrid = HybridRetriever(vector_retriever, keyword_retriever)

    results = asyncio.run(hybrid.retrieve("query", "test-kb", top_k=3))

    assert len(results) == 3
