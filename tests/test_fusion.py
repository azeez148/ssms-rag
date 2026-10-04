from app.retrieval.fusion import reciprocal_rank_fusion
from app.schemas.retrieval import RetrievedChunk


def _chunk(document_id: str, chunk_index: int = 0, similarity: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content for {document_id}",
        document_id=document_id,
        chunk_index=chunk_index,
        document_title=document_id,
        source="test_suite",
        knowledge_base_id="test-kb",
        metadata={},
        similarity=similarity,
        distance=1.0 - similarity,
    )


def test_chunk_appearing_in_both_lists_outranks_one_appearing_in_only_one():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")

    # "a" is rank 1 in list 1 only. "b" is rank 2 in both lists.
    list1 = [a, b]
    list2 = [c, b]

    fused = reciprocal_rank_fusion([list1, list2])

    assert fused[0].document_id == "b"


def test_single_list_preserves_original_order():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")

    fused = reciprocal_rank_fusion([[a, b, c]])

    assert [chunk.document_id for chunk in fused] == ["a", "b", "c"]


def test_deduplicates_by_document_id_and_chunk_index():
    a = _chunk("a", chunk_index=0)
    a_again = _chunk("a", chunk_index=0)

    fused = reciprocal_rank_fusion([[a], [a_again]])

    assert len(fused) == 1


def test_same_document_different_chunk_index_not_deduplicated():
    a0 = _chunk("a", chunk_index=0)
    a1 = _chunk("a", chunk_index=1)

    fused = reciprocal_rank_fusion([[a0], [a1]])

    assert len(fused) == 2


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []
