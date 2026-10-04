from app.schemas.retrieval import RetrievedChunk


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Combines multiple ranked result lists into one ranking using Reciprocal
    Rank Fusion: score(chunk) = sum(1 / (k + rank)) across every list it
    appears in (rank is 1-indexed; missing from a list contributes nothing).

    Uses each result's *rank position*, not its raw score, specifically
    because different retrievers produce scores on incomparable scales
    (bounded cosine similarity vs. unbounded ts_rank) — rank position is the
    only thing that's directly comparable across retrieval methods.

    `k` dampens the influence of rank differences further down the list
    (a common default is 60); a chunk appearing in multiple lists, even at
    modest ranks, outranks one appearing in only a single list at rank 1.
    """
    scores: dict[tuple[str, int], float] = {}
    chunks_by_key: dict[tuple[str, int], RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = (chunk.document_id, chunk.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks_by_key.setdefault(key, chunk)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunks_by_key[key] for key in ordered_keys]
