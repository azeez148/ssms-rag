def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """1.0 if any expected id appears in the top k retrieved ids, else 0.0."""
    return 1.0 if set(retrieved_ids[:k]) & expected_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """1/rank of the first correct hit (1-indexed), 0.0 if none of the expected ids appear."""
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected_ids:
            return 1.0 / rank
    return 0.0
