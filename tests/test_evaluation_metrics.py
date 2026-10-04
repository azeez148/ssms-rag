from app.evaluation.metrics import reciprocal_rank, recall_at_k


def test_recall_at_k_hit_within_k():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_at_k_no_match_at_all():
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_recall_at_k_matches_any_of_multiple_expected_ids():
    assert recall_at_k(["a", "b"], {"x", "b"}, k=2) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
