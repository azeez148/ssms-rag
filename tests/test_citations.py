from app.generation.citations import extract_cited_indices


def test_extracts_single_citation():
    assert extract_cited_indices("Available in size L [2].", num_sources=3) == {2}


def test_extracts_multiple_citations():
    assert extract_cited_indices("See [1] and also [3].", num_sources=3) == {1, 3}


def test_deduplicates_repeated_citations():
    assert extract_cited_indices("[1] confirms this. Also see [1] again.", num_sources=3) == {1}


def test_ignores_out_of_range_citations():
    # Only 2 sources were actually retrieved — a [5] reference is not a valid citation.
    assert extract_cited_indices("Based on [5].", num_sources=2) == set()


def test_no_citations_returns_empty_set():
    assert extract_cited_indices("There is insufficient information to answer.", num_sources=3) == set()


def test_ignores_zero_and_negative_like_patterns():
    assert extract_cited_indices("Model number [0] was discontinued.", num_sources=3) == set()
