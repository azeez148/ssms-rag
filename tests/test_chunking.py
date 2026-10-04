import pytest

from app.ingestion.chunking import RecursiveTokenChunker, count_tokens
from app.ingestion.config import ChunkingConfig


def test_short_text_is_a_single_chunk():
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    chunks = chunker.chunk(document_id="doc-1", content="A short sentence.")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "A short sentence."
    assert chunks[0].token_count == count_tokens("A short sentence.")


def test_long_text_splits_into_multiple_chunks_within_budget():
    paragraph = "The quick brown fox jumps over the lazy dog. " * 200
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.chunk(document_id="doc-2", content=paragraph)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 100 + 20  # small slack for merge boundary rounding

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    paragraph = ". ".join(f"Sentence number {i}" for i in range(100))
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=50, chunk_overlap=15))

    chunks = chunker.chunk(document_id="doc-3", content=paragraph)

    assert len(chunks) > 1
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    overlap = set(first_words[-5:]) & set(second_words[:10])
    assert overlap, "expected trailing words of chunk 0 to reappear near the start of chunk 1"


def test_metadata_is_propagated_to_every_chunk():
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    chunks = chunker.chunk(document_id="doc-4", content="Hello world.", metadata={"category": "Jersey"})

    assert all(c.metadata == {"category": "Jersey"} for c in chunks)


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_empty_content_produces_no_chunks():
    chunker = RecursiveTokenChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    assert chunker.chunk(document_id="doc-5", content="   ") == []
