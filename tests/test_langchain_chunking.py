from app.ingestion.chunking import LangChainRecursiveChunker, count_tokens
from app.ingestion.config import ChunkingConfig


def test_short_text_is_a_single_chunk():
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    chunks = chunker.chunk(document_id="doc-1", content="A short sentence.")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "A short sentence."
    assert chunks[0].token_count == count_tokens("A short sentence.")


def test_long_text_splits_into_multiple_chunks_within_budget():
    paragraph = "The quick brown fox jumps over the lazy dog. " * 200
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.chunk(document_id="doc-2", content=paragraph)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_every_chunk_strictly_respects_chunk_size():
    """Plan Set 22 finding: RecursiveTokenChunker's hand-rolled `_merge` lets
    a chunk exceed `chunk_size` (confirmed on real content in
    scripts/chunking_experiment.py — 7 of 12 chunks over budget at
    chunk_size=500). LangChain's splitter never does, on the same input."""
    text = "The quick brown fox jumps over the lazy dog. " * 400
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.chunk(document_id="doc-budget", content=text)

    assert all(c.token_count <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    paragraph = ". ".join(f"Sentence number {i}" for i in range(100))
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=50, chunk_overlap=15))

    chunks = chunker.chunk(document_id="doc-3", content=paragraph)

    assert len(chunks) > 1
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    overlap = set(first_words[-5:]) & set(second_words[:10])
    assert overlap, "expected trailing words of chunk 0 to reappear near the start of chunk 1"


def test_metadata_is_propagated_to_every_chunk():
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    chunks = chunker.chunk(document_id="doc-4", content="Hello world.", metadata={"category": "Jersey"})

    assert all(c.metadata == {"category": "Jersey"} for c in chunks)


def test_empty_content_produces_no_chunks():
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    assert chunker.chunk(document_id="doc-5", content="   ") == []
