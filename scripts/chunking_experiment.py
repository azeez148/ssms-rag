"""Compare chunk_size/chunk_overlap configurations on a real long document —
and, since Plan Set 22, compare the manual chunker against its LangChain
replacement on the same input.

Usage:
    .venv/bin/python scripts/chunking_experiment.py
"""

from pathlib import Path

from app.ingestion.chunking import LangChainRecursiveChunker, RecursiveTokenChunker, count_tokens
from app.ingestion.config import ChunkingConfig

TEXT_PATH = Path(__file__).resolve().parent.parent / "prompt.md"
CONFIGS = [(300, 30), (500, 50), (800, 100)]


def _stats(label: str, chunks: list) -> None:
    token_counts = [c.token_count for c in chunks]
    print(f"  {label}: {len(chunks)} chunks, tokens min/avg/max = "
          f"{min(token_counts)}/{sum(token_counts) // len(token_counts)}/{max(token_counts)}")


def main() -> None:
    text = TEXT_PATH.read_text()
    print(f"Source document: {TEXT_PATH.name} ({count_tokens(text)} tokens)\n")

    for chunk_size, chunk_overlap in CONFIGS:
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        manual_chunks = RecursiveTokenChunker(config).chunk(document_id="prompt-md", content=text)
        langchain_chunks = LangChainRecursiveChunker(config).chunk(document_id="prompt-md", content=text)

        manual_over = sum(1 for c in manual_chunks if c.token_count > chunk_size)
        langchain_over = sum(1 for c in langchain_chunks if c.token_count > chunk_size)

        print(f"chunk_size={chunk_size} chunk_overlap={chunk_overlap}")
        _stats("manual   ", manual_chunks)
        print(f"             chunks over chunk_size limit: {manual_over}")
        _stats("langchain", langchain_chunks)
        print(f"             chunks over chunk_size limit: {langchain_over}")
        print(f"  manual    first chunk preview: {manual_chunks[0].content[:80]!r}")
        print(f"  langchain first chunk preview: {langchain_chunks[0].content[:80]!r}")
        print()


if __name__ == "__main__":
    main()
