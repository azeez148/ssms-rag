from abc import ABC, abstractmethod
from typing import Callable

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.config import ChunkingConfig
from app.schemas.chunk import ChunkData

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document_id: str, content: str, metadata: dict | None = None) -> list[ChunkData]:
        ...


class RecursiveTokenChunker(ChunkingStrategy):
    """Splits text on progressively finer separators (paragraphs, lines,
    sentences, then words) while keeping each chunk under `chunk_size` tokens,
    carrying `chunk_overlap` tokens of trailing context into the next chunk.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, config: ChunkingConfig, token_counter: Callable[[str], int] = count_tokens):
        self._config = config
        self._count_tokens = token_counter

    def chunk(self, document_id: str, content: str, metadata: dict | None = None) -> list[ChunkData]:
        pieces = self._split(content.strip(), self.SEPARATORS)
        return [
            ChunkData(
                document_id=document_id,
                chunk_index=index,
                content=piece,
                token_count=self._count_tokens(piece),
                metadata=metadata or {},
            )
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[-1]
        remaining_separators: list[str] = []
        for i, candidate in enumerate(separators):
            if candidate == "" or candidate in text:
                separator = candidate
                remaining_separators = separators[i + 1 :]
                break

        splits = list(text) if separator == "" else text.split(separator)
        splits = [s for s in splits if s]

        good_splits: list[str] = []
        chunks: list[str] = []

        for split in splits:
            if self._count_tokens(split) < self._config.chunk_size:
                good_splits.append(split)
                continue

            if good_splits:
                chunks.extend(self._merge(good_splits, separator))
                good_splits = []

            if remaining_separators:
                chunks.extend(self._split(split, remaining_separators))
            else:
                chunks.append(split)

        if good_splits:
            chunks.extend(self._merge(good_splits, separator))

        return chunks

    def _merge(self, splits: list[str], separator: str) -> list[str]:
        """Greedily packs small pieces into chunks up to chunk_size tokens,
        sliding a `chunk_overlap`-token window of trailing pieces forward."""
        chunks: list[str] = []
        window: list[str] = []
        window_tokens = 0

        for split in splits:
            split_tokens = self._count_tokens(split)

            if window and window_tokens + split_tokens > self._config.chunk_size:
                chunks.append(separator.join(window))

                while window and (
                    window_tokens > self._config.chunk_overlap
                    or window_tokens + split_tokens > self._config.chunk_size
                ):
                    window_tokens -= self._count_tokens(window[0])
                    window.pop(0)

            window.append(split)
            window_tokens += split_tokens

        if window:
            chunks.append(separator.join(window))

        return chunks


class LangChainRecursiveChunker(ChunkingStrategy):
    """Plan Set 22: same recursive-split-then-greedily-merge algorithm as
    RecursiveTokenChunker above, delegated to LangChain's
    RecursiveCharacterTextSplitter instead of our hand-rolled `_split`/`_merge`.

    Underneath, `from_tiktoken_encoder` does exactly what our `_split` did:
    walk SEPARATORS from coarsest to finest, recursing into any piece still
    too large, then pack the resulting pieces back together up to
    `chunk_size` tokens with `chunk_overlap` tokens of trailing context
    carried into the next chunk — the same two-phase split-then-merge shape,
    just implemented and battle-tested outside this codebase.
    """

    SEPARATORS = RecursiveTokenChunker.SEPARATORS

    def __init__(self, config: ChunkingConfig, token_counter: Callable[[str], int] = count_tokens):
        self._config = config
        self._count_tokens = token_counter
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            separators=self.SEPARATORS,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, document_id: str, content: str, metadata: dict | None = None) -> list[ChunkData]:
        pieces = self._splitter.split_text(content.strip())
        return [
            ChunkData(
                document_id=document_id,
                chunk_index=index,
                content=piece,
                token_count=self._count_tokens(piece),
                metadata=metadata or {},
            )
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]
