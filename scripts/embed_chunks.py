"""Ingest a handful of Sports Store documents, chunk them, embed the chunks in
batches, and store everything in pgvector. Manual/simple version — Plan Set 7
adds idempotency (content hashing, skip-unchanged, ingestion run tracking).

Usage:
    .venv/bin/python scripts/embed_chunks.py --limit 20
"""

import argparse
import asyncio
import hashlib

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.ingestion.chunking import LangChainRecursiveChunker
from app.ingestion.config import ChunkingConfig
from app.models import Chunk, Document
from app.sources.config import load_postgres_source_config
from app.sources.postgres_source import PostgresKnowledgeSource

CONFIG_PATH = "config/sports_store_source.yaml"


async def main(limit: int) -> None:
    settings = get_settings()

    source_config = load_postgres_source_config(CONFIG_PATH)
    source = PostgresKnowledgeSource(source_config)
    documents = (await source.load_documents())[:limit]
    print(f"Loaded {len(documents)} documents from '{source_config.name}'")

    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    all_chunks = []
    for document in documents:
        chunk_data = chunker.chunk(document.id, document.content, document.metadata)
        all_chunks.append((document, chunk_data))
    total_chunks = sum(len(c) for _, c in all_chunks)
    print(f"Produced {total_chunks} chunks")

    provider = get_embedding_provider(settings)
    contents = [chunk.content for _, chunks in all_chunks for chunk in chunks]
    embeddings = await provider.embed_batch(contents)
    print(f"Embedded {len(embeddings)} chunks (dimensions={provider.dimensions})")

    session = SessionLocal()
    embedding_iter = iter(embeddings)
    try:
        external_ids = [d.id for d in documents]
        session.query(Document).filter(Document.external_id.in_(external_ids)).delete(synchronize_session=False)
        session.commit()

        for document, chunks in all_chunks:
            db_document = Document(
                external_id=document.id,
                knowledge_base_id=document.metadata["knowledge_base_id"],
                source=document.source,
                source_type=document.source_type,
                title=document.title,
                content_hash=hashlib.sha256(document.content.encode()).hexdigest(),
                metadata_=document.metadata,
            )
            session.add(db_document)
            session.flush()

            for chunk_data in chunks:
                session.add(
                    Chunk(
                        document_id=db_document.id,
                        content=chunk_data.content,
                        chunk_index=chunk_data.chunk_index,
                        token_count=chunk_data.token_count,
                        embedding=next(embedding_iter),
                        metadata_=chunk_data.metadata,
                    )
                )
        session.commit()
        print(f"Stored {len(documents)} documents and {total_chunks} chunks in pgvector.")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(main(args.limit))
