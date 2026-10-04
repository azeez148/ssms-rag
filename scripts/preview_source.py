"""Load documents from any configured source and print them. Verifies document
extraction only — no chunking, embeddings, or LLM calls happen here.

Usage:
    .venv/bin/python scripts/preview_source.py config/sports_store_source.yaml [--limit N]
"""

import argparse
import asyncio
import json

from app.sources.config import load_postgres_source_config
from app.sources.postgres_source import PostgresKnowledgeSource


async def main(config_path: str, limit: int) -> None:
    config = load_postgres_source_config(config_path)
    source = PostgresKnowledgeSource(config)
    documents = await source.load_documents()

    print(f"Loaded {len(documents)} documents from '{config.name}'\n")
    for document in documents[:limit]:
        print(json.dumps(document.model_dump(), indent=2, default=str))
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main(args.config_path, args.limit))
