"""Evaluate retrieval quality: Recall@1/3/5 and MRR.

Runs the evaluation dataset against:
1. The live pgvector index (whatever's currently ingested) — vector, keyword,
   hybrid (RRF), and vector+rerank (top 20 -> cross-encoder -> top 5), compared
   head-to-head.
2. Three in-memory chunk configurations (300/30, 500/50, 800/100), built
   fresh from the source without touching the persisted pgvector data, so
   they can be compared head-to-head.

Usage:
    .venv/bin/python scripts/run_retrieval_eval.py
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.embeddings.factory import get_embedding_provider
from app.evaluation.dataset import load_eval_dataset
from app.evaluation.in_memory_retriever import InMemoryRetriever
from app.evaluation.runner import EvalReport, evaluate_retrieval
from app.ingestion.chunking import LangChainRecursiveChunker
from app.ingestion.cleaning import clean_content
from app.ingestion.config import ChunkingConfig
from app.reranking.factory import get_reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.reranking_retriever import RerankingRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.sources.config import load_postgres_source_config
from app.sources.postgres_source import PostgresKnowledgeSource

DATASET_PATH = "evaluation/datasets/sports_store_questions.json"
SOURCE_CONFIG_PATH = "config/sports_store_source.yaml"
KNOWLEDGE_BASE_ID = "sports-store"
CHUNK_CONFIGS = [(300, 30), (500, 50), (800, 100)]
RESULTS_PATH = Path("evaluation/results/latest.json")


def _print_report(label: str, report: EvalReport) -> None:
    print(f"\n{label}")
    print(f"  questions: {report.num_questions}")
    for k, value in sorted(report.recall_at.items()):
        print(f"  Recall@{k}: {value:.3f}")
    print(f"  MRR: {report.mrr:.3f}")


async def main() -> None:
    settings = get_settings()
    dataset = load_eval_dataset(DATASET_PATH)
    embedding_provider = get_embedding_provider(settings)

    all_reports: dict[str, dict] = {}

    # 1. Live pgvector index (whatever's currently ingested) — vector, keyword, hybrid, reranked.
    vector_retriever = VectorRetriever(embedding_provider)
    keyword_retriever = KeywordRetriever()
    hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)
    reranker = get_reranker(settings)
    reranking_retriever = RerankingRetriever(vector_retriever, reranker, settings.rerank_candidate_multiplier)

    for label, retriever in [
        ("Vector only (live pgvector)", vector_retriever),
        ("Keyword only (Postgres full-text)", keyword_retriever),
        ("Hybrid (RRF of vector + keyword)", hybrid_retriever),
        ("Vector + rerank (top 20 -> cross-encoder -> top 5)", reranking_retriever),
    ]:
        report = await evaluate_retrieval(retriever, KNOWLEDGE_BASE_ID, dataset)
        _print_report(label, report)
        all_reports[label] = asdict(report)

    # 2. In-memory comparison across chunk configurations.
    source_config = load_postgres_source_config(SOURCE_CONFIG_PATH)
    source = PostgresKnowledgeSource(source_config)
    documents = await source.load_documents()

    for chunk_size, chunk_overlap in CHUNK_CONFIGS:
        chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        indexed_chunks = []
        contents = []
        chunk_document_pairs = []
        for document in documents:
            cleaned = clean_content(document.content)
            for chunk_data in chunker.chunk(document.id, cleaned, document.metadata):
                contents.append(chunk_data.content)
                chunk_document_pairs.append((document, chunk_data))

        embeddings = await embedding_provider.embed_batch(contents)
        for (document, chunk_data), embedding in zip(chunk_document_pairs, embeddings):
            indexed_chunks.append((document, chunk_data, embedding))

        retriever = InMemoryRetriever(embedding_provider, indexed_chunks)
        report = await evaluate_retrieval(retriever, KNOWLEDGE_BASE_ID, dataset)

        label = f"In-memory chunk_size={chunk_size} chunk_overlap={chunk_overlap} ({len(indexed_chunks)} chunks)"
        _print_report(label, report)
        all_reports[f"chunk_{chunk_size}_{chunk_overlap}"] = asdict(report)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "reports": all_reports},
            indent=2,
        )
    )
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
