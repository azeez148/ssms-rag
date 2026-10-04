import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import ChunkingStrategy
from app.ingestion.cleaning import clean_content
from app.models import Chunk, Document, IngestionRun
from app.sources.base import KnowledgeSource


@dataclass
class IngestionResult:
    documents_processed: int = 0
    documents_created: int = 0
    documents_updated: int = 0
    documents_unchanged: int = 0
    chunks_created: int = 0


class IngestionService:
    """Orchestrates Source -> Documents -> Cleaning -> Chunking -> Embeddings -> pgvector.

    Idempotent: a document whose cleaned-content hash matches what's already
    stored is skipped entirely (no re-chunking, no re-embedding).
    """

    def __init__(
        self,
        source: KnowledgeSource,
        source_name: str,
        chunker: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
    ):
        self._source = source
        self._source_name = source_name
        self._chunker = chunker
        self._embedding_provider = embedding_provider

    async def run(self, session: Session) -> tuple[IngestionRun, IngestionResult]:
        run = IngestionRun(source=self._source_name, status="running")
        session.add(run)
        session.commit()

        result = IngestionResult()
        try:
            documents = await self._source.load_documents()

            for document in documents:
                result.documents_processed += 1
                cleaned_content = clean_content(document.content)
                content_hash = hashlib.sha256(cleaned_content.encode()).hexdigest()

                knowledge_base_id = document.metadata.get("knowledge_base_id")
                if not knowledge_base_id:
                    raise ValueError(
                        f"Document '{document.id}' has no knowledge_base_id in its metadata — every source "
                        "config must set one, so retrieval can never accidentally mix knowledge bases."
                    )

                existing = session.query(Document).filter_by(external_id=document.id).one_or_none()

                if existing is not None and existing.content_hash == content_hash:
                    result.documents_unchanged += 1
                    continue

                if existing is None:
                    db_document = Document(
                        external_id=document.id,
                        knowledge_base_id=knowledge_base_id,
                        source=document.source,
                        source_type=document.source_type,
                        title=document.title,
                        content_hash=content_hash,
                        metadata_=document.metadata,
                    )
                    session.add(db_document)
                    session.flush()
                    result.documents_created += 1
                else:
                    db_document = existing
                    db_document.knowledge_base_id = knowledge_base_id
                    db_document.source = document.source
                    db_document.source_type = document.source_type
                    db_document.title = document.title
                    db_document.content_hash = content_hash
                    db_document.metadata_ = document.metadata
                    session.query(Chunk).filter_by(document_id=db_document.id).delete()
                    result.documents_updated += 1

                chunk_data = self._chunker.chunk(document.id, cleaned_content, document.metadata)
                if chunk_data:
                    embeddings = await self._embedding_provider.embed_batch([c.content for c in chunk_data])
                    for data, embedding in zip(chunk_data, embeddings):
                        session.add(
                            Chunk(
                                document_id=db_document.id,
                                content=data.content,
                                chunk_index=data.chunk_index,
                                token_count=data.token_count,
                                embedding=embedding,
                                metadata_=data.metadata,
                            )
                        )
                    result.chunks_created += len(chunk_data)

                session.commit()

            run.status = "completed"
            run.documents_processed = result.documents_processed
            run.chunks_created = result.chunks_created
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            raise

        return run, result
