import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.ingestion.chunking import LangChainRecursiveChunker
from app.ingestion.config import ChunkingConfig
from app.schemas.ingestion import IngestRequest, IngestResponse
from app.services.ingestion_service import IngestionService
from app.sources.config import load_postgres_source_config, resolve_source_config_path
from app.sources.postgres_source import PostgresKnowledgeSource

router = APIRouter(prefix="/admin")
logger = logging.getLogger(__name__)

# One ingestion at a time per source config. Several chat sessions opening at
# once each trigger a sync; the later ones wait for the first and then find
# everything already up to date (unchanged documents are skipped), instead of
# all embedding the same rows concurrently.
_ingest_locks: dict[str, asyncio.Lock] = {}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    settings = get_settings()

    # One generic error for every failure mode (outside the config dir, not a
    # yaml file, missing, malformed): the caller learns nothing about what does
    # or doesn't exist on the server. The real reason goes to the server log.
    try:
        config_path = resolve_source_config_path(request.config_path, Path(settings.source_config_dir))
        source_config = load_postgres_source_config(str(config_path))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        logger.warning("rejected ingest config %r: %s", request.config_path, exc)
        raise HTTPException(status_code=400, detail="Invalid source config") from exc

    source = PostgresKnowledgeSource(source_config)
    chunker = LangChainRecursiveChunker(ChunkingConfig(chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap))
    embedding_provider = get_embedding_provider(settings)

    service = IngestionService(
        source=source,
        source_name=source_config.name,
        chunker=chunker,
        embedding_provider=embedding_provider,
    )

    lock = _ingest_locks.setdefault(str(config_path), asyncio.Lock())
    async with lock:
        session = SessionLocal()
        try:
            run, result = await service.run(session)
        except OperationalError as exc:
            # Either the source database or the knowledge database is unreachable.
            logger.warning("ingestion could not reach a database: %s", exc)
            raise HTTPException(status_code=503, detail="A database is unavailable") from exc
        finally:
            session.close()

    return IngestResponse(
        run_id=run.id,
        status=run.status,
        documents_processed=result.documents_processed,
        documents_created=result.documents_created,
        documents_updated=result.documents_updated,
        documents_unchanged=result.documents_unchanged,
        chunks_created=result.chunks_created,
        error=run.error,
    )
