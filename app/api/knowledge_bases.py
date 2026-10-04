import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import func
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.generation.factory import resolve_llm_identity
from app.models import Document
from app.schemas.knowledge_base import KnowledgeBaseInfo, KnowledgeBasesResponse
from app.sources.config import discover_source_configs

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.get("/knowledge-bases", response_model=KnowledgeBasesResponse)
def list_knowledge_bases() -> KnowledgeBasesResponse:
    """The knowledge bases this deployment has source configs for, how many
    documents of each are currently indexed, and which engine answers questions.
    The chat UI uses this to know what it can talk to and whether it needs to
    ingest first."""
    settings = get_settings()
    config_dir = Path(settings.source_config_dir)

    session = SessionLocal()
    try:
        counts = dict(session.query(Document.knowledge_base_id, func.count(Document.id)).group_by(Document.knowledge_base_id))
    except OperationalError as exc:
        logger.warning("knowledge base listing failed: %s", exc)
        raise HTTPException(status_code=503, detail="Knowledge database unavailable") from exc
    finally:
        session.close()

    knowledge_bases = [
        KnowledgeBaseInfo(
            id=config.knowledge_base_id,
            name=config.title or config.knowledge_base_id,
            description=config.description,
            suggested_questions=config.suggested_questions,
            config_path=path.relative_to(config_dir.parent).as_posix(),
            documents_indexed=counts.get(config.knowledge_base_id, 0),
        )
        for path, config in discover_source_configs(config_dir)
    ]

    # Largest first: the fullest knowledge base is the sensible default in the UI.
    knowledge_bases.sort(key=lambda kb: (-kb.documents_indexed, kb.name.lower()))

    provider, model = resolve_llm_identity(settings)
    return KnowledgeBasesResponse(answer_engine=provider, answer_model=model, knowledge_bases=knowledge_bases)
