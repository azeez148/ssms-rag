import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.actions.base import ActionFailedError
from app.actions.pending import pending_actions
from app.actions.registry import get_action_handler
from app.core.config import get_settings
from app.schemas.chat import ActionOutcome

router = APIRouter(prefix="/api/actions")
logger = logging.getLogger(__name__)

GONE = "That request has expired or was already handled. Please ask again."


@router.post("/{action_id}/confirm", response_model=ActionOutcome)
async def confirm_action(action_id: str) -> ActionOutcome:
    """Runs a proposal the chat made (e.g. records a sale). Single use: a
    second confirm of the same id is a 404, so a double click can't sell twice."""
    pending = pending_actions.pop(action_id)
    if pending is None:
        raise HTTPException(status_code=404, detail=GONE)

    handler = get_action_handler(pending.knowledge_base_id, Path(get_settings().source_config_dir))
    if handler is None:
        raise HTTPException(status_code=404, detail=GONE)

    try:
        message = await handler.execute(pending.proposal.payload)
    except ActionFailedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("confirmed action %s failed", action_id)
        raise HTTPException(status_code=502, detail="The store couldn't complete that. Nothing was recorded.") from exc

    logger.info("confirmed action %s for knowledge base %s", action_id, pending.knowledge_base_id)
    return ActionOutcome(status="done", message=message)


@router.post("/{action_id}/cancel", response_model=ActionOutcome)
async def cancel_action(action_id: str) -> ActionOutcome:
    pending_actions.pop(action_id)
    return ActionOutcome(status="cancelled", message="Cancelled. Nothing was recorded.")
