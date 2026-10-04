import json
import logging
from dataclasses import asdict

from app.observability.events import ChatObservabilityEvent

logger = logging.getLogger("rag.observability")


def log_chat_event(event: ChatObservabilityEvent) -> None:
    """Emits one structured JSON line per chat request. Parseable by any log
    aggregator without special-casing this app's format."""
    level = logging.ERROR if event.outcome == "error" else logging.INFO
    logger.log(level, json.dumps(asdict(event), default=str))
