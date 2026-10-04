import importlib
import logging
from pathlib import Path

from app.actions.base import ActionHandler
from app.sources.config import PostgresSourceConfig, discover_source_configs

logger = logging.getLogger(__name__)

# Per knowledge base, including None (no actions) so configs aren't re-read per chat message.
_handlers: dict[str, ActionHandler | None] = {}


def _load_handler(config: PostgresSourceConfig) -> ActionHandler:
    module_name, _, class_name = config.actions.handler.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"actions.handler must look like 'package.module:ClassName', got {config.actions.handler!r}")
    handler_class = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(handler_class, type) and issubclass(handler_class, ActionHandler)):
        raise TypeError(f"{config.actions.handler} is not an ActionHandler")
    return handler_class(config, config.actions.options)


def get_action_handler(knowledge_base_id: str, config_dir: Path) -> ActionHandler | None:
    """The action handler declared by this knowledge base's source config, or
    None if it declares none. Source configs are trusted code (they already
    carry SQL), so importing the module they name is no new exposure; one
    that fails to load is logged and the knowledge base falls back to plain RAG."""
    if knowledge_base_id in _handlers:
        return _handlers[knowledge_base_id]

    handler = None
    for _, config in discover_source_configs(config_dir):
        if config.knowledge_base_id == knowledge_base_id and config.actions is not None:
            try:
                handler = _load_handler(config)
            except Exception:
                logger.exception("could not load actions for knowledge base %s", knowledge_base_id)
            break
    _handlers[knowledge_base_id] = handler
    return handler
