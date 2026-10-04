import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ConnectionConfig(BaseModel):
    env: str


class EntityConfig(BaseModel):
    name: str
    query: str
    id_column: str
    title_column: str | None = None
    content_template: str
    metadata_columns: list[str] = Field(default_factory=list)
    static_metadata: dict = Field(default_factory=dict)


class ActionsConfig(BaseModel):
    """Optional chat actions for a knowledge base (e.g. "sell 2 of X"),
    handled by domain code outside the RAG core. `handler` is a
    "package.module:ClassName" import path implementing
    app.actions.base.ActionHandler; `options` is passed to it verbatim."""

    handler: str
    options: dict = Field(default_factory=dict)


class PostgresSourceConfig(BaseModel):
    type: Literal["postgres"]
    name: str
    knowledge_base_id: str
    # Optional, shown by the chat UI (never used by the RAG core).
    title: str | None = None
    description: str | None = None
    suggested_questions: list[str] = Field(default_factory=list)
    connection: ConnectionConfig
    entities: list[EntityConfig]
    actions: ActionsConfig | None = None

    @field_validator("entities")
    @classmethod
    def must_have_at_least_one_entity(cls, value: list[EntityConfig]) -> list[EntityConfig]:
        if not value:
            raise ValueError("at least one entity must be configured")
        return value


def load_postgres_source_config(path: str) -> PostgresSourceConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PostgresSourceConfig.model_validate(raw["source"])


SOURCE_CONFIG_SUFFIXES = {".yaml", ".yml"}


def discover_source_configs(config_dir: Path) -> list[tuple[Path, PostgresSourceConfig]]:
    """Every loadable source config in `config_dir`, skipping `*.example.*`
    files. A malformed file is logged and skipped so it can't hide the valid ones."""
    found = []
    for path in sorted(config_dir.iterdir()) if config_dir.is_dir() else []:
        if path.suffix.lower() not in SOURCE_CONFIG_SUFFIXES or ".example." in path.name:
            continue
        try:
            found.append((path, load_postgres_source_config(str(path))))
        except Exception as exc:
            logger.warning("skipping source config %s: %s", path.name, exc)
    return found


def resolve_source_config_path(requested: str, config_dir: Path) -> Path:
    """Maps a caller-supplied config reference onto a real file *inside*
    `config_dir`, or raises ValueError.

    The caller only chooses among the source configs that ship with the
    deployment; it can't point ingestion at an arbitrary file on the server.
    Source configs contain SQL that gets executed and name the environment
    variable holding a DB connection string, so they're trusted code — which
    is why the set of loadable ones is restricted to one directory.

    `requested` is resolved relative to the project root (so "config/x.yaml"
    works from any working directory) or may be absolute; either way the fully
    resolved path (symlinks and ".." collapsed) must land inside `config_dir`.
    """
    project_root = config_dir.parent
    candidate = (project_root / requested).resolve()
    resolved_dir = config_dir.resolve()

    if resolved_dir not in candidate.parents:
        raise ValueError("config path is outside the source config directory")
    if candidate.suffix.lower() not in SOURCE_CONFIG_SUFFIXES:
        raise ValueError("config path must be a .yaml/.yml file")
    if not candidate.is_file():
        raise ValueError("config file does not exist")
    return candidate
