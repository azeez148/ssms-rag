from typing import Any

from sqlalchemy.orm import Query

from app.models import Chunk


def apply_metadata_filters(db_query: Query, filters: dict[str, Any] | None) -> Query:
    """Generic key/value equality filtering on chunk metadata (JSON). Shared by
    every retriever that queries `chunks` directly, so vector and keyword
    search apply filters identically."""
    if not filters:
        return db_query
    for key, value in filters.items():
        db_query = db_query.filter(Chunk.metadata_[key].as_string() == str(value))
    return db_query
