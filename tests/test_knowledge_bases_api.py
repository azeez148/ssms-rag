import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.api import admin
from app.core.database import engine
from app.main import app
from app.services.ingestion_service import IngestionService

client = TestClient(app)


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


needs_database = pytest.mark.skipif(not _database_available(), reason="RAG postgres not running")


@needs_database
def test_lists_the_configured_knowledge_bases_with_display_info():
    response = client.get("/api/knowledge-bases")

    assert response.status_code == 200
    body = response.json()
    by_id = {kb["id"]: kb for kb in body["knowledge_bases"]}

    assert {"sports-store", "library"} <= set(by_id)
    sports = by_id["sports-store"]
    assert sports["name"] == "Sports Store"
    assert sports["config_path"] == "config/sports_store_source.yaml"
    assert sports["suggested_questions"]
    assert isinstance(sports["documents_indexed"], int)


@needs_database
def test_example_configs_are_not_offered_as_knowledge_bases():
    body = client.get("/api/knowledge-bases").json()

    assert all("example" not in kb["config_path"] for kb in body["knowledge_bases"])


@needs_database
def test_reports_which_engine_answers(monkeypatch):
    monkeypatch.setattr("app.api.knowledge_bases.get_settings", lambda: _settings(llm_provider="extractive"))

    body = client.get("/api/knowledge-bases").json()

    assert body["answer_engine"] == "extractive"
    assert body["answer_model"] == "extractive-v1"


def _settings(**overrides):
    from app.core.config import Settings

    return Settings(**overrides)


def test_every_listed_config_path_is_accepted_by_the_ingest_resolver():
    """The UI sends exactly what the listing returned to /admin/ingest."""
    from pathlib import Path

    from app.sources.config import resolve_source_config_path

    config_dir = Path(_settings().source_config_dir)
    for path in ("config/sports_store_source.yaml", "config/library_source.yaml"):
        assert resolve_source_config_path(path, config_dir).is_file()


def test_ingest_returns_503_when_a_database_is_unreachable(monkeypatch):
    monkeypatch.setenv("DUMMY_SOURCE_DATABASE_URL", "postgresql+psycopg://x:x@localhost:1/x")

    async def unreachable(self, session):
        raise OperationalError("connect", {}, ConnectionError("refused"))

    monkeypatch.setattr(IngestionService, "run", unreachable)

    response = client.post("/admin/ingest", json={"config_path": "config/dummy_source.example.yaml"})

    assert response.status_code == 503
    assert response.json() == {"detail": "A database is unavailable"}


def test_concurrent_ingests_of_the_same_config_run_one_at_a_time(monkeypatch):
    monkeypatch.setenv("DUMMY_SOURCE_DATABASE_URL", "postgresql+psycopg://x:x@localhost:1/x")
    admin._ingest_locks.clear()
    active = 0
    peak = 0

    async def slow_run(self, session):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        raise OperationalError("connect", {}, ConnectionError("stop here"))

    monkeypatch.setattr(IngestionService, "run", slow_run)

    async def fire_two():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            body = {"config_path": "config/dummy_source.example.yaml"}
            return await asyncio.gather(
                async_client.post("/admin/ingest", json=body), async_client.post("/admin/ingest", json=body)
            )

    responses = asyncio.run(fire_two())

    assert [r.status_code for r in responses] == [503, 503]
    assert peak == 1
