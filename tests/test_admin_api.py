import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.database import SessionLocal, engine
from app.main import app
from app.models import Document

client = TestClient(app)


def _database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="RAG postgres not running")


@pytest.fixture
def dummy_items_table(monkeypatch):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS items"))
        conn.execute(
            text(
                """
                CREATE TABLE items (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description VARCHAR,
                    category VARCHAR
                )
                """
            )
        )
        conn.execute(
            text("INSERT INTO items (name, description, category) VALUES ('Gadget', 'A useful gadget', 'Tools')")
        )

    monkeypatch.setenv("DUMMY_SOURCE_DATABASE_URL", engine.url.render_as_string(hide_password=False))

    yield

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS items"))

    session = SessionLocal()
    try:
        session.query(Document).filter(Document.external_id.like("items-%")).delete(synchronize_session=False)
        session.commit()
    finally:
        session.close()


def test_ingest_endpoint_creates_documents_and_chunks(dummy_items_table):
    response = client.post("/admin/ingest", json={"config_path": "config/dummy_source.example.yaml"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["documents_created"] == 1
    assert body["chunks_created"] == 1
    assert body["error"] is None


def test_ingest_endpoint_is_idempotent_on_second_call(dummy_items_table):
    client.post("/admin/ingest", json={"config_path": "config/dummy_source.example.yaml"})
    response = client.post("/admin/ingest", json={"config_path": "config/dummy_source.example.yaml"})

    body = response.json()
    assert body["documents_unchanged"] == 1
    assert body["documents_created"] == 0
    assert body["chunks_created"] == 0


def test_ingest_endpoint_rejects_missing_config():
    response = client.post("/admin/ingest", json={"config_path": "config/does_not_exist.yaml"})
    assert response.status_code == 400
