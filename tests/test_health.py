from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "rag-platform"
    assert body["app_env"] == "development"


def test_readiness_ok_when_database_reachable(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, statement):
            return None

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    monkeypatch.setattr("app.api.health.engine", FakeEngine())

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_readiness_503_when_database_unreachable(monkeypatch):
    class BrokenEngine:
        def connect(self):
            raise ConnectionError("db down")

    monkeypatch.setattr("app.api.health.engine", BrokenEngine())

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unreachable"}
