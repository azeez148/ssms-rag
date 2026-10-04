"""Doesn't need a database: /admin/ingest rejects a bad config reference
before anything touches the source or RAG database."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sources.config import resolve_source_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

client = TestClient(app)


def test_accepts_a_config_inside_the_config_dir():
    resolved = resolve_source_config_path("config/sports_store_source.yaml", CONFIG_DIR)

    assert resolved == (CONFIG_DIR / "sports_store_source.yaml").resolve()


def test_accepts_an_absolute_path_inside_the_config_dir():
    absolute = str(CONFIG_DIR / "library_source.yaml")

    assert resolve_source_config_path(absolute, CONFIG_DIR) == (CONFIG_DIR / "library_source.yaml").resolve()


@pytest.mark.parametrize(
    "requested",
    [
        "../../etc/passwd",
        "config/../../etc/passwd",
        "config/../requirements.txt",
        "/etc/passwd",
        "alembic.ini",
        "config/../.env",
    ],
)
def test_rejects_paths_outside_the_config_dir(requested):
    with pytest.raises(ValueError):
        resolve_source_config_path(requested, CONFIG_DIR)


def test_rejects_non_yaml_files_even_inside_the_config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "secrets.txt").write_text("nope")

    with pytest.raises(ValueError):
        resolve_source_config_path("config/secrets.txt", config_dir)


def test_rejects_a_missing_file():
    with pytest.raises(ValueError):
        resolve_source_config_path("config/does_not_exist.yaml", CONFIG_DIR)


def test_rejects_a_symlink_that_escapes_the_config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("source: {}")
    (config_dir / "link.yaml").symlink_to(outside)

    with pytest.raises(ValueError):
        resolve_source_config_path("config/link.yaml", config_dir)


@pytest.mark.parametrize(
    "requested",
    ["../../etc/passwd", "/etc/passwd", "config/does_not_exist.yaml", "requirements.txt"],
)
def test_ingest_endpoint_returns_a_generic_400_and_leaks_nothing(requested):
    response = client.post("/admin/ingest", json={"config_path": requested})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid source config"}
