"""Read-only checks of the live store queries against the real ssms_db.
Skips if it isn't reachable. Nothing here writes."""

import os
from datetime import date

import pytest
from sqlalchemy import create_engine, text

from app.sources.config import load_postgres_source_config
from plugins.sports_store.repository import StoreRepository, search_tokens

DEFAULT_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/ssms_db"
URL = os.environ.get("SPORTS_STORE_DATABASE_URL", DEFAULT_URL)


def _database_available() -> bool:
    try:
        with create_engine(URL).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_search_tokens_drop_plurals_but_keep_codes():
    assert search_tokens("Argentina jerseys 25-26 boots glass") == ["argentina", "jersey", "25-26", "boot", "glass"]


def test_sports_store_config_declares_the_actions_handler():
    config = load_postgres_source_config("config/sports_store_source.yaml")
    assert config.actions.handler == "plugins.sports_store.actions:SportsStoreActions"


live = pytest.mark.skipif(not _database_available(), reason="ssms_db not reachable")


@pytest.fixture
def repository():
    return StoreRepository(create_engine(URL))


@live
def test_products_are_found_by_name_and_by_id(repository):
    products, exact = repository.find_products("argentina")
    assert exact and products
    assert all("argentina" in (p.name + p.category).lower() for p in products)

    by_id, _ = repository.find_products(f"#{products[0].id}")
    assert [p.id for p in by_id] == [products[0].id]


@live
def test_sizes_are_live_quantities(repository):
    products, _ = repository.find_products("jersey")
    sized = [p for p in products if p.sizes]
    assert sized
    assert all(isinstance(q, int) for p in sized for q in p.sizes.values())


@live
def test_lookups_and_summary_run(repository):
    assert repository.payment_types()
    assert repository.delivery_types()
    summary = repository.sales_summary(date(2020, 1, 1), date.today(), None)
    assert summary["sales"] >= 0 and isinstance(summary["top"], list)


@live
def test_last_sales_are_newest_first_with_items(repository):
    sales = repository.last_sales(3, None)
    assert sales
    assert [s["id"] for s in sales] == sorted((s["id"] for s in sales), reverse=True)
    assert all(isinstance(s["items"], list) for s in sales)
