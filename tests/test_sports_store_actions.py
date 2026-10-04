import asyncio
import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest

from app.actions.base import ActionFailedError
from plugins.sports_store.actions import SportsStoreActions
from plugins.sports_store.repository import Product


def _jersey(**sizes) -> Product:
    return Product(41, "FC ARGENTINA 25-26", "First Copy Jersey", 700, 650, sizes or {"S": 0, "M": 3, "L": 1})


class FakeRepository:
    def __init__(self, products: list[Product], exact: bool = True):
        self.products = products
        self.exact = exact
        self.summary = {"sales": 0, "revenue": 0.0, "items": 0, "top": []}
        self.summary_args = None

    def find_products(self, query):
        return list(self.products), self.exact

    def get_product(self, product_id, include_inactive=False):
        product = next((p for p in self.products if p.id == product_id), None)
        # A copy, as each database read is: executing must not see the proposal's objects.
        return Product(**{**product.__dict__, "sizes": dict(product.sizes)}) if product else None

    def product_fields(self, product_id):
        product = self.get_product(product_id)
        return product and {
            "id": product.id, "name": product.name, "description": None, "unit_price": 400,
            "selling_price": product.selling_price, "discounted_price": product.sale_price,
            "category_id": 5, "category": product.category, "is_active": True, "can_listed": True,
        }

    def categories(self):
        return [(1, "Jersey"), (4, "Kids Jersey"), (5, "First Copy Jersey")]

    def list_products(self, query, stock, limit):
        self.list_args = (query, stock, limit)
        return list(self.products)[:limit], len(self.products)

    def get_sale(self, sale_id):
        return next((sale for sale in getattr(self, "sales", []) if sale["id"] == sale_id), None)

    def last_sales(self, count, shop_id):
        return getattr(self, "sales", [])[:count]

    def payment_types(self):
        return [(1, "Cash on Delivery"), (2, "Card"), (3, "UPI")]

    def delivery_types(self):
        return [(1, "Pickup from Store", 0), (3, "PO-Express Delivery", 65)]

    def sales_summary(self, start, end, shop_id):
        self.summary_args = (start, end, shop_id)
        return self.summary


def _actions(repository, handler=None, options=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler)) if handler else None
    config = SimpleNamespace(connection=SimpleNamespace(env="UNUSED"))
    return SportsStoreActions(
        config, options or {}, repository=repository, http_client=client, today=lambda: date(2026, 9, 26)
    )


def _ask(actions, question):
    return asyncio.run(actions.handle(question, []))


# --- proposals --------------------------------------------------------------


def test_sale_is_proposed_with_pos_pricing_and_nothing_is_written():
    calls = []
    actions = _actions(FakeRepository([_jersey()]), handler=lambda r: calls.append(r) or httpx.Response(200))

    reply = _ask(actions, "sell 2 FC ARGENTINA 25-26 size m upi express delivery")

    payload = reply.proposal.payload["sale"]
    item = payload["sale_items"][0]
    assert calls == []
    assert (item["product_id"], item["size"], item["quantity"], item["sale_price"]) == (41, "M", 2, 650)
    assert item["total_price"] == 1300
    assert payload["sub_total"] == 1400  # MRP x qty, as the POS records it
    assert payload["total_price"] == 1365  # discounted price x qty + delivery charge
    assert (payload["payment_type_id"], payload["delivery_type_id"], payload["shop_id"]) == (3, 3, 1)
    assert payload["date"] == "2026-09-26"
    assert payload["customer_id"] == 0
    assert payload["customer_address"] == {"name": "In-store Customer", "mobile": "0000000000", "address": "In-store"}
    assert "Total: ₹1,365" in reply.proposal.lines
    assert "Stock after sale: 1 left in size M" in reply.proposal.lines


def test_named_customer_is_sent_for_the_store_to_find_or_create():
    reply = _ask(_actions(FakeRepository([_jersey()])), "sell 1 #41 size L to rahul 9876543210")

    assert reply.proposal.payload["sale"]["customer_id"] == 0
    assert reply.proposal.payload["sale"]["customer_address"] == {"name": "Rahul", "mobile": "9876543210"}


def test_name_without_mobile_asks_for_both():
    reply = _ask(_actions(FakeRepository([_jersey()])), "sell 1 #41 size L to rahul")
    assert reply.proposal is None
    assert "mobile" in reply.answer


def test_missing_size_asks_which_one_in_stock():
    reply = _ask(_actions(FakeRepository([_jersey()])), "sell 1 #41")
    assert reply.proposal is None
    assert "M (3), L (1)" in reply.answer


def test_the_only_size_in_stock_is_used_without_asking():
    reply = _ask(_actions(FakeRepository([_jersey(M=0, L=2)])), "sell 1 #41")
    assert reply.proposal.payload["sale"]["sale_items"][0]["size"] == "L"


def test_cannot_sell_more_than_is_in_stock():
    reply = _ask(_actions(FakeRepository([_jersey()])), "sell 5 #41 size M")
    assert reply.proposal is None
    assert "Only 3" in reply.answer


def test_unknown_size_lists_what_exists():
    reply = _ask(_actions(FakeRepository([_jersey()])), "sell 1 #41 size XXL")
    assert "no size XXL" in reply.answer


def test_several_matches_ask_which_product():
    other = Product(217, "ARGENTINA HOME MARADONA", "Jersey", 250, 250, {"M": 1})
    reply = _ask(_actions(FakeRepository([_jersey(), other])), "sell 1 argentina size M")
    assert reply.proposal is None
    assert "#41" in reply.answer and "#217" in reply.answer


def test_unknown_product():
    reply = _ask(_actions(FakeRepository([])), "sell 1 unicorn boots size 9")
    assert "couldn't find" in reply.answer


def test_other_questions_are_left_to_rag():
    assert _ask(_actions(FakeRepository([_jersey()])), "Do you have Argentina jerseys?") is None


# --- stock and reports ------------------------------------------------------


def test_stock_check_reports_live_quantities():
    reply = _ask(_actions(FakeRepository([_jersey()])), "how many FC ARGENTINA 25-26 left in size M?")
    assert reply.proposal is None
    assert "Size M: 3 in stock" in reply.answer


def test_sales_report_uses_the_period_and_shop():
    repository = FakeRepository([])
    repository.summary = {"sales": 5, "revenue": 5077.0, "items": 5, "top": [("Anza 999 Blue", 2)]}

    reply = _ask(_actions(repository, options={"shop_id": 2}), "sales this week")

    assert repository.summary_args == (date(2026, 9, 21), date(2026, 9, 26), 2)
    assert "Revenue: ₹5,077" in reply.answer
    assert "Anza 999 Blue × 2" in reply.answer


# --- execute ----------------------------------------------------------------


def test_confirm_posts_the_sale_to_the_store_api(monkeypatch):
    monkeypatch.setenv("SPORTS_STORE_API_URL", "http://store.test/")
    seen = {}

    def store(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1521})

    actions = _actions(FakeRepository([_jersey()]), handler=store)
    payload = _ask(actions, "sell 2 #41 size M").proposal.payload

    message = asyncio.run(actions.execute(payload))

    assert seen["url"] == "http://store.test/sales/addSale"
    assert seen["body"]["sale_items"][0]["quantity"] == 2
    assert message.startswith("Sale #1521 recorded")
    assert "1 left in size M" in message


def test_confirm_rechecks_stock_before_writing(monkeypatch):
    monkeypatch.setenv("SPORTS_STORE_API_URL", "http://store.test")
    repository = FakeRepository([_jersey()])
    actions = _actions(repository, handler=lambda r: pytest.fail("must not post"))
    payload = _ask(actions, "sell 2 #41 size M").proposal.payload
    repository.products = [_jersey(M=1)]  # someone sold one meanwhile

    with pytest.raises(ActionFailedError, match="Only 1"):
        asyncio.run(actions.execute(payload))


def test_confirm_without_store_api_url_fails_clearly(monkeypatch):
    monkeypatch.delenv("SPORTS_STORE_API_URL", raising=False)
    actions = _actions(FakeRepository([_jersey()]))
    payload = _ask(actions, "sell 1 #41 size M").proposal.payload

    with pytest.raises(ActionFailedError, match="SPORTS_STORE_API_URL"):
        asyncio.run(actions.execute(payload))


def test_store_api_error_is_reported(monkeypatch):
    monkeypatch.setenv("SPORTS_STORE_API_URL", "http://store.test")
    actions = _actions(FakeRepository([_jersey()]), handler=lambda r: httpx.Response(500))
    payload = _ask(actions, "sell 1 #41 size M").proposal.payload

    with pytest.raises(ActionFailedError, match="HTTP 500"):
        asyncio.run(actions.execute(payload))


def test_exact_name_wins_over_partial_matches():
    maradona = Product(217, "ARGENTINA HOME MARADONA", "Jersey", 250, 250, {"M": 2})
    collar = Product(216, "ARGENTINA HOME MARADONA COLLAR", "Jersey", 300, 300, {"M": 1})
    actions = _actions(FakeRepository([maradona, collar]))

    stock = _ask(actions, "how many argentina home maradona left?")
    sale = _ask(actions, "sell 1 argentina home maradona size M")

    assert "#216" not in stock.answer
    assert sale.proposal.payload["sale"]["sale_items"][0]["product_id"] == 217


def test_choice_list_puts_sellable_products_first():
    sold_out = _jersey(M=0, L=0)
    in_stock = Product(539, "ARGENTINA 2018 AWAY MESSI", "Jersey", 250, 250, {"M": 1})
    reply = _ask(_actions(FakeRepository([sold_out, in_stock])), "sell 1 argentina")

    assert reply.answer.index("#539") < reply.answer.index("#41")
    assert "sell 1 #539 size M" in reply.answer


def test_last_sales_lists_items_and_customer():
    repository = FakeRepository([])
    repository.last_sales = lambda count, shop_id: [
        {"id": 1524, "date": "2026-09-26", "status": "SaleStatus.COMPLETED", "total": 250.0, "payment": "UPI",
         "customer": None, "items": [("ARGENTINA 2018 AWAY MESSI", "M", 1)]},
    ]

    reply = _ask(_actions(repository), "last order?")

    assert "**Sale #1524** · 2026-09-26 · ₹250 · UPI · walk-in" in reply.answer
    assert "- 1 × ARGENTINA 2018 AWAY MESSI (size M)" in reply.answer


def test_unsupported_shop_command_explains_what_is_possible():
    reply = _ask(_actions(FakeRepository([])), "create a customer?")
    assert reply.proposal is None
    assert "can't do that from the chat" in reply.answer
    assert "sell 1 #539" in reply.answer


def test_all_time_report_covers_every_sale():
    repository = FakeRepository([])
    repository.summary = {"sales": 1524, "revenue": 900000.0, "items": 2229, "top": []}

    reply = _ask(_actions(repository), "total how many orders?")

    assert repository.summary_args[0] == date.min
    assert "**All time**" in reply.answer and "Sales: 1524" in reply.answer


def test_greeting_lists_what_the_assistant_can_do():
    assert "sell 1 #539" in _ask(_actions(FakeRepository([])), "hi").answer
