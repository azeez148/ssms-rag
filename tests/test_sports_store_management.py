"""Products, sale edits and day management from the chat (the sports store plugin)."""

import asyncio
import json

import httpx
import pytest

from app.actions.base import ActionFailedError
from plugins.sports_store.intents import (
    CreateProductIntent,
    DayStatusIntent,
    EndDayIntent,
    ExpenseIntent,
    ListCategoriesIntent,
    ListProductsIntent,
    SaleStatusIntent,
    SalesReportIntent,
    ShowSaleIntent,
    StartDayIntent,
    StockChangeIntent,
    UpdateProductIntent,
    UpdateSaleIntent,
    parse_intent,
)
from plugins.sports_store.repository import Product
from tests.test_sports_store_actions import FakeRepository, _actions, _ask, _jersey


class Store:
    """A fake store API: records requests, answers from `routes` ("GET /path" -> response)."""

    def __init__(self, routes: dict | None = None):
        self.routes = routes or {}
        self.requests: list[tuple[str, str, dict | None, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body, request.headers.get("authorization")))
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"token": "t0k", "user": {}})
        answer = self.routes.get(key, httpx.Response(200, json={}))
        return answer(request) if callable(answer) else answer

    def writes(self):
        return [(method, path, body) for method, path, body, _ in self.requests if method != "GET" and path != "/auth/login"]


@pytest.fixture(autouse=True)
def store_env(monkeypatch):
    monkeypatch.setenv("SPORTS_STORE_API_URL", "http://store.test")
    monkeypatch.setenv("SPORTS_STORE_API_MOBILE", "8089325733")
    monkeypatch.setenv("SPORTS_STORE_API_PASSWORD", "secret")


def _run(coroutine):
    return asyncio.run(coroutine)


# --- parsing -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("list products", ListProductsIntent()),
        ("list products argentina", ListProductsIntent("argentina")),
        ("show out of stock products", ListProductsIntent(None, "out")),
        ("list categories", ListCategoriesIntent()),
        (
            "add product Argentina Home 26 category Jersey price 799 cost 400 sizes S:2 M:5",
            CreateProductIntent(
                "Argentina Home 26",
                {"category": "Jersey", "selling_price": 799, "unit_price": 400, "sizes": {"S": 2, "M": 5}},
            ),
        ),
        ("update #41 price 799 offer 699", UpdateProductIntent("#41", {"selling_price": 799, "discounted_price": 699})),
        ("set price of #41 to 799", UpdateProductIntent("#41", {"selling_price": 799})),
        ("update #41 offer none", UpdateProductIntent("#41", {"discounted_price": None})),
        ("change #41 category to Kids Jersey", UpdateProductIntent("#41", {"category": "Kids Jersey"})),
        ("rename #41 to FC ARGENTINA 26", UpdateProductIntent("#41", {"name": "FC ARGENTINA 26"})),
        ("deactivate #41", UpdateProductIntent("#41", {"active": False})),
        ("add 5 #41 size M", StockChangeIntent("#41", "add", {"M": 5})),
        ("add stock #41 S=2 M=3", StockChangeIntent("#41", "add", {"S": 2, "M": 3})),
        ("set stock of #41 size M to 10", StockChangeIntent("#41", "set", {"M": 10})),
        ("reduce stock #41 size M by 2", StockChangeIntent("#41", "remove", {"M": 2})),
        ("sale #1521", ShowSaleIntent(1521)),
        ("cancel sale #1521", SaleStatusIntent(1521, "CANCELLED")),
        ("cancel the last sale", SaleStatusIntent(None, "CANCELLED")),
        ("mark sale 1521 as shipped", SaleStatusIntent(1521, "SHIPPED")),
        ("update sale #1521 payment gpay", UpdateSaleIntent(1521, payment="upi")),
        ("change sale 1521 qty 2 size L", UpdateSaleIntent(1521, quantity=2, size="L")),
        ("start day with ₹2,000", StartDayIntent(2000)),
        ("add expense 150 for tea", ExpenseIntent(150, "tea")),
        ("spent 50 on auto fare", ExpenseIntent(50, "auto fare")),
        ("end day with 5400 reason: paid supplier", EndDayIntent(5400, "paid supplier")),
        ("day status", DayStatusIntent()),
        ("expenses today", DayStatusIntent()),
        ("list today's sales", SalesReportIntent("today")),
    ],
)
def test_management_commands_are_parsed(message, intent):
    assert parse_intent(message) == intent


@pytest.mark.parametrize("message", ["Show me white Arsenal shorts", "update me", "add #41", "open the catalogue"])
def test_lookalikes_are_left_to_rag(message):
    assert parse_intent(message) is None


# --- products --------------------------------------------------------------------


def test_list_products_shows_price_and_stock():
    repository = FakeRepository([_jersey()])
    reply = _ask(_actions(repository), "list products argentina")
    assert repository.list_args[:2] == ("argentina", None)
    assert "**#41** FC ARGENTINA 25-26 (First Copy Jersey) · ₹650 · M (3), L (1)" in reply.answer


def test_create_product_proposes_then_posts():
    store = Store({"POST /products/addProduct": httpx.Response(200, json={"id": 1700, "name": "Argentina Home 26"})})
    actions = _actions(FakeRepository([]), handler=store)

    reply = _ask(actions, "add product Argentina Home 26 category jersey price 799 cost 400 offer 699 sizes S:2 M:5")

    product = reply.proposal.payload["product"]
    assert store.requests == []
    assert (product["category_id"], product["selling_price"], product["unit_price"], product["discounted_price"]) == (1, 799, 400, 699)
    assert product["size_map"] == [{"size": "S", "quantity": 2}, {"size": "M", "quantity": 5}]
    assert product["is_active"] and product["can_listed"] and product["shop_ids"] == [1]

    message = _run(actions.execute(reply.proposal.payload))
    assert store.writes()[0][1] == "/products/addProduct"
    assert message.startswith("Product #1700")


def test_create_product_asks_for_what_is_missing():
    reply = _ask(_actions(FakeRepository([])), "add product Argentina Home 26 price 799")
    assert reply.proposal is None
    assert "category, cost" in reply.answer


def test_unknown_category_lists_the_choices():
    reply = _ask(_actions(FakeRepository([])), "add product X category boots price 799 cost 400")
    assert "Jersey, Kids Jersey, First Copy Jersey" in reply.answer


def test_update_product_resends_kept_fields_and_changes_one():
    store = Store({"POST /products/updateProduct": httpx.Response(200, json={"id": 41, "name": "FC ARGENTINA 25-26"})})
    actions = _actions(FakeRepository([_jersey()]), handler=store)

    reply = _ask(actions, "update #41 price 799")

    body = reply.proposal.payload["product"]
    assert (body["id"], body["selling_price"], body["unit_price"], body["category_id"], body["discounted_price"]) == (41, 799, 400, 5, 650)
    assert "Price (MRP): ₹700 → ₹799" in reply.proposal.lines
    _run(actions.execute(reply.proposal.payload))
    assert store.writes()[0][2]["selling_price"] == 799


def test_offer_above_price_is_refused():
    reply = _ask(_actions(FakeRepository([_jersey()])), "update #41 offer 900")
    assert reply.proposal is None and "offer price" in reply.answer


def test_add_stock_is_applied_to_the_stock_at_confirm_time():
    store = Store()
    repository = FakeRepository([_jersey()])
    actions = _actions(repository, handler=store)

    reply = _ask(actions, "add 5 #41 size m")
    assert "Size M: 3 → 8" in reply.proposal.lines
    repository.products = [_jersey(S=0, M=1, L=1)]  # two sold meanwhile

    message = _run(actions.execute(reply.proposal.payload))

    assert store.writes() == [("POST", "/products/updateSizeMap", {"product_id": 41, "size_map": {"S": 0, "M": 6, "L": 1}})]
    assert "M: 1 → 6" in message


def test_set_stock_can_add_a_new_size():
    reply = _ask(_actions(FakeRepository([_jersey()])), "set stock of #41 XL=4")
    assert "Size XL: new → 4" in reply.proposal.lines


def test_cannot_remove_more_than_in_stock():
    reply = _ask(_actions(FakeRepository([_jersey()])), "remove 5 #41 size L")
    assert reply.proposal is None and "Only 1" in reply.answer


def test_stock_change_without_size_asks_which():
    reply = _ask(_actions(FakeRepository([_jersey()])), "add 5 #41")
    assert "Which size" in reply.answer


# --- sales -----------------------------------------------------------------------

_SALE_ROW = {"id": 1521, "date": "2026-09-26", "status": "SaleStatus.COMPLETED", "total": 650.0, "payment": "Cash on Delivery",
             "customer": None, "items": [("FC ARGENTINA 25-26", "M", 1)]}
_SALE_API = {
    "id": 1521, "date": "2026-09-26", "total_quantity": 1, "sub_total": 700.0, "total_price": 650.0,
    "payment_type_id": 1, "payment_reference_number": None, "delivery_type_id": 1, "shop_id": 1, "customer_id": 9,
    "status": "COMPLETED",
    "sale_items": [{"id": 1, "sale_id": 1521, "product_id": 41, "product_name": "FC ARGENTINA 25-26",
                    "product_category": "First Copy Jersey", "size": "M", "quantity_available": 4, "quantity": 1,
                    "sale_price": 650.0, "total_price": 650.0}],
}


def _sale_repository():
    repository = FakeRepository([_jersey()])
    repository.sales = [dict(_SALE_ROW)]
    return repository


def test_cancel_sale_is_proposed_and_uses_the_cancel_endpoint():
    store = Store()
    actions = _actions(_sale_repository(), handler=store)

    reply = _ask(actions, "cancel the last sale")

    assert reply.proposal.payload == {"op": "sale_status", "sale_id": 1521, "status": "CANCELLED"}
    assert "Stock goes back and the amount comes off that day's sales." in reply.proposal.lines
    _run(actions.execute(reply.proposal.payload))
    assert store.writes() == [("PUT", "/sales/1521/cancel", None)]


def test_status_change_uses_update_status():
    store = Store()
    actions = _actions(_sale_repository(), handler=store)
    _run(actions.execute(_ask(actions, "mark sale 1521 as shipped").proposal.payload))
    assert store.writes() == [("PUT", "/sales/1521/updateStatus", {"status": "SHIPPED"})]


def test_update_sale_payment_keeps_everything_else():
    store = Store({"GET /sales/1521": httpx.Response(200, json=_SALE_API)})
    actions = _actions(_sale_repository(), handler=store)

    reply = _ask(actions, "update sale #1521 payment upi")

    sale = reply.proposal.payload["sale"]
    assert (sale["payment_type_id"], sale["customer_id"], sale["total_price"], sale["status"]) == (3, 9, 650.0, "COMPLETED")
    assert "Payment: Cash on Delivery → UPI" in reply.proposal.lines
    _run(actions.execute(reply.proposal.payload))
    assert store.writes()[0][:2] == ("POST", "/sales/1521/updateSale")


def test_update_sale_quantity_recomputes_totals_and_checks_stock():
    store = Store({"GET /sales/1521": httpx.Response(200, json=_SALE_API)})
    actions = _actions(_sale_repository(), handler=store)

    reply = _ask(actions, "change sale 1521 qty 3")

    sale = reply.proposal.payload["sale"]
    item = sale["sale_items"][0]
    assert (item["quantity"], item["total_price"], sale["total_quantity"]) == (3, 1950.0, 3)
    assert (sale["total_price"], sale["sub_total"]) == (1950.0, 2100.0)
    # 3 in stock + the 1 the store puts back = 4 available
    assert _ask(actions, "change sale 1521 qty 5").answer.startswith("Only 4")


def test_update_sale_delivery_adds_the_charge_difference():
    store = Store({"GET /sales/1521": httpx.Response(200, json=_SALE_API)})
    reply = _ask(_actions(_sale_repository(), handler=store), "update sale 1521 express delivery")
    assert reply.proposal.payload["sale"]["total_price"] == 715.0


def test_show_sale():
    reply = _ask(_actions(_sale_repository()), "sale #1521")
    assert "**Sale #1521** · 2026-09-26 · ₹650" in reply.answer


# --- day management ---------------------------------------------------------------

_OPEN_DAY = {
    "id": 143, "shop_id": 1, "start_time": "2026-09-26T09:05:00", "end_time": None, "opening_balance": 2000.0,
    "total_expense": 100.0, "cash_in_hand": 3550.0, "total_sales": 2150.0, "total_cash_sales": 1650.0,
    "total_account_sales": 500.0, "expenses": [{"id": 1, "description": "tea", "amount": 100.0, "timestamp": "2026-09-26T10:00:00"}],
}


def _today(started=True, ended=False, day=None):
    entry = {"shop_id": 1, "shop_name": "Main", "day_started": started, "day_ended": ended, "active_day": day if started else None}
    return httpx.Response(200, json=[entry])


def test_start_day_logs_in_and_posts_the_opening_balance():
    store = Store({
        "GET /day-management/today": _today(started=False),
        "POST /day-management/startDay": httpx.Response(200, json={**_OPEN_DAY, "total_expense": 0}),
    })
    actions = _actions(FakeRepository([]), handler=store)

    reply = _ask(actions, "start day with 2000")
    message = _run(actions.execute(reply.proposal.payload))

    assert store.writes() == [("POST", "/day-management/startDay", {"opening_balance": 2000.0, "shop_id": 1})]
    assert all(auth == "Bearer t0k" for method, path, _, auth in store.requests if path.startswith("/day-management"))
    assert message == "Day #143 is open. Opening balance ₹2,000."


def test_start_day_when_already_open_says_so():
    store = Store({"GET /day-management/today": _today(day=_OPEN_DAY)})
    reply = _ask(_actions(FakeRepository([]), handler=store), "start day with 2000")
    assert reply.proposal is None and "already open" in reply.answer


def test_start_day_without_amount_asks_for_it():
    store = Store({"GET /day-management/today": _today(started=False)})
    assert "How much cash" in _ask(_actions(FakeRepository([]), handler=store), "start the day").answer


def test_expense_needs_an_open_day():
    store = Store({"GET /day-management/today": _today(started=False)})
    reply = _ask(_actions(FakeRepository([]), handler=store), "add expense 150 for tea")
    assert reply.proposal is None and "hasn't been started" in reply.answer


def test_expense_is_added_to_the_open_day():
    store = Store({"GET /day-management/today": _today(day=_OPEN_DAY)})
    actions = _actions(FakeRepository([]), handler=store)

    reply = _ask(actions, "spent 50 on auto fare")
    assert "Day's expenses: ₹100 → ₹150" in reply.proposal.lines
    _run(actions.execute(reply.proposal.payload))

    assert store.writes() == [
        ("POST", "/day-management/addExpense", {"description": "auto fare", "amount": 50.0, "day_id": 143, "shop_id": 1})
    ]


def test_end_day_without_amount_tells_the_expected_cash():
    store = Store({"GET /day-management/today": _today(day=_OPEN_DAY)})
    reply = _ask(_actions(FakeRepository([]), handler=store), "end day")
    assert reply.proposal is None and "Expected cash in the drawer: **₹3,550**" in reply.answer


def test_end_day_shows_variance_and_posts_the_count():
    summary = {"day_id": 143, "total_sales": 2150.0, "total_expense": 100.0, "cash_in_hand": 3550.0,
               "closing_balance": 3500.0, "variance": -50.0}
    store = Store({
        "GET /day-management/today": _today(day=_OPEN_DAY),
        "POST /day-management/endDay/143": httpx.Response(200, json=summary),
    })
    actions = _actions(FakeRepository([]), handler=store)

    reply = _ask(actions, "end day with 3500 reason: gave change")
    assert "Variance: −₹50" in reply.proposal.lines
    message = _run(actions.execute(reply.proposal.payload))

    assert store.writes() == [("POST", "/day-management/endDay/143", {"closing_balance": 3500.0, "variance_reason": "gave change"})]
    assert "variance −₹50" in message


def test_day_status_lists_totals_and_expenses():
    store = Store({"GET /day-management/today": _today(day=_OPEN_DAY)})
    answer = _ask(_actions(FakeRepository([]), handler=store), "day status").answer
    assert "Expected cash in hand: ₹3,550" in answer and "- tea: ₹100" in answer


def test_day_management_without_a_login_explains_the_setup(monkeypatch):
    monkeypatch.delenv("SPORTS_STORE_API_MOBILE")
    monkeypatch.delenv("SPORTS_STORE_API_TOKEN", raising=False)
    reply = _ask(_actions(FakeRepository([]), handler=Store()), "day status")
    assert "SPORTS_STORE_API_MOBILE" in reply.answer


def test_expired_token_logs_in_again():
    calls = {"today": 0}

    def today(request):
        calls["today"] += 1
        return httpx.Response(401) if calls["today"] == 1 else _today(started=False)

    store = Store({"GET /day-management/today": today})
    reply = _ask(_actions(FakeRepository([]), handler=store), "day status")
    assert "hasn't been started" in reply.answer
    assert [path for _, path, _, _ in store.requests].count("/auth/login") == 2


def test_store_errors_are_reported_with_its_message():
    store = Store({
        "GET /day-management/today": _today(started=False),
        "POST /day-management/startDay": httpx.Response(403, json={"detail": "You do not have access to this shop's day management."}),
    })
    actions = _actions(FakeRepository([]), handler=store)
    payload = _ask(actions, "start day with 100").proposal.payload
    with pytest.raises(ActionFailedError, match="do not have access"):
        _run(actions.execute(payload))
