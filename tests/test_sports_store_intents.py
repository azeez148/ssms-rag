import pytest

from plugins.sports_store.intents import (
    HelpIntent,
    LastSalesIntent,
    SalesReportIntent,
    SellIntent,
    StockIntent,
    UnsupportedIntent,
    parse_intent,
)


def test_full_sale_request_is_parsed():
    assert parse_intent("sell 2 FC ARGENTINA 25-26 size M to Rahul 9876543210 upi") == SellIntent(
        product="FC ARGENTINA 25-26", quantity=2, size="M", customer_name="Rahul", mobile="9876543210", payment="upi"
    )


@pytest.mark.parametrize(
    ("message", "product", "quantity", "size"),
    [
        ("Sold one Arsenal shorts L by cash", "Arsenal shorts", 1, "L"),
        ("sell FC Argentina 25-26 x2 size m", "FC Argentina 25-26", 2, "M"),
        ("bill #41 size M", "#41", 1, "M"),
        ("record a sale of 3 pcs adidas 3line black size 32", "adidas 3line black", 3, "32"),
        ("sell arsenal shorts", "arsenal shorts", 1, None),
    ],
)
def test_sale_variants(message, product, quantity, size):
    intent = parse_intent(message)
    assert isinstance(intent, SellIntent)
    assert (intent.product, intent.quantity, intent.size) == (product, quantity, size)


def test_wallet_apps_are_upi_and_delivery_is_picked_up():
    intent = parse_intent("sell 1 #41 size M paid via gpay, express delivery")
    assert (intent.payment, intent.delivery) == ("upi", "express delivery")


def test_for_is_part_of_the_product_not_a_customer():
    intent = parse_intent("sell 1 jersey for kids size 28")
    assert intent.customer_name is None
    assert intent.product == "jersey for kids"


@pytest.mark.parametrize(
    ("message", "product", "size"),
    [
        ("How many FC Argentina 25-26 are left in size M?", "FC Argentina 25-26", "M"),
        ("Is the argentina jersey in stock?", "argentina jersey", None),
        ("check stock arsenal shorts", "arsenal shorts", None),
    ],
)
def test_stock_checks(message, product, size):
    assert parse_intent(message) == StockIntent(product, size)


@pytest.mark.parametrize(
    ("message", "period"),
    [("today's sales", "today"), ("how much did we sell this week?", "this_week"), ("Yesterday sales", "yesterday")],
)
def test_sales_reports(message, period):
    assert parse_intent(message) == SalesReportIntent(period)


@pytest.mark.parametrize(
    "message",
    ["Do you have Argentina jerseys?", "Which sizes are available?", "sell", "what's the stock?", "", "Show me white Arsenal shorts"],
)
def test_everything_else_is_left_to_rag(message):
    assert parse_intent(message) is None


@pytest.mark.parametrize(
    ("message", "count"),
    [("last order?", 1), ("last sale", 1), ("recent orders", 5), ("last 3 sales", 3), ("latest two orders", 2), ("last 50 sales", 10)],
)
def test_recent_sales(message, count):
    assert parse_intent(message) == LastSalesIntent(count)


@pytest.mark.parametrize("message", ["create a customer?", "delete product #41", "refund the last sale"])
def test_shop_commands_the_chat_cannot_do_are_not_sent_to_rag(message):
    assert isinstance(parse_intent(message), UnsupportedIntent)


@pytest.mark.parametrize("message", ["hi", "Hello!", "what can you do?", "thanks"])
def test_greetings_get_the_help_text(message):
    assert parse_intent(message) == HelpIntent()


@pytest.mark.parametrize(
    ("message", "period"),
    [("total  how many orders?", "all_time"), ("how many orders?", "today"), ("total sales", "all_time"),
     ("how many orders this month", "this_month")],
)
def test_order_counts_are_sales_reports_not_product_searches(message, period):
    assert parse_intent(message) == SalesReportIntent(period)
