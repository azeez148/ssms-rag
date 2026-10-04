"""Shop-staff chat actions for the sports store: products (list, add, edit,
stock), sales (record, show, edit, cancel, reports) and day management (start
day, expenses, end day). Reads are answered live from the store database;
every write is proposed first and made through the store API only on confirm.

Wired in by config/sports_store_source.yaml's `actions:` block; the RAG core
only sees the generic ActionHandler interface.
"""

import asyncio
import os
from datetime import date, timedelta

import httpx
from sqlalchemy import create_engine

from app.actions.base import ActionFailedError, ActionHandler, ActionProposal, ActionReply
from app.schemas.chat import ConversationTurn
from app.sources.config import PostgresSourceConfig
from plugins.sports_store.catalog import CatalogActions
from plugins.sports_store.day import DayActions
from plugins.sports_store.intents import (
    CreateProductIntent,
    DayStatusIntent,
    EndDayIntent,
    ExpenseIntent,
    HelpIntent,
    LastSalesIntent,
    ListCategoriesIntent,
    ListProductsIntent,
    SaleStatusIntent,
    SalesReportIntent,
    SellIntent,
    ShowSaleIntent,
    StartDayIntent,
    StockChangeIntent,
    StockIntent,
    UnsupportedIntent,
    UpdateProductIntent,
    UpdateSaleIntent,
    parse_intent,
)
from plugins.sports_store.repository import Product, StoreRepository
from plugins.sports_store.sale_edits import SaleEditActions
from plugins.sports_store.store_api import StoreApi, error_detail

CAPABILITIES = (
    "Here's what I can do from the chat:\n\n"
    "**Products**\n"
    "- List: “list products”, “list products argentina”, “show out of stock products”, “list categories”\n"
    "- Live stock: “how many FC ARGENTINA 25-26 left in size M?”\n"
    "- Add: “add product Argentina Home 26 category Jersey price 799 cost 400 sizes S:2 M:5”\n"
    "- Edit: “update #41 price 799”, “update #41 offer 699”, “change #41 category to Jersey”, "
    "“rename #41 to …”, “deactivate #41”\n"
    "- Stock: “add 5 #41 size M”, “remove 1 #41 size M”, “set stock of #41 M=5 L=2”\n\n"
    "**Sales**\n"
    "- Record: “sell 1 #539 size M to Rahul 9876543210 upi”\n"
    "- Show: “sale #1521”, “last sale”, “last 5 sales”\n"
    "- Edit: “update sale #1521 payment upi”, “change sale #1521 qty 2 size L”\n"
    "- Cancel or status: “cancel sale #1521”, “mark sale #1521 as shipped”\n"
    "- Reports: “today's sales”, “sales this week”, “how many orders this month?”, “total sales”\n\n"
    "**Day**\n"
    "- “start day with 2000”, “add expense 150 for tea”, “day status”, “end day with 5400”\n\n"
    "Anything else, like “Do you have Argentina jerseys?”, is answered from the catalogue."
)

_PERIOD_LABELS = {
    "today": "Today",
    "yesterday": "Yesterday",
    "this_week": "This week",
    "this_month": "This month",
    "all_time": "All time",
}


def _named(products: list[Product], query: str) -> list[Product]:
    """Products whose name is exactly the query (ignoring case)."""
    return [p for p in products if p.name.strip().lower() == query.strip().lower()]


class SportsStoreActions(CatalogActions, SaleEditActions, DayActions, ActionHandler):
    """Options (from the source config):
      api_url_env       env var holding the store API base URL (default SPORTS_STORE_API_URL)
      api_mobile_env    env var holding the store login mobile, for day management
                        (default SPORTS_STORE_API_MOBILE)
      api_password_env  env var holding its password (default SPORTS_STORE_API_PASSWORD)
      api_token_env     env var holding a ready token instead of a login (default SPORTS_STORE_API_TOKEN)
      shop_id           shop the sales are recorded against (default 1)
      default_payment   payment type name used when the message names none (default "cash")
      default_delivery  delivery type name used when the message names none (default "pickup")
      currency          prefix for amounts (default "₹")
      walk_in_customer  {name, mobile, address} sent for sales with no customer named; defaults
                        to the POS's own walk-in ("In-store Customer", 0000000000) so reports match
    """

    def __init__(
        self,
        config: PostgresSourceConfig,
        options: dict,
        repository: StoreRepository | None = None,
        http_client: httpx.AsyncClient | None = None,
        today=date.today,
    ):
        self._repository = repository or StoreRepository(
            create_engine(os.environ[config.connection.env], pool_pre_ping=True)
        )
        self._api = StoreApi(options, http_client)
        self._shop_id = int(options.get("shop_id", 1))
        self._default_payment = str(options.get("default_payment", "cash"))
        self._default_delivery = str(options.get("default_delivery", "pickup"))
        self._currency = str(options.get("currency", "₹"))
        self._walk_in = {"name": "In-store Customer", "mobile": "0000000000", "address": "In-store"}
        self._walk_in.update(options.get("walk_in_customer") or {})
        self._today = today

    # ---- ActionHandler -----------------------------------------------------

    async def handle(self, question: str, history: list[ConversationTurn]) -> ActionReply | None:
        intent = parse_intent(question)
        if intent is None:
            return None
        if isinstance(intent, HelpIntent):
            return ActionReply("Hi! " + CAPABILITIES)
        if isinstance(intent, UnsupportedIntent):
            return ActionReply(
                "I can't do that from the chat yet. Please use the store app for it.\n\n" + CAPABILITIES
            )
        try:
            return await self._answer(intent)
        except ActionFailedError as exc:
            # A missing API URL or login, or the store being down: say so instead of failing the chat.
            return ActionReply(str(exc))

    async def _answer(self, intent) -> ActionReply:
        # Database reads and proposals built from them run in a thread; ones that
        # ask the store API (sales being edited, the day's status) are async.
        in_thread = {
            SellIntent: self._propose_sale,
            StockIntent: self._stock,
            SalesReportIntent: self._sales_report,
            LastSalesIntent: self._last_sales,
            ListProductsIntent: self._list_products,
            ListCategoriesIntent: lambda _: self._list_categories(),
            CreateProductIntent: self._propose_create_product,
            UpdateProductIntent: self._propose_update_product,
            StockChangeIntent: self._propose_stock_change,
            ShowSaleIntent: self._show_sale,
            SaleStatusIntent: self._propose_sale_status,
        }
        if type(intent) in in_thread:
            return await asyncio.to_thread(in_thread[type(intent)], intent)
        async_handlers = {
            UpdateSaleIntent: self._propose_sale_update,
            StartDayIntent: self._propose_start_day,
            ExpenseIntent: self._propose_expense,
            EndDayIntent: self._propose_end_day,
            DayStatusIntent: lambda _: self._day_status(),
        }
        return await async_handlers[type(intent)](intent)

    async def execute(self, payload: dict) -> str:
        operations = {
            "add_sale": self._add_sale,
            "update_sale": self._update_sale,
            "sale_status": self._change_sale_status,
            "create_product": self._create_product,
            "update_product": self._update_product,
            "stock": self._change_stock,
            "start_day": self._start_day,
            "add_expense": self._add_expense,
            "end_day": self._end_day,
        }
        return await operations[payload["op"]](payload)

    async def _add_sale(self, payload: dict) -> str:
        sale = payload["sale"]
        # Stock may have moved since the proposal (another till, the website).
        item = sale["sale_items"][0]
        product = await asyncio.to_thread(self._repository.get_product, item["product_id"])
        stock = product.stock_for(item["size"]) if product else None
        if stock is None or stock[1] < item["quantity"]:
            left = stock[1] if stock else 0
            raise ActionFailedError(
                f"Only {left} of {item['product_name']} in size {item['size']} left now, so the sale wasn't recorded."
            )
        item["quantity_available"] = stock[1]

        response = await self._api.post("/sales/addSale", sale)
        if response.status_code >= 400:
            raise ActionFailedError(
                f"The store API refused the sale ({error_detail(response)}). "
                "Check the Sales page before trying again."
            )

        sale_id = response.json().get("id")
        remaining = stock[1] - item["quantity"]
        return (
            f"Sale #{sale_id} recorded: {item['quantity']} × {item['product_name']} (size {item['size']}) "
            f"for {self._money(sale['total_price'])}. {remaining} left in size {item['size']}."
        )

    # ---- stock -------------------------------------------------------------

    def _stock(self, intent: StockIntent) -> ActionReply:
        products, exact = self._repository.find_products(intent.product)
        if not products:
            return ActionReply(f"I couldn't find a product matching “{intent.product}”.")
        products = _named(products, intent.product) or products

        lines = [] if exact else [f"Nothing matches “{intent.product}” exactly. Closest products:", ""]
        for product in products:
            lines.append(f"**{product.name}** (#{product.id}, {product.category}) · {self._money(product.sale_price)}")
            if intent.size:
                stock = product.stock_for(intent.size)
                lines.append(f"- Size {stock[0]}: {stock[1]} in stock" if stock else f"- No size {intent.size}")
            elif product.sizes:
                lines.append("- " + ", ".join(f"{size}: {quantity}" for size, quantity in product.sizes.items()))
                lines.append(f"- Total: {sum(product.sizes.values())} in stock")
            else:
                lines.append("- No stock recorded")
            lines.append("")
        lines.append("_Live stock from the store._")
        return ActionReply("\n".join(lines).strip())

    # ---- sales summary -----------------------------------------------------

    def _sales_report(self, intent: SalesReportIntent) -> ActionReply:
        today = self._today()
        start, end = {
            "today": (today, today),
            "yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
            "this_week": (today - timedelta(days=today.weekday()), today),
            "this_month": (today.replace(day=1), today),
            "all_time": (date.min, today),
        }[intent.period]
        summary = self._repository.sales_summary(start, end, self._shop_id)

        label = _PERIOD_LABELS[intent.period]
        if summary["sales"] == 0:
            return ActionReply(f"{label}: no sales recorded yet.")
        lines = [
            f"**{label}**"
            + ("" if intent.period == "all_time" else f" ({start.isoformat()}{'' if start == end else ' to ' + end.isoformat()})"),
            f"- Sales: {summary['sales']}",
            f"- Items sold: {summary['items']}",
            f"- Revenue: {self._money(summary['revenue'])}",
        ]
        if summary["top"]:
            lines += ["", "Top sellers:"] + [f"- {name} × {quantity}" for name, quantity in summary["top"]]
        lines += ["", "_Cancelled sales excluded._"]
        return ActionReply("\n".join(lines))

    # ---- recent sales ------------------------------------------------------

    def _last_sales(self, intent: LastSalesIntent) -> ActionReply:
        sales = self._repository.last_sales(intent.count, self._shop_id)
        if not sales:
            return ActionReply("No sales recorded yet.")

        lines = []
        for sale in sales:
            status = "" if sale["status"].upper().endswith("COMPLETED") else f" · {sale['status'].split('.')[-1].title()}"
            who = sale["customer"] or "walk-in"
            lines.append(f"**Sale #{sale['id']}** · {sale['date']} · {self._money(sale['total'])} · {sale['payment'] or '—'} · {who}{status}")
            lines += [f"- {qty} × {name} (size {size})" for name, size, qty in sale["items"]]
            lines.append("")
        return ActionReply("\n".join(lines).strip())

    # ---- sell --------------------------------------------------------------

    def _propose_sale(self, intent: SellIntent) -> ActionReply:
        if intent.quantity < 1:
            return ActionReply("The quantity must be at least 1.")

        products, exact = self._repository.find_products(intent.product)
        product = self._single(products, exact, intent.product)
        if isinstance(product, ActionReply):
            return product

        size = self._pick_size(product, intent.size)
        if isinstance(size, ActionReply):
            return size
        size_name, in_stock = size
        if in_stock < intent.quantity:
            return ActionReply(
                f"Only {in_stock} of **{product.name}** left in size {size_name}, so I can't sell {intent.quantity}."
            )

        if bool(intent.customer_name) != bool(intent.mobile):
            return ActionReply(
                "To attach a customer I need both a name and a 10-digit mobile number, "
                "e.g. “… to Rahul 9876543210”. Leave both out for a walk-in sale."
            )

        payment = self._lookup(self._repository.payment_types(), intent.payment or self._default_payment)
        if payment is None:
            names = ", ".join(name for _, name in self._repository.payment_types())
            return ActionReply(f"I don't know the payment type “{intent.payment}”. Use one of: {names}.")
        delivery = self._lookup(self._repository.delivery_types(), intent.delivery or self._default_delivery)
        if delivery is None:
            names = ", ".join(row[1] for row in self._repository.delivery_types())
            return ActionReply(f"I don't know the delivery type “{intent.delivery}”. Use one of: {names}.")
        delivery_charge = float(delivery[2])

        quantity = intent.quantity
        items_total = product.sale_price * quantity
        total = items_total + delivery_charge
        sale = {
            "date": self._today().isoformat(),
            "total_quantity": quantity,
            "sub_total": product.selling_price * quantity,
            "total_price": total,
            "payment_type_id": payment[0],
            "delivery_type_id": delivery[0],
            "shop_id": self._shop_id,
            "status": "COMPLETED",
            "sale_items": [
                {
                    "product_id": product.id,
                    "product_name": product.name,
                    "product_category": product.category,
                    "size": size_name,
                    "quantity_available": in_stock,
                    "quantity": quantity,
                    "sale_price": product.sale_price,
                    "total_price": items_total,
                }
            ],
        }
        # customer_id 0 = the store finds (by mobile) or creates the customer, as its POS does.
        sale["customer_id"] = 0
        sale["customer_address"] = (
            {"name": intent.customer_name, "mobile": intent.mobile} if intent.mobile else dict(self._walk_in)
        )

        price = self._money(product.sale_price)
        if product.sale_price != product.selling_price:
            price += f" (MRP {self._money(product.selling_price)})"
        lines = [
            f"{product.name} (#{product.id}) · size {size_name}",
            f"Qty {quantity} × {price} = {self._money(items_total)}",
            f"Delivery: {delivery[1]}" + (f" (+{self._money(delivery_charge)})" if delivery_charge else ""),
            f"Payment: {payment[1]}",
            f"Customer: {intent.customer_name}, {intent.mobile}" if intent.mobile else f"Customer: {self._walk_in['name']}",
            f"Total: {self._money(total)}",
            f"Stock after sale: {in_stock - quantity} left in size {size_name}",
        ]
        return ActionReply(
            f"Ready to record this sale of **{product.name}**. Check the details and confirm.",
            ActionProposal(
                title="Record this sale?", lines=lines, confirm_label="Confirm sale", payload={"op": "add_sale", "sale": sale}
            ),
        )

    def _single(
        self, products: list[Product], exact: bool, query: str, example: str = "sell 1 {id} size {size}"
    ) -> Product | ActionReply:
        if not products:
            return ActionReply(f"I couldn't find a product matching “{query}”.")
        named = _named(products, query)
        if len(named) == 1:
            return named[0]
        if exact and len(products) == 1:
            return products[0]

        # Sellable ones first, and the example uses one of them.
        example_text = example
        products = sorted(products, key=lambda p: not any(q > 0 for q in p.sizes.values()))
        example = products[0]
        example_size = next((size for size, q in example.sizes.items() if q > 0), "M")

        intro = "Which product did you mean?" if exact else f"Nothing matches “{query}” exactly. Did you mean one of these?"
        lines = [intro, ""]
        for product in products:
            in_stock = ", ".join(f"{size} ({qty})" for size, qty in product.sizes.items() if qty > 0) or "out of stock"
            lines.append(f"- **#{product.id}** {product.name} ({product.category}) · {self._money(product.sale_price)} · {in_stock}")
        lines += ["", "Reply with the product number, e.g. “" + example_text.format(id=f"#{example.id}", size=example_size) + "”."]
        return ActionReply("\n".join(lines))

    def _pick_size(self, product: Product, size: str | None) -> tuple[str, int] | ActionReply:
        in_stock = {name: qty for name, qty in product.sizes.items() if qty > 0}
        if not in_stock:
            return ActionReply(f"**{product.name}** is out of stock in every size.")
        if size is None:
            if len(in_stock) == 1:
                return next(iter(in_stock.items()))
            sizes = ", ".join(f"{name} ({qty})" for name, qty in in_stock.items())
            return ActionReply(f"Which size of **{product.name}**? In stock: {sizes}.")
        stock = product.stock_for(size)
        if stock is None:
            sizes = ", ".join(f"{name} ({qty})" for name, qty in in_stock.items())
            return ActionReply(f"**{product.name}** has no size {size}. In stock: {sizes}.")
        return stock

    @staticmethod
    def _lookup(rows: list[tuple], wanted: str) -> tuple | None:
        """Row whose name contains `wanted` ("cash" -> "Cash on Delivery", "express" -> "PO-Express Delivery")."""
        wanted = wanted.lower().replace(" ", "")
        for row in rows:
            if wanted in row[1].lower().replace(" ", "").replace("-", ""):
                return row
        return None

    def _money(self, amount: float) -> str:
        return f"{self._currency}{amount:,.0f}" if float(amount).is_integer() else f"{self._currency}{amount:,.2f}"
