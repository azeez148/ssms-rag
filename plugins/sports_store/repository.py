"""Live, read-only queries against the store database.

The RAG index is a snapshot taken at ingestion; stock and sales change
between ingestions, so anything about quantities is read here, live. Nothing
in this module writes. Every write goes through the store's own API (see
store_api.py) so its stock, day-total and notification logic stays in one place.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Engine, bindparam, text

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_PRODUCT_ID = re.compile(r"^\s*(?:#|id\s*:?\s*)(\d+)\s*$", re.I)
MAX_MATCHES = 5

_PRODUCT_COLUMNS = """
    SELECT p.id, p.name, COALESCE(c.name, '') AS category, p.selling_price,
           COALESCE(p.discounted_price, p.selling_price) AS sale_price
    FROM products p
    LEFT JOIN categories c ON c.id = p.category_id
    WHERE p.is_active = true AND p.can_listed = true
"""


@dataclass
class Product:
    id: int
    name: str
    category: str
    selling_price: float
    sale_price: float
    # size -> quantity in stock, in the store's size order
    sizes: dict[str, int] = field(default_factory=dict)

    def stock_for(self, size: str) -> tuple[str, int] | None:
        """The store's own spelling of `size` and its quantity, matched case-insensitively."""
        for name, quantity in self.sizes.items():
            if name.strip().lower() == size.strip().lower():
                return name, quantity
        return None


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def search_tokens(query: str) -> list[str]:
    return [_stem(token) for token in _TOKEN.findall(query.lower())]


class StoreRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def find_products(self, query: str) -> tuple[list[Product], bool]:
        """Products matching `query` ("#41" / "id 41", or name words), best
        first, and whether every word matched (False = closest guesses)."""
        by_id = _PRODUCT_ID.match(query)
        if by_id:
            product = self.get_product(int(by_id.group(1)))
            return ([product] if product else []), True

        tokens = search_tokens(query)
        if not tokens:
            return [], True

        params = {f"t{i}": f"%{token}%" for i, token in enumerate(tokens)}
        matches = [f"(p.name ILIKE :t{i} OR c.name ILIKE :t{i})" for i in range(len(tokens))]
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(f"{_PRODUCT_COLUMNS} AND {' AND '.join(matches)} ORDER BY length(p.name), p.id LIMIT :limit"),
                {**params, "limit": MAX_MATCHES},
            ).all()
            exact = True
            if not rows:
                # Nothing has every word: rank by how many words match instead.
                score = " + ".join(f"(CASE WHEN {m} THEN 1 ELSE 0 END)" for m in matches)
                rows = connection.execute(
                    text(
                        f"SELECT id, name, category, selling_price, sale_price FROM ("
                        f"SELECT p.id, p.name, COALESCE(c.name, '') AS category, p.selling_price, "
                        f"COALESCE(p.discounted_price, p.selling_price) AS sale_price, {score} AS score "
                        f"FROM products p LEFT JOIN categories c ON c.id = p.category_id "
                        f"WHERE p.is_active = true AND p.can_listed = true) ranked "
                        f"WHERE score > 0 ORDER BY score DESC, length(name), id LIMIT :limit"
                    ),
                    {**params, "limit": MAX_MATCHES},
                ).all()
                exact = False
            products = [Product(row.id, row.name, row.category, row.selling_price, row.sale_price) for row in rows]
            self._attach_sizes(connection, products)
        return products, exact

    def get_product(self, product_id: int, include_inactive: bool = False) -> Product | None:
        columns = _PRODUCT_COLUMNS
        if include_inactive:
            columns = columns.replace("p.is_active = true AND p.can_listed = true", "true")
        with self._engine.connect() as connection:
            row = connection.execute(text(f"{columns} AND p.id = :id"), {"id": product_id}).first()
            if row is None:
                return None
            product = Product(row.id, row.name, row.category, row.selling_price, row.sale_price)
            self._attach_sizes(connection, [product])
        return product

    def product_fields(self, product_id: int) -> dict | None:
        """The editable columns the store's updateProduct expects back, for any
        product (active or not), so an edit can resend the fields it keeps."""
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT p.id, p.name, p.description, p.unit_price, p.selling_price, p.discounted_price, "
                    "p.category_id, COALESCE(c.name, '') AS category, p.is_active, p.can_listed "
                    "FROM products p LEFT JOIN categories c ON c.id = p.category_id WHERE p.id = :id"
                ),
                {"id": product_id},
            ).first()
        return dict(row._mapping) if row else None

    def categories(self) -> list[tuple[int, str]]:
        with self._engine.connect() as connection:
            return [tuple(row) for row in connection.execute(text("SELECT id, name FROM categories ORDER BY id"))]

    def list_products(self, query: str | None, stock: str | None, limit: int) -> tuple[list[Product], int]:
        """Up to `limit` products whose name or category has every word of
        `query` (all products if None), newest first, and how many match in
        total. `stock` is "in" / "out" to keep only in-stock / sold-out ones."""
        tokens = search_tokens(query or "")
        params: dict = {f"t{i}": f"%{token}%" for i, token in enumerate(tokens)}
        where = "".join(f" AND (p.name ILIKE :t{i} OR c.name ILIKE :t{i})" for i in range(len(tokens)))
        if stock:
            in_stock = "EXISTS (SELECT 1 FROM product_sizes ps WHERE ps.product_id = p.id AND ps.quantity > 0)"
            where += f" AND {'' if stock == 'in' else 'NOT '}{in_stock}"
        with self._engine.connect() as connection:
            total = connection.execute(
                text(f"SELECT count(*) FROM ({_PRODUCT_COLUMNS}{where}) matched"), params
            ).scalar_one()
            rows = connection.execute(
                text(f"{_PRODUCT_COLUMNS}{where} ORDER BY p.id DESC LIMIT :limit"), {**params, "limit": limit}
            ).all()
            products = [Product(row.id, row.name, row.category, row.selling_price, row.sale_price) for row in rows]
            self._attach_sizes(connection, products)
        return products, int(total)

    def payment_types(self) -> list[tuple[int, str]]:
        with self._engine.connect() as connection:
            return [tuple(row) for row in connection.execute(text("SELECT id, name FROM payment_types ORDER BY id"))]

    def delivery_types(self) -> list[tuple[int, str, float]]:
        with self._engine.connect() as connection:
            return [
                tuple(row)
                for row in connection.execute(text("SELECT id, name, charge FROM delivery_types ORDER BY id"))
            ]

    def sales_summary(self, start: date, end: date, shop_id: int | None) -> dict:
        """Totals for non-cancelled sales dated start..end inclusive. The store
        keeps sales.date as 'YYYY-MM-DD' text, so ISO strings compare correctly."""
        params = {"start": start.isoformat(), "end": end.isoformat(), "shop_id": shop_id}
        where = "s.date BETWEEN :start AND :end AND s.status <> 'CANCELLED'" + (
            " AND s.shop_id = :shop_id" if shop_id is not None else ""
        )
        with self._engine.connect() as connection:
            totals = connection.execute(
                text(
                    f"SELECT count(*) AS sales, COALESCE(sum(s.total_price), 0) AS revenue, "
                    f"COALESCE(sum(s.total_quantity), 0) AS items FROM sales s WHERE {where}"
                ),
                params,
            ).one()
            top = connection.execute(
                text(
                    f"SELECT si.product_name, sum(si.quantity) AS quantity FROM sale_items si "
                    f"JOIN sales s ON s.id = si.sale_id WHERE {where} "
                    f"GROUP BY si.product_name ORDER BY quantity DESC, si.product_name LIMIT 5"
                ),
                params,
            ).all()
        return {
            "sales": totals.sales,
            "revenue": float(totals.revenue),
            "items": int(totals.items),
            "top": [(row.product_name, int(row.quantity)) for row in top],
        }

    def last_sales(self, count: int, shop_id: int | None) -> list[dict]:
        """The most recent sales, newest first, with their items."""
        where = "WHERE s.shop_id = :shop_id" if shop_id is not None else ""
        return self._sales(f"{where} ORDER BY s.id DESC LIMIT :count", {"shop_id": shop_id, "count": count})

    def get_sale(self, sale_id: int) -> dict | None:
        sales = self._sales("WHERE s.id = :id", {"id": sale_id})
        return sales[0] if sales else None

    def _sales(self, where: str, params: dict) -> list[dict]:
        with self._engine.connect() as connection:
            sales = connection.execute(
                text(
                    "SELECT s.id, s.date, s.status, s.total_price, pt.name AS payment, "
                    "NULLIF(trim(concat_ws(' ', c.first_name, c.last_name)), '') AS customer "
                    "FROM sales s LEFT JOIN payment_types pt ON pt.id = s.payment_type_id "
                    f"LEFT JOIN customers c ON c.id = s.customer_id {where}"
                ),
                params,
            ).all()
            if not sales:
                return []
            items = connection.execute(
                text(
                    "SELECT sale_id, product_name, size, quantity FROM sale_items "
                    "WHERE sale_id IN :ids ORDER BY id"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": [sale.id for sale in sales]},
            ).all()
        by_sale: dict[int, list[tuple[str, str, int]]] = {}
        for item in items:
            by_sale.setdefault(item.sale_id, []).append((item.product_name, item.size, int(item.quantity or 0)))
        return [
            {
                "id": sale.id,
                "date": sale.date,
                "status": str(sale.status),
                "total": float(sale.total_price or 0),
                "payment": sale.payment,
                "customer": sale.customer,
                "items": by_sale.get(sale.id, []),
            }
            for sale in sales
        ]

    @staticmethod
    def _attach_sizes(connection, products: list[Product]) -> None:
        if not products:
            return
        rows = connection.execute(
            text(
                "SELECT product_id, size, COALESCE(sum(quantity), 0) AS quantity FROM product_sizes "
                "WHERE product_id IN :ids AND size IS NOT NULL GROUP BY product_id, size ORDER BY product_id, min(id)"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": [product.id for product in products]},
        ).all()
        by_id = {product.id: product for product in products}
        for row in rows:
            by_id[row.product_id].sizes[row.size] = int(row.quantity)
