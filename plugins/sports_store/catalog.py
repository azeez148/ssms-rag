"""Product chat actions: list products and categories, create a product, edit
one, and change its stock. Mixed into SportsStoreActions, whose repository,
store API and formatting helpers these use."""

import asyncio

from app.actions.base import ActionFailedError, ActionProposal, ActionReply
from plugins.sports_store.intents import (
    CreateProductIntent,
    ListProductsIntent,
    StockChangeIntent,
    UpdateProductIntent,
)
from plugins.sports_store.repository import Product
from plugins.sports_store.store_api import error_detail

LIST_LIMIT = 20

CREATE_EXAMPLE = "“add product Argentina Home 26 category Jersey price 799 cost 400 sizes S:2 M:5 L:3”"
_FIELD_LABELS = {
    "name": "Name",
    "category": "Category",
    "selling_price": "Price (MRP)",
    "discounted_price": "Offer price",
    "unit_price": "Cost",
    "description": "Description",
    "active": "Active",
}


class CatalogActions:
    # ---- read ----------------------------------------------------------------

    def _list_products(self, intent: ListProductsIntent) -> ActionReply:
        products, total = self._repository.list_products(intent.query, intent.stock, LIST_LIMIT)
        kind = {"in": "in-stock ", "out": "out-of-stock ", None: ""}[intent.stock]
        matching = f" matching “{intent.query}”" if intent.query else ""
        if not products:
            return ActionReply(f"No {kind}products{matching}.")

        lines = [f"**{total} {kind}product{'' if total == 1 else 's'}{matching}**", ""]
        for product in products:
            lines.append(f"- **#{product.id}** {product.name} ({product.category}) · {self._money(product.sale_price)} · {self._in_stock(product)}")
        if total > len(products):
            lines += ["", f"_Newest {len(products)} of {total} shown. Add words to narrow it, e.g. “list products argentina”._"]
        return ActionReply("\n".join(lines))

    def _list_categories(self) -> ActionReply:
        categories = self._repository.categories()
        return ActionReply("**Categories**\n\n" + "\n".join(f"- {name}" for _, name in categories))

    # ---- create ----------------------------------------------------------------

    def _propose_create_product(self, intent: CreateProductIntent) -> ActionReply:
        fields = intent.fields
        missing = [
            label
            for label, present in (
                ("name", intent.name),
                ("category", fields.get("category")),
                ("price", fields.get("selling_price")),
                ("cost", fields.get("unit_price")),
            )
            if not present
        ]
        if missing:
            return ActionReply(f"To add a product I still need its {', '.join(missing)}. For example: {CREATE_EXAMPLE}.")

        category = self._category(fields["category"])
        if isinstance(category, ActionReply):
            return category
        price, cost, offer = fields["selling_price"], fields["unit_price"], fields.get("discounted_price")
        problem = self._price_problem(price, cost, offer)
        if problem:
            return ActionReply(problem)

        sizes = fields.get("sizes") or {}
        body = {
            "name": intent.name,
            "description": fields.get("description"),
            "unit_price": cost,
            "selling_price": price,
            "discounted_price": offer,
            "category_id": category[0],
            # Active and listed, so it can be sold straight away (from the chat too).
            "is_active": True,
            "can_listed": True,
            "size_map": [{"size": size, "quantity": quantity} for size, quantity in sizes.items()],
            "shop_ids": [self._shop_id],
        }

        lines = [
            f"Name: {intent.name}",
            f"Category: {category[1]}",
            f"Price (MRP): {self._money(price)}" + (f" · offer {self._money(offer)}" if offer else ""),
            f"Cost: {self._money(cost)}",
            "Stock: " + (", ".join(f"{size}: {quantity}" for size, quantity in sizes.items()) or "none yet"),
        ]
        if fields.get("description"):
            lines.append(f"Description: {fields['description']}")
        same_name = [p for p in self._repository.find_products(intent.name)[0] if p.name.strip().lower() == intent.name.strip().lower()]
        if same_name:
            lines.append(f"Note: #{same_name[0].id} already has this name.")
        return ActionReply(
            f"Ready to add **{intent.name}**. Check the details and confirm.",
            ActionProposal("Add this product?", lines, "Add product", {"op": "create_product", "product": body}),
        )

    async def _create_product(self, payload: dict) -> str:
        response = await self._api.post("/products/addProduct", payload["product"])
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't add the product ({error_detail(response)}).")
        product = response.json()
        return f"Product #{product.get('id')} **{product.get('name')}** added."

    # ---- edit ------------------------------------------------------------------

    def _propose_update_product(self, intent: UpdateProductIntent) -> ActionReply:
        changes = dict(intent.changes)
        if "sizes" in changes:
            sizes = changes.pop("sizes")
            if changes:
                return ActionReply("Change the stock on its own, e.g. “set stock of #41 M=5 L=2”, then the other details.")
            return self._propose_stock_change(StockChangeIntent(intent.product, "set", sizes))

        product = self._product(intent.product, include_inactive=True)
        if isinstance(product, ActionReply):
            return product
        if not changes:
            return ActionReply(
                f"What should change on **{product.name}**? For example: “update #{product.id} price 799”, "
                f"“update #{product.id} offer 699”, “update #{product.id} cost 400”, "
                f"“change #{product.id} category to Jersey”, “rename #{product.id} to …” or “deactivate #{product.id}”."
            )

        current = self._repository.product_fields(product.id)
        body = {key: current[key] for key in (
            "id", "name", "description", "unit_price", "selling_price", "discounted_price",
            "category_id", "is_active", "can_listed",
        )}
        lines = [f"Product: #{product.id} {current['name']}"]
        for field, value in changes.items():
            if field == "category":
                category = self._category(value)
                if isinstance(category, ActionReply):
                    return category
                body["category_id"] = category[0]
                lines.append(f"Category: {current['category']} → {category[1]}")
            elif field == "active":
                body["is_active"] = body["can_listed"] = value
                lines.append(f"Active: {'yes' if current['is_active'] and current['can_listed'] else 'no'} → {'yes' if value else 'no'}")
            elif field in {"selling_price", "unit_price", "discounted_price"}:
                body[field] = value
                lines.append(f"{_FIELD_LABELS[field]}: {self._price_or_none(current[field])} → {self._price_or_none(value)}")
            else:
                body[field] = value
                lines.append(f"{_FIELD_LABELS[field]}: {current[field] or '—'} → {value}")

        problem = self._price_problem(body["selling_price"], body["unit_price"], body["discounted_price"])
        if problem:
            return ActionReply(problem)
        return ActionReply(
            f"Ready to update **{current['name']}**. Check the changes and confirm.",
            ActionProposal("Update this product?", lines, "Update product", {"op": "update_product", "product": body}),
        )

    async def _update_product(self, payload: dict) -> str:
        response = await self._api.post("/products/updateProduct", payload["product"])
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't update the product ({error_detail(response)}).")
        product = response.json()
        return f"Product #{product.get('id')} **{product.get('name')}** updated."

    # ---- stock -----------------------------------------------------------------

    def _propose_stock_change(self, intent: StockChangeIntent) -> ActionReply:
        product = self._product(intent.product, include_inactive=True)
        if isinstance(product, ActionReply):
            return product
        verb = {"add": "add", "remove": "remove", "set": "set"}[intent.mode]
        if not intent.sizes:
            return ActionReply(
                f"How many? For example: “add 5 #{product.id} size M”, “remove 1 #{product.id} size M” "
                f"or “set stock of #{product.id} M=5 L=2”."
            )

        changes: dict[str, int] = {}
        for size, quantity in intent.sizes.items():
            if size is None:
                if len(product.sizes) != 1:
                    sizes = ", ".join(f"{name} ({qty})" for name, qty in product.sizes.items()) or "none yet"
                    return ActionReply(f"Which size of **{product.name}** should I {verb}? Sizes: {sizes}.")
                size = next(iter(product.sizes))
            existing = product.stock_for(size)
            if existing is None and intent.mode == "remove":
                return ActionReply(f"**{product.name}** has no size {size}.")
            changes[existing[0] if existing else size] = quantity

        new_sizes = _apply_stock(product.sizes, intent.mode, changes)
        if isinstance(new_sizes, str):
            return ActionReply(new_sizes)

        lines = [f"Product: #{product.id} {product.name}"]
        for size in changes:
            before = product.sizes.get(size)
            lines.append(f"Size {size}: {'new' if before is None else before} → {new_sizes[size]}")
        lines.append(f"Total stock: {sum(product.sizes.values())} → {sum(new_sizes.values())}")
        payload = {"op": "stock", "product_id": product.id, "product_name": product.name, "mode": intent.mode, "sizes": changes}
        return ActionReply(
            f"Ready to update the stock of **{product.name}**. Check and confirm.",
            ActionProposal("Update stock?", lines, "Update stock", payload),
        )

    async def _change_stock(self, payload: dict) -> str:
        # Apply the change to the stock as it is now, not as it was when proposed:
        # sales may have happened in between, and the store replaces the whole size map.
        product = await asyncio.to_thread(self._repository.get_product, payload["product_id"], True)
        if product is None:
            raise ActionFailedError(f"Product #{payload['product_id']} no longer exists. Nothing was changed.")
        new_sizes = _apply_stock(product.sizes, payload["mode"], payload["sizes"])
        if isinstance(new_sizes, str):
            raise ActionFailedError(new_sizes + " Nothing was changed.")

        response = await self._api.post("/products/updateSizeMap", {"product_id": product.id, "size_map": new_sizes})
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't update the stock ({error_detail(response)}).")
        changed = ", ".join(f"{size}: {product.sizes.get(size, 0)} → {new_sizes[size]}" for size in payload["sizes"])
        return f"Stock updated for **{product.name}** ({changed})."

    # ---- helpers ---------------------------------------------------------------

    def _product(self, query: str, include_inactive: bool = False) -> Product | ActionReply:
        """The one product `query` names ("#41" may be inactive; names only find active ones)."""
        if include_inactive and query.strip().startswith("#") and query.strip()[1:].isdigit():
            product = self._repository.get_product(int(query.strip()[1:]), True)
            return product or ActionReply(f"There's no product {query.strip()}.")
        products, exact = self._repository.find_products(query)
        return self._single(products, exact, query, example="update {id} price 799")

    def _category(self, wanted: str) -> tuple[int, str] | ActionReply:
        categories = self._repository.categories()
        wanted_key = wanted.strip().lower()
        exact = [row for row in categories if row[1].strip().lower() == wanted_key]
        if exact:
            return exact[0]
        partial = [row for row in categories if wanted_key in row[1].lower()]
        if len(partial) == 1:
            return partial[0]
        options = partial or categories
        return ActionReply(
            f"Which category did you mean by “{wanted}”? "
            + ", ".join(name for _, name in options)
            + "."
        )

    def _price_problem(self, price: int, cost: int, offer: int | None) -> str | None:
        if price is None or price <= 0 or cost is None or cost <= 0:
            return "Price and cost must both be more than 0."
        if offer is not None and not 0 < offer <= price:
            return f"The offer price must be between 1 and the price ({self._money(price)})."
        return None

    def _price_or_none(self, amount) -> str:
        return "none" if amount is None else self._money(amount)

    @staticmethod
    def _in_stock(product: Product) -> str:
        return ", ".join(f"{size} ({qty})" for size, qty in product.sizes.items() if qty > 0) or "out of stock"


def _apply_stock(sizes: dict[str, int], mode: str, changes: dict[str, int]) -> dict[str, int] | str:
    """The full size map after the change, or why it can't be made."""
    new_sizes = dict(sizes)
    for size, quantity in changes.items():
        current = new_sizes.get(size, 0)
        if mode == "add":
            new_sizes[size] = current + quantity
        elif mode == "set":
            new_sizes[size] = quantity
        elif quantity > current:
            return f"Only {current} in size {size}, so I can't remove {quantity}."
        else:
            new_sizes[size] = current - quantity
    return new_sizes
