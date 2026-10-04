"""Chat actions on recorded sales: show one, change its payment, delivery,
quantity or size, and cancel it or change its status. Mixed into
SportsStoreActions. The store reverses and re-applies stock and day totals
itself when a sale changes, so these only send the new version of the sale."""

import asyncio

from app.actions.base import ActionFailedError, ActionProposal, ActionReply
from plugins.sports_store.intents import SaleStatusIntent, ShowSaleIntent, UpdateSaleIntent
from plugins.sports_store.store_api import error_detail

# The fields of the store's SaleCreate, which updateSale takes back in full.
_SALE_FIELDS = (
    "date", "total_quantity", "sub_total", "total_price", "payment_type_id", "payment_reference_number",
    "delivery_type_id", "shop_id", "customer_id", "status",
)
_ITEM_FIELDS = (
    "product_id", "product_name", "product_category", "size", "quantity_available", "quantity", "sale_price",
    "total_price",
)
_STATUS_VERBS = {
    "CANCELLED": "Cancel",
    "RETURNED": "Mark as returned",
    "COMPLETED": "Mark as completed",
    "SHIPPED": "Mark as shipped",
    "PENDING": "Mark as pending",
}


def _status(sale: dict) -> str:
    return str(sale["status"]).split(".")[-1].upper()


class SaleEditActions:
    # ---- show ------------------------------------------------------------------

    def _show_sale(self, intent: ShowSaleIntent) -> ActionReply:
        sale = self._repository.get_sale(intent.sale_id)
        if sale is None:
            return ActionReply(f"There's no sale #{intent.sale_id}.")
        return ActionReply("\n".join(self._sale_lines(sale)))

    # ---- status (cancel, return, ship...) ----------------------------------------

    def _propose_sale_status(self, intent: SaleStatusIntent) -> ActionReply:
        sale = self._find_sale(intent.sale_id)
        if isinstance(sale, ActionReply):
            return sale
        current = _status(sale)
        if current == intent.status:
            return ActionReply(f"Sale #{sale['id']} is already {current.lower()}.")

        lines = [line.replace("**", "") for line in self._sale_lines(sale)]
        if intent.status == "CANCELLED":
            lines.append("Stock goes back and the amount comes off that day's sales.")
        elif current == "CANCELLED":
            lines.append("The stock is taken again and the amount is added back to that day's sales.")
        lines.append(f"Status: {current.title()} → {intent.status.title()}")
        verb = _STATUS_VERBS[intent.status]
        payload = {"op": "sale_status", "sale_id": sale["id"], "status": intent.status}
        return ActionReply(
            f"{verb} sale #{sale['id']}? Check it's the right one and confirm.",
            ActionProposal(f"{verb} sale #{sale['id']}?", lines, f"{verb} sale", payload),
        )

    async def _change_sale_status(self, payload: dict) -> str:
        sale_id, status = payload["sale_id"], payload["status"]
        if status == "CANCELLED":
            response = await self._api.put(f"/sales/{sale_id}/cancel")
        else:
            response = await self._api.put(f"/sales/{sale_id}/updateStatus", {"status": status})
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't change sale #{sale_id} ({error_detail(response)}).")
        return f"Sale #{sale_id} is now {status.lower()}."

    # ---- edit ------------------------------------------------------------------

    async def _propose_sale_update(self, intent: UpdateSaleIntent) -> ActionReply:
        if not any((intent.quantity, intent.size, intent.payment, intent.delivery)):
            return ActionReply(
                "What should change on the sale? For example: “update sale #1521 payment upi”, "
                "“change sale #1521 qty 2”, “change sale #1521 size L” or “update sale #1521 express delivery”."
            )
        found = await asyncio.to_thread(self._find_sale, intent.sale_id)
        if isinstance(found, ActionReply):
            return found
        if _status(found) == "CANCELLED":
            return ActionReply(f"Sale #{found['id']} is cancelled. Restore it first (“mark sale #{found['id']} as completed”).")

        response = await self._api.get(f"/sales/{found['id']}")
        if response.status_code >= 400:
            return ActionReply(f"The store couldn't load sale #{found['id']} ({error_detail(response)}).")
        sale = response.json()
        return await asyncio.to_thread(self._build_sale_update, intent, sale)

    def _build_sale_update(self, intent: UpdateSaleIntent, sale: dict) -> ActionReply:
        body = {field: sale.get(field) for field in _SALE_FIELDS}
        body["sale_items"] = [{field: item.get(field) for field in _ITEM_FIELDS} for item in sale["sale_items"]]
        body["status"] = _status(sale)
        lines = [f"Sale #{sale['id']} · {sale['date']}"]
        stock_check = None

        if intent.payment:
            payment = self._lookup(self._repository.payment_types(), intent.payment)
            if payment is None:
                names = ", ".join(name for _, name in self._repository.payment_types())
                return ActionReply(f"I don't know the payment type “{intent.payment}”. Use one of: {names}.")
            old = next((name for id_, name in self._repository.payment_types() if id_ == sale["payment_type_id"]), "—")
            body["payment_type_id"] = payment[0]
            lines.append(f"Payment: {old} → {payment[1]}")

        if intent.delivery:
            deliveries = self._repository.delivery_types()
            delivery = self._lookup(deliveries, intent.delivery)
            if delivery is None:
                names = ", ".join(row[1] for row in deliveries)
                return ActionReply(f"I don't know the delivery type “{intent.delivery}”. Use one of: {names}.")
            old = next((row for row in deliveries if row[0] == sale["delivery_type_id"]), None)
            body["delivery_type_id"] = delivery[0]
            body["total_price"] += float(delivery[2]) - float(old[2] if old else 0)
            lines.append(f"Delivery: {old[1] if old else '—'} → {delivery[1]}")

        if intent.quantity is not None or intent.size:
            item = self._sale_item(body["sale_items"], intent.product)
            if isinstance(item, ActionReply):
                return item
            product = self._repository.get_product(item["product_id"], True)
            if product is None:
                return ActionReply(f"Product #{item['product_id']} no longer exists, so that item can't be changed.")
            quantity = intent.quantity if intent.quantity is not None else item["quantity"]
            if quantity < 1:
                return ActionReply(f"The quantity must be at least 1. To undo the whole sale: “cancel sale #{sale['id']}”.")
            size = item["size"]
            if intent.size:
                stock = product.stock_for(intent.size)
                if stock is None:
                    sizes = ", ".join(f"{name} ({qty})" for name, qty in product.sizes.items()) or "none"
                    return ActionReply(f"**{product.name}** has no size {intent.size}. Sizes: {sizes}.")
                size = stock[0]
            # The store puts the old quantity back before taking the new one.
            in_stock = (product.stock_for(size) or (size, 0))[1]
            available = in_stock + (item["quantity"] if size == item["size"] else 0)
            if quantity > available:
                return ActionReply(f"Only {available} of **{product.name}** available in size {size}.")

            old_total, old_quantity, old_size = item["total_price"], item["quantity"], item["size"]
            item.update(size=size, quantity=quantity, quantity_available=available, total_price=item["sale_price"] * quantity)
            body["total_price"] += item["total_price"] - old_total
            body["sub_total"] = (body["sub_total"] or 0) + product.selling_price * (quantity - old_quantity)
            body["total_quantity"] = sum(i["quantity"] for i in body["sale_items"])
            lines.append(f"{item['product_name']}: {old_quantity} × size {old_size} → {quantity} × size {size}")
            needed = quantity - (old_quantity if size == old_size else 0)
            if needed > 0:
                stock_check = {"product_id": product.id, "product_name": product.name, "size": size, "needed": needed}

        if body["total_price"] != sale["total_price"]:
            lines.append(f"Total: {self._money(sale['total_price'])} → {self._money(body['total_price'])}")
        payload = {"op": "update_sale", "sale_id": sale["id"], "sale": body, "stock_check": stock_check}
        return ActionReply(
            f"Ready to update sale #{sale['id']}. Check the changes and confirm.",
            ActionProposal(f"Update sale #{sale['id']}?", lines, "Update sale", payload),
        )

    async def _update_sale(self, payload: dict) -> str:
        check = payload.get("stock_check")
        if check:
            product = await asyncio.to_thread(self._repository.get_product, check["product_id"], True)
            stock = product.stock_for(check["size"]) if product else None
            if stock is None or stock[1] < check["needed"]:
                raise ActionFailedError(
                    f"Not enough {check['product_name']} left in size {check['size']} now, so the sale wasn't changed."
                )
        sale_id = payload["sale_id"]
        response = await self._api.post(f"/sales/{sale_id}/updateSale", payload["sale"])
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't update sale #{sale_id} ({error_detail(response)}).")
        return f"Sale #{sale_id} updated. Total now {self._money(payload['sale']['total_price'])}."

    # ---- helpers ---------------------------------------------------------------

    def _find_sale(self, sale_id: int | None) -> dict | ActionReply:
        if sale_id is None:
            sales = self._repository.last_sales(1, self._shop_id)
            return sales[0] if sales else ActionReply("No sales recorded yet.")
        return self._repository.get_sale(sale_id) or ActionReply(f"There's no sale #{sale_id}.")

    def _sale_item(self, items: list[dict], product: str | None) -> dict | ActionReply:
        if product:
            matching = [item for item in items if f"#{item['product_id']}" == product]
            if len(matching) == 1:
                return matching[0]
            if not matching:
                return ActionReply(f"Product {product} isn't in that sale.")
        elif len(items) == 1:
            return items[0]
        else:
            matching = items
        options = ", ".join(f"#{i['product_id']} {i['product_name']} ({i['quantity']} × size {i['size']})" for i in matching)
        return ActionReply(f"Which item? Name it by product number, e.g. “… #{matching[0]['product_id']} qty 2”. Items: {options}.")

    def _sale_lines(self, sale: dict) -> list[str]:
        status = _status(sale)
        who = sale["customer"] or "walk-in"
        lines = [
            f"**Sale #{sale['id']}** · {sale['date']} · {self._money(sale['total'])} · {sale['payment'] or '—'} · {who}"
            + ("" if status == "COMPLETED" else f" · {status.title()}")
        ]
        lines += [f"- {qty} × {name} (size {size})" for name, size, qty in sale["items"]]
        return lines
