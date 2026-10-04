"""Day management from the chat: start the day with an opening balance, add
expenses, see the day's status, and end the day with the counted cash. Mixed
into SportsStoreActions. The store computes the day's totals (cash vs account
sales, expected cash in hand); these endpoints need a store login."""

from datetime import datetime

from app.actions.base import ActionFailedError, ActionProposal, ActionReply
from plugins.sports_store.intents import EndDayIntent, ExpenseIntent, StartDayIntent
from plugins.sports_store.store_api import error_detail

START_EXAMPLE = "“start day with 2000”"


def _time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value


class DayActions:
    async def _today_status(self) -> dict | None:
        """This shop's entry from the store's today status: {day_started,
        day_ended, active_day}, or None if the store doesn't list the shop."""
        response = await self._api.get("/day-management/today", auth=True)
        if response.status_code >= 400:
            raise ActionFailedError(f"The store couldn't give the day's status ({error_detail(response)}).")
        shops = response.json()
        return next((shop for shop in shops if shop.get("shop_id") == self._shop_id), None)

    async def _active_day(self) -> dict | ActionReply:
        today = await self._today_status()
        if today and today["day_started"] and not today["day_ended"]:
            return today["active_day"]
        if today and today["day_ended"]:
            return ActionReply(f"Today's day has already been ended. Reopen it with {START_EXAMPLE} first.")
        return ActionReply(f"The day hasn't been started yet. Start it with {START_EXAMPLE} (your opening cash).")

    # ---- status ----------------------------------------------------------------

    async def _day_status(self) -> ActionReply:
        today = await self._today_status()
        if not today or not today["day_started"]:
            return ActionReply(f"The day hasn't been started yet. Start it with {START_EXAMPLE} (your opening cash).")
        day = today["active_day"]
        state = f"ended at {_time(day.get('end_time'))}" if today["day_ended"] else "open"
        lines = [f"**Day #{day['id']}** · started {_time(day['start_time'])} · {state}", ""]
        lines += self._day_lines(day)
        if today["day_ended"] and day.get("closing_balance") is not None:
            lines.append(f"- Counted at close: {self._money(day['closing_balance'])} (variance {self._signed(day.get('variance') or 0)})")
        expenses = day.get("expenses") or []
        if expenses:
            lines += ["", "Expenses:"] + [f"- {e['description']}: {self._money(e['amount'])}" for e in expenses]
        return ActionReply("\n".join(lines))

    # ---- start -----------------------------------------------------------------

    async def _propose_start_day(self, intent: StartDayIntent) -> ActionReply:
        today = await self._today_status()
        if today and today["day_started"] and not today["day_ended"]:
            day = today["active_day"]
            return ActionReply(
                f"The day is already open (started {_time(day['start_time'])}, "
                f"opening balance {self._money(day['opening_balance'])})."
            )
        if today and today["day_ended"]:
            day = today["active_day"]
            lines = [
                f"Day #{day['id']} was ended at {_time(day.get('end_time'))}.",
                f"It reopens with its opening balance of {self._money(day['opening_balance'])}.",
            ]
            payload = {"op": "start_day", "body": {"opening_balance": day["opening_balance"], "shop_id": self._shop_id}}
            return ActionReply(
                "Today's day was already ended. Reopen it?",
                ActionProposal("Reopen today's day?", lines, "Reopen day", payload),
            )
        if intent.opening_balance is None:
            return ActionReply(f"How much cash is in the drawer to start with? For example: {START_EXAMPLE}.")

        lines = [f"Opening balance: {self._money(intent.opening_balance)}", f"Shop: #{self._shop_id}"]
        payload = {"op": "start_day", "body": {"opening_balance": intent.opening_balance, "shop_id": self._shop_id}}
        return ActionReply(
            "Ready to start the day. Confirm the opening cash.",
            ActionProposal("Start the day?", lines, "Start day", payload),
        )

    async def _start_day(self, payload: dict) -> str:
        response = await self._api.post("/day-management/startDay", payload["body"], auth=True)
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't start the day ({error_detail(response)}).")
        day = response.json()
        return f"Day #{day['id']} is open. Opening balance {self._money(day['opening_balance'])}."

    # ---- expenses --------------------------------------------------------------

    async def _propose_expense(self, intent: ExpenseIntent) -> ActionReply:
        if intent.amount is None or intent.amount <= 0 or not intent.description:
            return ActionReply("What was it for, and how much? For example: “add expense 150 for tea”.")
        day = await self._active_day()
        if isinstance(day, ActionReply):
            return day

        spent = float(day.get("total_expense") or 0)
        lines = [
            f"Expense: {intent.description}",
            f"Amount: {self._money(intent.amount)}",
            f"Day's expenses: {self._money(spent)} → {self._money(spent + intent.amount)}",
            f"Expected cash in hand after: {self._money(float(day.get('cash_in_hand') or 0) - intent.amount)}",
        ]
        body = {"description": intent.description, "amount": intent.amount, "day_id": day["id"], "shop_id": self._shop_id}
        return ActionReply(
            "Ready to add this expense. Check and confirm.",
            ActionProposal("Add this expense?", lines, "Add expense", {"op": "add_expense", "body": body}),
        )

    async def _add_expense(self, payload: dict) -> str:
        response = await self._api.post("/day-management/addExpense", payload["body"], auth=True)
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't add the expense ({error_detail(response)}).")
        body = payload["body"]
        return f"Expense added: {body['description']}, {self._money(body['amount'])}."

    # ---- end -------------------------------------------------------------------

    async def _propose_end_day(self, intent: EndDayIntent) -> ActionReply:
        day = await self._active_day()
        if isinstance(day, ActionReply):
            return day
        expected = float(day.get("cash_in_hand") or 0)
        if intent.closing_balance is None:
            lines = self._day_lines(day)
            return ActionReply(
                "\n".join(
                    [f"Expected cash in the drawer: **{self._money(expected)}**.", ""]
                    + lines
                    + ["", "Count the cash and tell me, e.g. “end day with " + f"{expected:.0f}”."]
                )
            )

        variance = intent.closing_balance - expected
        lines = [line.lstrip("- ") for line in self._day_lines(day)]
        lines += [f"Counted cash: {self._money(intent.closing_balance)}", f"Variance: {self._signed(variance)}"]
        if intent.reason:
            lines.append(f"Reason: {intent.reason}")
        elif round(variance, 2):
            lines.append("No reason given for the variance (add one with “reason: …”).")
        payload = {
            "op": "end_day",
            "day_id": day["id"],
            "body": {"closing_balance": intent.closing_balance, "variance_reason": intent.reason},
        }
        return ActionReply(
            "Ready to end the day. Check the totals and confirm.",
            ActionProposal("End the day?", lines, "End day", payload),
        )

    async def _end_day(self, payload: dict) -> str:
        response = await self._api.post(f"/day-management/endDay/{payload['day_id']}", payload["body"], auth=True)
        if response.status_code >= 400:
            raise ActionFailedError(f"The store didn't end the day ({error_detail(response)}).")
        summary = response.json()
        return (
            f"Day #{summary['day_id']} ended. Sales {self._money(summary['total_sales'])}, "
            f"expenses {self._money(summary['total_expense'])}, expected cash {self._money(summary['cash_in_hand'])}, "
            f"counted {self._money(summary.get('closing_balance') or 0)} (variance {self._signed(summary.get('variance') or 0)})."
        )

    # ---- helpers ---------------------------------------------------------------

    def _day_lines(self, day: dict) -> list[str]:
        return [
            f"- Opening balance: {self._money(day['opening_balance'])}",
            f"- Sales: {self._money(day.get('total_sales') or 0)} "
            f"(cash {self._money(day.get('total_cash_sales') or 0)}, account {self._money(day.get('total_account_sales') or 0)})",
            f"- Expenses: {self._money(day.get('total_expense') or 0)}",
            f"- Expected cash in hand: {self._money(day.get('cash_in_hand') or 0)}",
        ]

    def _signed(self, amount: float) -> str:
        if not round(amount, 2):
            return self._money(0)
        return ("+" if amount > 0 else "−") + self._money(abs(amount))
