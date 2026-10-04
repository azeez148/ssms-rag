"""Turns a staff chat message into a shop request, deterministically.

Rules rather than the LLM: the parse decides what gets written to the till,
and it has to behave the same on a 1.5B local model, a hosted model, or none.
Anything that doesn't clearly look like a shop request returns None and is
answered by normal RAG instead.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SellIntent:
    product: str
    quantity: int = 1
    size: str | None = None
    customer_name: str | None = None
    mobile: str | None = None
    payment: str | None = None
    delivery: str | None = None


@dataclass
class StockIntent:
    product: str
    size: str | None = None


@dataclass
class SalesReportIntent:
    period: str  # today | yesterday | this_week | this_month | all_time


@dataclass
class HelpIntent:
    """A greeting or "what can you do?": answered with the capabilities, not RAG."""


@dataclass
class LastSalesIntent:
    count: int = 1


@dataclass
class UnsupportedIntent:
    """A shop command the chat can't carry out (create a customer, cancel a
    sale...). Answered with what it can do, instead of being sent to RAG,
    which would answer it with whatever product it retrieves."""

    request: str


@dataclass
class ListProductsIntent:
    query: str | None = None
    stock: str | None = None  # "in" | "out" | None (all)


@dataclass
class ListCategoriesIntent:
    pass


@dataclass
class CreateProductIntent:
    name: str
    # category, selling_price, unit_price, discounted_price, sizes, description
    fields: dict = field(default_factory=dict)


@dataclass
class UpdateProductIntent:
    product: str
    # name, category, selling_price, unit_price, discounted_price (None = remove
    # the offer), description, active (bool)
    changes: dict = field(default_factory=dict)


@dataclass
class StockChangeIntent:
    product: str
    mode: str  # add | remove | set
    # size -> quantity; the key None means the message named no size
    sizes: dict = field(default_factory=dict)


@dataclass
class ShowSaleIntent:
    sale_id: int


@dataclass
class UpdateSaleIntent:
    sale_id: int | None  # None = the last sale
    product: str | None = None  # "#41": which item, for a sale of several
    quantity: int | None = None
    size: str | None = None
    payment: str | None = None
    delivery: str | None = None


@dataclass
class SaleStatusIntent:
    sale_id: int | None  # None = the last sale
    status: str  # CANCELLED | COMPLETED | SHIPPED | RETURNED | PENDING


@dataclass
class StartDayIntent:
    opening_balance: float | None = None


@dataclass
class ExpenseIntent:
    amount: float | None
    description: str


@dataclass
class EndDayIntent:
    closing_balance: float | None = None
    reason: str | None = None


@dataclass
class DayStatusIntent:
    pass


Intent = (
    SellIntent | StockIntent | SalesReportIntent | LastSalesIntent | UnsupportedIntent | HelpIntent
    | ListProductsIntent | ListCategoriesIntent | CreateProductIntent | UpdateProductIntent | StockChangeIntent
    | ShowSaleIntent | UpdateSaleIntent | SaleStatusIntent
    | StartDayIntent | ExpenseIntent | EndDayIntent | DayStatusIntent
)

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUMBER = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_UNITS = r"(?:\s*(?:x|pcs|pc|pieces?|nos|units?|qty))?"

_SELL = re.compile(
    r"^\s*(?:please\s+)?(?:sell|sold|bill|(?:record|add|create|make|new)\s+(?:a\s+)?(?:new\s+)?sale\s+(?:of|for)?)\b\s*",
    re.I,
)
_LEADING_QTY = re.compile(rf"^{_NUMBER}{_UNITS}\b\s*(?:of\s+)?", re.I)
_TAGGED_QTY = re.compile(r"(?:\bqty\s*:?\s*|\bx\s*|×\s*)(\d{1,3})\b|\b(\d{1,3})\s*(?:pcs|pc|pieces?|nos|units?)\b", re.I)

_STOCK = re.compile(
    r"\b(how\s+many|in\s+stock|stock|quantity|qty|left|remaining|check\s+stock)\b", re.I
)
_PERIODS = {
    "today": re.compile(r"\btoday'?s?\b", re.I),
    "yesterday": re.compile(r"\byesterday'?s?\b", re.I),
    "this_week": re.compile(r"\bth(?:is|e)\s+week'?s?\b", re.I),
    "this_month": re.compile(r"\bth(?:is|e)\s+month'?s?\b", re.I),
    "all_time": re.compile(r"\b(total|overall|all[\s-]*time|ever|so\s+far|till\s+now|until\s+now)\b", re.I),
}
_REPORT_WORDS = re.compile(
    r"\b(sales?|sold|sell|revenue|business|collection|turnover|orders?|bills?|invoices?|transactions?)\b", re.I
)
_COUNT_QUESTION = re.compile(r"\b(how\s+many|how\s+much|number\s+of|count)\b", re.I)
_HELP = re.compile(
    r"^\s*(hi+|hello|hey|hai|good\s+(?:morning|afternoon|evening)|help|menu|thanks?|thank\s+you|ok(?:ay)?|"
    r"what\s+can\s+you\s+do|how\s+(?:do|can)\s+i\s+use\s+(?:this|you))\b[\s!.?,]*(?:there|bot|assistant)?[\s!.?]*$",
    re.I,
)

_LAST_SALES = re.compile(
    r"\b(?:last|latest|recent|previous|most\s+recent)\s+(?:(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s+)?"
    r"(sales?|orders?|bills?|transactions?|invoices?)\b",
    re.I,
)
_UNSUPPORTED = re.compile(
    r"\b(create|add|new|register|delete|remove|update|edit|change|modify|cancel|refund|return|void|discount)\b"
    r".*\b(customers?|orders?|sales?|bills?|products?|items?|stock|prices?|invoices?|payments?)\b",
    re.I,
)

_MOBILE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{9})(?!\d)")
_PAYMENT = re.compile(
    r"\b(?:(?:paid|pay(?:ment)?)\s+(?:by|via|in|with|through)?\s*|via\s+|by\s+|in\s+)?(cash|card|upi|gpay|google\s*pay|phonepe|paytm)\b",
    re.I,
)
_DELIVERY = re.compile(r"\b(pick\s*-?\s*up|standard\s+delivery|express(?:\s+delivery)?|dtdc|courier)\b", re.I)
_SIZE = re.compile(r"\b(?:size|sz)\s*:?\s*([a-z0-9][a-z0-9.]*)\b", re.I)
_TRAILING_SIZE = re.compile(r"(?:\s+|^)\(?(xxs|xs|s|m|l|xl|xxl|xxxl|[2-5]xl)\)?\s*$", re.I)
# Not "for": "sell 1 jersey for kids" names a product, not a customer.
_CUSTOMER = re.compile(r"\b(?:to|customer)\s*:?\s+([a-z][a-z.' ]{0,40}?)\s*(?:[,;]|$)", re.I)

_STOCK_FILLER = re.compile(
    r"\b(how\s+many|are|is|there|do|does|we|have|has|got|left|remaining|in\s+stock|stock|quantity|qty|"
    r"check|of|the|for|in|pieces?|pcs|units?|available|what|show|me|tell|please|any)\b",
    re.I,
)
_SELL_FILLER = re.compile(r"\b(of|the|please|pieces?|pcs|units?|nos)\b", re.I)


def _clean(text: str) -> str:
    text = re.sub(r"[?!]+", " ", text)
    text = re.sub(r"\s*[,;]\s*", " ", text)
    return " ".join(text.split()).strip(" -:")


def _names_something(product: str) -> bool:
    return bool(re.search(r"#\d+|[a-z0-9]{2,}", product, re.I))


def _take(pattern: re.Pattern, text: str) -> tuple[re.Match | None, str]:
    match = pattern.search(text)
    if match is None:
        return None, text
    return match, text[: match.start()] + " " + text[match.end() :]


def _size_from(text: str) -> tuple[str | None, str]:
    match, text = _take(_SIZE, text)
    if match:
        return match.group(1).rstrip(".").upper(), text
    match, text = _take(_TRAILING_SIZE, text.rstrip(" ?!.,"))
    return (match.group(1).upper(), text) if match else (None, text)


def _period(text: str) -> str | None:
    return next((name for name, pattern in _PERIODS.items() if pattern.search(text)), None)


def parse_intent(message: str) -> Intent | None:
    text = message.strip()
    if not text:
        return None

    if _HELP.match(text):
        return HelpIntent()

    for parse in (_parse_day, _parse_listing, _parse_sale_edit):
        intent = parse(text)
        if intent is not None:
            return intent

    sell = _SELL.match(text)
    if sell is None and _REPORT_WORDS.search(text):
        # "today's sales", "total orders", "how many orders?" (-> today)
        period = _period(text)
        if period or _COUNT_QUESTION.search(text):
            if not _LAST_SALES.search(text) and not _UNSUPPORTED.search(text):
                return SalesReportIntent(period or "today")

    if sell is not None:
        return _parse_sell(text[sell.end() :])

    intent = _parse_catalog(text)
    if intent is not None:
        return intent

    if _UNSUPPORTED.search(text):
        return UnsupportedIntent(text)

    last = _LAST_SALES.search(text)
    if last:
        # "last order" -> 1, "recent orders" -> 5, "last 3 sales" -> 3
        plural = last.group(2).lower().endswith("s")
        word = (last.group(1) or ("5" if plural else "1")).lower()
        return LastSalesIntent(min(_NUMBER_WORDS.get(word) or int(word), 10))

    if _STOCK.search(text):
        size, rest = _size_from(text)
        product = _clean(_STOCK_FILLER.sub(" ", rest))
        return StockIntent(product, size) if _names_something(product) else None
    return None


def _parse_sell(text: str) -> SellIntent | None:
    quantity = 1
    leading = _LEADING_QTY.match(text.strip())
    if leading:
        word = leading.group(1).lower()
        quantity = _NUMBER_WORDS.get(word) or int(word)
        text = text.strip()[leading.end() :]

    mobile, text = _take(_MOBILE, text)
    payment, text = _take(_PAYMENT, text)
    delivery, text = _take(_DELIVERY, text)
    customer, text = _take(_CUSTOMER, text)
    if not leading:
        tagged, text = _take(_TAGGED_QTY, text)
        if tagged:
            quantity = int(tagged.group(1) or tagged.group(2))
    size, text = _size_from(text)

    product = _clean(_SELL_FILLER.sub(" ", text))
    if not _names_something(product):
        return None

    payment_name = None
    if payment:
        payment_name = payment.group(1).lower().replace(" ", "")
        if payment_name in {"gpay", "googlepay", "phonepe", "paytm"}:
            payment_name = "upi"

    return SellIntent(
        product=product,
        quantity=quantity,
        size=size,
        customer_name=customer.group(1).strip().title() if customer else None,
        mobile=mobile.group(1) if mobile else None,
        payment=payment_name,
        delivery=re.sub(r"[\s-]+", " ", delivery.group(1).lower()) if delivery else None,
    )


# ---- amounts ------------------------------------------------------------------

_AMOUNT = re.compile(r"(?<![\w#.])(?:₹\s*|rs\.?\s*|inr\s*)?(\d[\d,]*(?:\.\d{1,2})?)(?![\w.])", re.I)


def _amount(text: str) -> tuple[float | None, str]:
    match, rest = _take(_AMOUNT, text)
    return (float(match.group(1).replace(",", "")), rest) if match else (None, text)


# ---- day management -------------------------------------------------------------

_DAY = r"(?:the\s+)?(?:day|shop|store|till|counter|register)"
_START_DAY = re.compile(rf"^\s*(?:please\s+)?(?:start|open|begin)\s+{_DAY}\b", re.I)
_END_DAY = re.compile(rf"^\s*(?:please\s+)?(?:end|close|finish)\s+{_DAY}\b", re.I)
_EXPENSE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:add|record|log|new|enter)\s+(?:an?\s+)?(?:new\s+)?expenses?|expense|spent)\b\s*:?", re.I
)
_DAY_STATUS = re.compile(
    r"\b(?:day\s+(?:status|summary|report|details)|cash\s+in\s+hand|is\s+the\s+(?:day|shop|store)\s+"
    r"(?:started|open|running)|(?:today'?s?\s+)?expenses(?:\s+today)?)\b",
    re.I,
)
_REASON = re.compile(r"\b(?:reason|because|note|remarks?)\b\s*:?\s*(.+)$", re.I)
_EXPENSE_LEAD = re.compile(r"^\s*(?:of|for|on|towards|to|as|-|:)\s+", re.I)
_BALANCE_FILLER = re.compile(
    r"\b(with|opening|closing|balance|cash|counted|in|hand|amount|of|is|at|the|drawer|float|and)\b", re.I
)


def _parse_day(text: str) -> Intent | None:
    if _START_DAY.match(text):
        amount, _ = _amount(text[_START_DAY.match(text).end() :])
        return StartDayIntent(amount)

    if _END_DAY.match(text):
        rest = text[_END_DAY.match(text).end() :]
        reason, rest = _take(_REASON, rest)
        amount, _ = _amount(rest)
        return EndDayIntent(amount, reason.group(1).strip(" .") if reason else None)

    expense = _EXPENSE.match(text)
    if expense:
        amount, rest = _amount(text[expense.end() :])
        description = _clean(_EXPENSE_LEAD.sub("", " " + _clean(rest) + " ", count=1))
        if amount is None and not description:
            return DayStatusIntent() if re.search(r"expenses", text, re.I) else ExpenseIntent(None, "")
        return ExpenseIntent(amount, description)

    if _DAY_STATUS.search(text) and not _REPORT_WORDS.search(text):
        return DayStatusIntent()
    return None


# ---- listings -------------------------------------------------------------------

_LIST_CATEGORIES = re.compile(
    r"^\s*(?:please\s+)?(?:list|show(?:\s+me)?|what\s+are)\s+(?:all\s+)?(?:the\s+)?(?:product\s+)?categor(?:y|ies)\b",
    re.I,
)
_LIST_PRODUCTS = re.compile(
    r"^\s*(?:please\s+)?(?:list|show(?:\s+me)?|display|get)\s+(?:all\s+|every\s+)?(?:the\s+)?"
    r"(?:(?:out[\s-]*of[\s-]*stock|in[\s-]*stock|available|sold[\s-]*out)\s+)?(?:products?|items?)\b",
    re.I,
)
_LIST_ANY = re.compile(r"^\s*(?:please\s+)?list\s+(?:all\s+)?(?:the\s+)?", re.I)
_STOCK_FILTER = re.compile(r"\b(?:(out[\s-]*of[\s-]*stock|sold[\s-]*out)|(in[\s-]*stock|available))\b", re.I)
_LIST_FILLER = re.compile(
    r"\b(in|of|category|categories|matching|named|called|with|like|the|all|products?|items?|please|list|show|me)\b",
    re.I,
)


def _parse_listing(text: str) -> Intent | None:
    if _LIST_CATEGORIES.match(text):
        return ListCategoriesIntent()

    listing = _LIST_PRODUCTS.match(text)
    if listing is None:
        listing = _LIST_ANY.match(text)
        if listing is None or _REPORT_WORDS.search(text) or re.search(r"expenses", text, re.I):
            return None
    stock_filter, _ = _take(_STOCK_FILTER, text)
    stock = None if stock_filter is None else ("out" if stock_filter.group(1) else "in")
    rest = _STOCK_FILTER.sub(" ", text[listing.end() :])
    query = _clean(_LIST_FILLER.sub(" ", rest))
    return ListProductsIntent(query or None, stock)


# ---- sales: show, edit, cancel ----------------------------------------------------

_SALE_WORD = r"(?:sale|order|bill|invoice)"
_SALE_REF = re.compile(
    rf"\b{_SALE_WORD}\s*(?:no\.?|number|id)?\s*:?\s*#?\s*(\d+)\b|\b(?:the\s+)?(?:last|latest|previous)\s+{_SALE_WORD}\b",
    re.I,
)
_SHOW_SALE = re.compile(
    r"^\s*(?:please\s+)?(?:show|view|get|open|check|find)?\s*(?:me\s+)?(?:the\s+)?(?:details?\s+(?:of|for)\s+)?"
    rf"{_SALE_WORD}\s*(?:no\.?|number|id)?\s*:?\s*#?\s*(\d+)(?:\s+details?)?\s*\??\s*$",
    re.I,
)
_CANCEL_SALE = re.compile(r"^\s*(?:please\s+)?(cancel|void|return)\b", re.I)
_EDIT_SALE = re.compile(r"^\s*(?:please\s+)?(?:update|edit|change|modify|correct|fix|mark|set)\b", re.I)
_STATUS = re.compile(r"\b(completed?|shipped|returned|pending|cancell?ed)\b", re.I)
_SALE_QTY = re.compile(
    r"\b(?:qty|quantity)\s*(?:to|=|:|as)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:pcs|pc|pieces?|units?|nos)\b", re.I
)
_ITEM = re.compile(r"#\s*(\d+)")


def _sale_id(match: re.Match) -> int | None:
    return int(match.group(1)) if match.group(1) else None


def _parse_sale_edit(text: str) -> Intent | None:
    show = _SHOW_SALE.match(text)
    if show:
        return ShowSaleIntent(int(show.group(1)))

    ref = _SALE_REF.search(text)
    if ref is None:
        return None

    cancel = _CANCEL_SALE.match(text)
    if cancel:
        return SaleStatusIntent(_sale_id(ref), "RETURNED" if cancel.group(1).lower() == "return" else "CANCELLED")

    if not _EDIT_SALE.match(text):
        return None
    rest = text[: ref.start()] + " " + text[ref.end() :]
    status = _STATUS.search(rest)
    if status and (re.search(r"\b(status|mark|as)\b", rest, re.I)):
        word = status.group(1).upper()
        return SaleStatusIntent(
            _sale_id(ref), {"COMPLETE": "COMPLETED", "CANCELED": "CANCELLED"}.get(word, word)
        )

    payment, rest = _take(_PAYMENT, rest)
    delivery, rest = _take(_DELIVERY, rest)
    size, rest = _take(_SIZE, rest)
    quantity, rest = _take(_SALE_QTY, rest)
    item, rest = _take(_ITEM, rest)
    payment_name = None
    if payment:
        payment_name = payment.group(1).lower().replace(" ", "")
        if payment_name in {"gpay", "googlepay", "phonepe", "paytm"}:
            payment_name = "upi"
    return UpdateSaleIntent(
        sale_id=_sale_id(ref),
        product=f"#{item.group(1)}" if item else None,
        quantity=int(quantity.group(1) or quantity.group(2)) if quantity else None,
        size=size.group(1).rstrip(".").upper() if size else None,
        payment=payment_name,
        delivery=re.sub(r"[\s-]+", " ", delivery.group(1).lower()) if delivery else None,
    )


# ---- products: create, edit, stock ---------------------------------------------------

_CREATE_PRODUCT = re.compile(
    r"^\s*(?:please\s+)?(?:create|add|new|register)\s+(?:a\s+)?(?:new\s+)?product\b\s*:?", re.I
)
_UPDATE_PRODUCT = re.compile(
    r"^\s*(?:please\s+)?(?:update|edit|change|modify|set)\s+(?:the\s+)?(?:product\s+)?", re.I
)
_RENAME = re.compile(r"^\s*(?:please\s+)?rename\s+(?:the\s+)?(?:product\s+)?(.+?)\s+(?:to|as)\s+(.+?)\s*$", re.I)
_ACTIVATE = re.compile(
    r"^\s*(?:please\s+)?(activate|enable|unhide|deactivate|disable|hide)\s+(?:the\s+)?(?:product\s+)?(.+?)\s*$", re.I
)
# Order matters: the multi-word prices come before plain "price".
_FIELDS = {
    "unit_price": r"cost(?:\s+price)?|unit\s+price|purchase\s+price|buying\s+price",
    "discounted_price": r"offer(?:\s+price)?|discount(?:ed)?(?:\s+price)?|sale\s+price",
    "selling_price": r"selling\s+price|price|mrp",
    "category": r"category|cat",
    "sizes": r"sizes?|stock",
    "name": r"name",
    "description": r"description|desc",
}
_FIELD = re.compile(
    r"\b(?:" + "|".join(f"(?P<{name}>{pattern})" for name, pattern in _FIELDS.items()) + r")\b\s*(?:is|to|=|:|as)?\s*",
    re.I,
)
# "price of #41 to 799": field first, product after.
_FIELD_OF = re.compile(
    r"^(?P<field>" + "|".join(_FIELDS[name] for name in ("unit_price", "discounted_price", "selling_price", "category", "name", "description"))
    + r")\s+(?:of|for)\s+(?P<product>.+?)(?:\s+(?:to|=|as)\s+(?P<value>.+?))?\s*$",
    re.I,
)
_NO_VALUE = re.compile(r"^\s*(?:none|no|nil|remove|removed|clear|off|0)\s*$", re.I)
_SIZE_PAIRS = re.compile(
    r"(?<![\w#])([a-z]{1,4}|\d{1,2}(?:\.\d)?)\s*[:=]\s*(\d{1,4})\b|(?<![\w#])([a-z]{1,4})\s+(\d{1,4})\b", re.I
)

_STOCK_CHANGE = re.compile(
    r"^\s*(?:please\s+)?(?:(?P<verb>add|restock|increase|reduce|remove|decrease|deduct)"
    r"(?P<noun>\s+(?:to\s+)?(?:stock|qty|quantity|inventory))?"
    r"|(?P<set>set|update|change|adjust|correct)\s+(?:the\s+)?(?:stock|qty|quantity|inventory))\b\s*(?:of|for)?\s*",
    re.I,
)
_STOCK_QTY = re.compile(r"\b(?:to|by)\s+(\d{1,4})\b", re.I)
_STOCK_EDIT_FILLER = re.compile(r"\b(of|the|please|pieces?|pcs|units?|nos|to|by|from|more|stock|qty|quantity)\b", re.I)


def _sizes(text: str) -> dict[str, int]:
    return {
        (match.group(1) or match.group(3)).upper(): int(match.group(2) or match.group(4))
        for match in _SIZE_PAIRS.finditer(text)
    }


def _field_name(keyword: str) -> str:
    return next(name for name, pattern in _FIELDS.items() if re.fullmatch(pattern, keyword, re.I))


def _field_value(name: str, raw: str):
    raw = raw.strip(" ,;.")
    if name == "discounted_price" and _NO_VALUE.match(raw):
        return None
    if name in {"unit_price", "selling_price", "discounted_price"}:
        amount, _ = _amount(raw)
        return None if amount is None else int(round(amount))
    if name == "sizes":
        return _sizes(raw)
    return _clean(raw)


def _fields(text: str) -> tuple[str, dict]:
    """Splits "Argentina Home price 799 cost 400 sizes S:2 M:3" into the part
    before the first field word and {field: value}. A number-valued field with
    no number is left out, so the handler can ask for it."""
    matches = list(_FIELD.finditer(text))
    if not matches:
        return _clean(text), {}
    head = text[: matches[0].start()]
    fields: dict = {}
    for match, following in zip(matches, matches[1:] + [None]):
        name = match.lastgroup
        raw = text[match.end() : following.start() if following else len(text)]
        value = _field_value(name, raw)
        if value is None and name != "discounted_price":
            continue
        if name == "discounted_price" and value is None and not _NO_VALUE.match(raw.strip(" ,;.")):
            continue
        fields[name] = value
    return _clean(head), fields


def _parse_catalog(text: str) -> Intent | None:
    create = _CREATE_PRODUCT.match(text)
    if create:
        name, fields = _fields(text[create.end() :])
        return CreateProductIntent(name, fields)

    intent = _parse_stock_change(text)
    if intent is not None:
        return intent

    rename = _RENAME.match(text)
    if rename:
        return UpdateProductIntent(_clean(rename.group(1)), {"name": _clean(rename.group(2))})
    activate = _ACTIVATE.match(text)
    if activate:
        active = activate.group(1).lower() in {"activate", "enable", "unhide"}
        return UpdateProductIntent(_clean(activate.group(2)), {"active": active})

    update = _UPDATE_PRODUCT.match(text)
    if update is None:
        return None
    rest = text[update.end() :]
    field_of = _FIELD_OF.match(rest)
    if field_of:
        name = _field_name(field_of.group("field"))
        changes = {}
        if field_of.group("value"):
            value = _field_value(name, field_of.group("value"))
            if value is not None or name == "discounted_price":
                changes[name] = value
        return UpdateProductIntent(_clean(field_of.group("product")), changes)
    product, changes = _fields(rest)
    if not changes or not _names_something(product):
        return None
    return UpdateProductIntent(product, changes)


def _parse_stock_change(text: str) -> StockChangeIntent | None:
    change = _STOCK_CHANGE.match(text)
    if change is None:
        return None
    if change.group("set"):
        mode = "set"
    else:
        mode = "add" if change.group("verb").lower() in {"add", "restock", "increase"} else "remove"
    rest = text[change.end() :]

    pairs = {
        (m.group(1)).upper(): int(m.group(2))
        for m in re.finditer(r"(?<![\w#])([a-z]{1,4}|\d{1,2}(?:\.\d)?)\s*[:=]\s*(\d{1,4})\b", rest, re.I)
    }
    if pairs:
        rest = re.sub(r"(?<![\w#])([a-z]{1,4}|\d{1,2}(?:\.\d)?)\s*[:=]\s*(\d{1,4})\b", " ", rest, flags=re.I)
        rest = re.sub(r"\bsizes?\b", " ", rest, flags=re.I)
        product = _clean(_STOCK_EDIT_FILLER.sub(" ", rest))
        return StockChangeIntent(product, mode, pairs) if _names_something(product) else None

    quantity = None
    leading = _LEADING_QTY.match(rest.strip())
    if leading:
        word = leading.group(1).lower()
        quantity = _NUMBER_WORDS.get(word) or int(word)
        rest = rest.strip()[leading.end() :]
    else:
        tagged, rest = _take(_STOCK_QTY, rest)
        if tagged is None:
            tagged, rest = _take(_TAGGED_QTY, rest)
            if tagged:
                quantity = int(tagged.group(1) or tagged.group(2))
        else:
            quantity = int(tagged.group(1))
    size, rest = _size_from(rest)
    product = _clean(_STOCK_EDIT_FILLER.sub(" ", rest))
    if not _names_something(product):
        return None
    if quantity is None:
        # "add #41" isn't clearly about stock; "set stock of #41" is, and gets asked for the quantity.
        return StockChangeIntent(product, mode, {}) if change.group("noun") or change.group("set") else None
    return StockChangeIntent(product, mode, {size: quantity})
