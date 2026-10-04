import re

from app.generation.base import LLMProvider
from app.generation.prompts import SYSTEM_PROMPT

INSUFFICIENT_INFO = "There is insufficient information in the knowledge base to answer that question."

_MAX_ITEMS = 4
_MAX_VALUE_CHARS = 220
_BLOCK_HEADER = re.compile(r"(?m)^\[(\d+)\] (.*)$")
# A field label: a capitalised word, optionally followed by up to two lowercase
# words ("Price", "Available sizes"), then a colon. Content templates are often
# folded onto one line ("Product: X Price: 600 Available sizes: M, L"), so labels
# are found inline rather than assuming one field per line.
_FIELD_LABEL = re.compile(r"(?:^|(?<=\s))([A-Z][A-Za-z]*(?: [a-z]+){0,2}):\s+")
_WORD = re.compile(r"[a-z]+")


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _tokens(text: str) -> set[str]:
    return {_stem(word) for word in _WORD.findall(text.lower())}


class ExtractiveLLMProvider(LLMProvider):
    """A no-API-key, no-cost "answer engine" that writes a readable reply by
    laying out the retrieved passages, rather than generating new text.

    It has no domain logic: passages are the numbered `[n] Title` blocks that
    `build_context()` produces, and any `Key: value` line inside one is treated
    as a field. If the question mentions a field name ("price", "sizes"), only
    those fields are shown, so "Which sizes are available?" answers with sizes
    instead of the whole record. Every item carries its `[n]` citation.

    It only answers for the RAG answer prompt. The query rewriter and broadener
    also call an LLMProvider; for them this returns "" (no usable rewrite), which
    their existing fallbacks handle (history concatenation / dropping a leading
    interrogative).
    """

    async def generate(self, system_prompt: str, user_prompt: str, context: str) -> str:
        if system_prompt != SYSTEM_PROMPT:
            return ""

        blocks = self._parse_blocks(context)
        if not blocks:
            return INSUFFICIENT_INFO

        wanted = _tokens(user_prompt)

        # Each item is (index, title, rows to show). Rows are restricted to the
        # fields the question names when any passage has such a field — and
        # passages lacking them are dropped, rather than padded out with
        # unrelated details.
        items = []
        for index, title, fields, notes in blocks:
            candidates = [(k, v) for k, v in fields if v.strip().lower() != title.strip().lower()]
            relevant = [(k, v) for k, v in candidates if _tokens(k) & wanted]
            fallback = candidates + ([("", " ".join(notes))] if notes else [])
            items.append((index, title, relevant, fallback))

        if any(relevant for _, _, relevant, _ in items):
            items = [(i, t, relevant, relevant) for i, t, relevant, _ in items if relevant]
        shown = items[:_MAX_ITEMS]

        lines = ["Here's what I found in the knowledge base:", ""]
        for index, title, _, rows in shown:
            lines.append(f"**{title}** [{index}]")
            for key, value in rows:
                value = _truncate(value)
                lines.append(f"- {key}: {value}" if key else f"- {value}")
            lines.append("")

        if len(items) > len(shown):
            lines.append(f"_Showing the top {len(shown)} of {len(items)} matches._")
        return "\n".join(lines).strip()

    @staticmethod
    def _parse_blocks(context: str) -> list[tuple[int, str, list[tuple[str, str]], list[str]]]:
        headers = list(_BLOCK_HEADER.finditer(context))
        blocks = []
        for position, header in enumerate(headers):
            end = headers[position + 1].start() if position + 1 < len(headers) else len(context)
            body = context[header.end() : end]

            fields: list[tuple[str, str]] = []
            notes: list[str] = []
            labels = list(_FIELD_LABEL.finditer(body))
            preamble = body[: labels[0].start()] if labels else body
            if preamble.strip():
                notes.append(" ".join(preamble.split()))
            for position_in_body, label in enumerate(labels):
                value_end = labels[position_in_body + 1].start() if position_in_body + 1 < len(labels) else len(body)
                value = " ".join(body[label.end() : value_end].split())
                if value:
                    fields.append((label.group(1), value))
            blocks.append((int(header.group(1)), header.group(2).strip(), fields, notes))
        return blocks


def _truncate(value: str) -> str:
    value = " ".join(value.split())
    return value if len(value) <= _MAX_VALUE_CHARS else value[: _MAX_VALUE_CHARS - 1].rstrip() + "…"
