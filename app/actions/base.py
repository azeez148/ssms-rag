from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas.chat import ConversationTurn


@dataclass
class ActionProposal:
    """A write the handler wants to make, shown to the user for confirmation.
    Nothing is written until the user confirms it."""

    title: str
    lines: list[str]
    confirm_label: str = "Confirm"
    # Opaque to the RAG core. Kept server-side and handed back to
    # ActionHandler.execute() on confirmation, so the client can't alter it.
    payload: dict = field(default_factory=dict)


@dataclass
class ActionReply:
    answer: str
    proposal: ActionProposal | None = None


class ActionFailedError(RuntimeError):
    """The confirmed action could not be completed; the message is shown to the user."""


class ActionHandler(ABC):
    """Domain-specific chat actions for one knowledge base ("sell 2 of X",
    "how many X are left?"), wired in through the source config's `actions:`
    block. The RAG core never knows what the actions mean."""

    @abstractmethod
    async def handle(self, question: str, history: list[ConversationTurn]) -> ActionReply | None:
        """Answer `question` if it's an action request; None hands it to normal RAG."""

    @abstractmethod
    async def execute(self, payload: dict) -> str:
        """Perform a proposal the user confirmed. Returns the message to show,
        or raises ActionFailedError."""
