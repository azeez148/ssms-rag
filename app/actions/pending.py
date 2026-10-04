import secrets
import threading
import time
from dataclasses import dataclass

from app.actions.base import ActionProposal

DEFAULT_TTL_SECONDS = 600


@dataclass
class PendingAction:
    id: str
    knowledge_base_id: str
    proposal: ActionProposal
    expires_at: float


class PendingActionStore:
    """Proposals awaiting confirmation, keyed by an unguessable id. Each one
    can be confirmed or cancelled once, and expires after `ttl_seconds` so a
    stale proposal (prices, stock) can't be confirmed hours later.

    In-process memory: fine for one API process. Several workers or replicas
    would need a shared store (e.g. a DB table) instead.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, clock=time.time):
        self._ttl = ttl_seconds
        self._clock = clock
        self._items: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def put(self, knowledge_base_id: str, proposal: ActionProposal) -> PendingAction:
        pending = PendingAction(secrets.token_urlsafe(16), knowledge_base_id, proposal, self._clock() + self._ttl)
        with self._lock:
            self._drop_expired()
            self._items[pending.id] = pending
        return pending

    def pop(self, action_id: str) -> PendingAction | None:
        """Removes and returns the action, or None if unknown or expired."""
        with self._lock:
            self._drop_expired()
            return self._items.pop(action_id, None)

    def _drop_expired(self) -> None:
        now = self._clock()
        for key in [key for key, item in self._items.items() if item.expires_at <= now]:
            del self._items[key]


pending_actions = PendingActionStore()
