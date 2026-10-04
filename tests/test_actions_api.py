import pytest
from fastapi.testclient import TestClient

from app.actions import registry
from app.actions.base import ActionFailedError, ActionHandler, ActionProposal, ActionReply
from app.actions.pending import PendingActionStore
from app.api.chat import get_rag_service
from app.main import app
from app.services.rag_service import ChatResult


class FakeHandler(ActionHandler):
    def __init__(self, config=None, options=None):
        self.executed = []
        self.fail_with = None

    async def handle(self, question, history):
        if question.startswith("sell"):
            return ActionReply("Ready.", ActionProposal("Record this sale?", ["Total: 10"], "Confirm sale", {"n": 1}))
        if question.startswith("stock"):
            return ActionReply("3 left.")
        if question.startswith("boom"):
            raise RuntimeError("db down")
        return None

    async def execute(self, payload):
        if self.fail_with:
            raise self.fail_with
        self.executed.append(payload)
        return "Sale #7 recorded."


class FakeRAG:
    async def answer(self, question, knowledge_base_id, **kwargs):
        return ChatResult(answer="from rag", sources=[], retrieval_query=question)


@pytest.fixture
def handler(monkeypatch):
    fake = FakeHandler()
    monkeypatch.setattr("app.api.chat.get_action_handler", lambda kb, config_dir: fake if kb == "shop" else None)
    monkeypatch.setattr("app.api.actions.get_action_handler", lambda kb, config_dir: fake if kb == "shop" else None)
    monkeypatch.setattr("app.api.chat.pending_actions", store := PendingActionStore())
    monkeypatch.setattr("app.api.actions.pending_actions", store)
    app.dependency_overrides[get_rag_service] = FakeRAG
    yield fake
    app.dependency_overrides.clear()


def _chat(question, kb="shop"):
    return TestClient(app).post("/api/chat", json={"knowledge_base": kb, "question": question})


def test_action_question_returns_a_proposal_without_executing(handler):
    body = _chat("sell 1 x").json()

    assert body["answer"] == "Ready."
    assert body["action"]["title"] == "Record this sale?"
    assert body["action"]["lines"] == ["Total: 10"]
    assert "payload" not in body["action"]  # stays server-side
    assert handler.executed == []


def test_confirm_executes_once(handler):
    action_id = _chat("sell 1 x").json()["action"]["id"]
    client = TestClient(app)

    first = client.post(f"/api/actions/{action_id}/confirm")
    second = client.post(f"/api/actions/{action_id}/confirm")

    assert first.status_code == 200
    assert first.json() == {"status": "done", "message": "Sale #7 recorded."}
    assert second.status_code == 404
    assert handler.executed == [{"n": 1}]


def test_cancel_discards_the_proposal(handler):
    action_id = _chat("sell 1 x").json()["action"]["id"]
    client = TestClient(app)

    assert client.post(f"/api/actions/{action_id}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/actions/{action_id}/confirm").status_code == 404
    assert handler.executed == []


def test_failed_action_returns_its_message(handler):
    handler.fail_with = ActionFailedError("Only 1 left now")
    action_id = _chat("sell 1 x").json()["action"]["id"]

    response = TestClient(app).post(f"/api/actions/{action_id}/confirm")

    assert response.status_code == 409
    assert response.json()["detail"] == "Only 1 left now"


def test_unknown_action_id_is_404(handler):
    assert TestClient(app).post("/api/actions/nope/confirm").status_code == 404


def test_non_action_questions_and_other_knowledge_bases_use_rag(handler):
    assert _chat("Do you have jerseys?").json()["answer"] == "from rag"
    assert _chat("sell 1 x", kb="library").json()["answer"] == "from rag"


def test_informational_action_has_no_proposal(handler):
    body = _chat("stock of x").json()
    assert body["answer"] == "3 left."
    assert body["action"] is None


def test_handler_crash_is_a_friendly_message_not_a_500(handler):
    response = _chat("boom")
    assert response.status_code == 200
    assert "couldn't reach the live store data" in response.json()["answer"]


def test_pending_actions_expire():
    now = [1000.0]
    store = PendingActionStore(ttl_seconds=60, clock=lambda: now[0])
    pending = store.put("shop", ActionProposal("t", []))

    now[0] += 61

    assert store.pop(pending.id) is None


def test_registry_loads_the_handler_named_by_the_source_config(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_handlers", {})
    (tmp_path / "shop.yaml").write_text(
        "source:\n"
        "  type: postgres\n  name: shop\n  knowledge_base_id: shop\n"
        "  connection: {env: X}\n"
        "  entities: [{name: e, query: 'select 1', id_column: id, content_template: x}]\n"
        "  actions:\n    handler: tests.test_actions_api:FakeHandler\n    options: {a: 1}\n"
    )

    assert isinstance(registry.get_action_handler("shop", tmp_path), FakeHandler)
    assert registry.get_action_handler("other", tmp_path) is None


def test_registry_ignores_a_handler_that_is_not_an_action_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_handlers", {})
    (tmp_path / "shop.yaml").write_text(
        "source:\n"
        "  type: postgres\n  name: shop\n  knowledge_base_id: shop\n"
        "  connection: {env: X}\n"
        "  entities: [{name: e, query: 'select 1', id_column: id, content_template: x}]\n"
        "  actions: {handler: 'tests.test_actions_api:FakeRAG'}\n"
    )

    assert registry.get_action_handler("shop", tmp_path) is None
