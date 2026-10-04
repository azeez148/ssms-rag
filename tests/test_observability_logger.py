import json
import logging

from app.observability.events import ChatObservabilityEvent, RetrievedChunkLog
from app.observability.logger import log_chat_event


def _event(outcome: str = "answered", error: str | None = None) -> ChatObservabilityEvent:
    return ChatObservabilityEvent(
        question="Do you have Argentina jerseys?",
        knowledge_base_id="test-kb",
        retrieval_query="Do you have Argentina jerseys?",
        outcome=outcome,
        retrieval_latency_ms=12.3,
        rerank_latency_ms=0.0,
        generation_latency_ms=45.6,
        total_latency_ms=60.0,
        retrieved_chunks=[RetrievedChunkLog(document_id="products-41", chunk_index=0, similarity=0.67)],
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="stub",
        llm_model="stub-echo-v1",
        reranker_used=False,
        error=error,
    )


def test_log_chat_event_emits_valid_json(caplog):
    with caplog.at_level(logging.INFO, logger="rag.observability"):
        log_chat_event(_event())

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["question"] == "Do you have Argentina jerseys?"
    assert payload["retrieved_chunks"] == [{"document_id": "products-41", "chunk_index": 0, "similarity": 0.67}]


def test_log_chat_event_does_not_include_chunk_content():
    """Structured logs record chunk IDs and scores, never raw chunk content —
    the field simply doesn't exist on the event, so it can never leak."""
    event = _event()
    assert not hasattr(event.retrieved_chunks[0], "content")


def test_error_outcome_logs_at_error_level(caplog):
    with caplog.at_level(logging.INFO, logger="rag.observability"):
        log_chat_event(_event(outcome="error", error="boom"))

    assert caplog.records[0].levelno == logging.ERROR


def test_answered_outcome_logs_at_info_level(caplog):
    with caplog.at_level(logging.INFO, logger="rag.observability"):
        log_chat_event(_event(outcome="answered"))

    assert caplog.records[0].levelno == logging.INFO
