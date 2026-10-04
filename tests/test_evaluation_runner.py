import asyncio

from app.evaluation.dataset import EvalQuestion
from app.evaluation.runner import evaluate_retrieval
from app.retrieval.base import Retriever
from app.schemas.retrieval import RetrievedChunk


class StubRetriever(Retriever):
    """Returns a fixed, hand-crafted ranking per query, keyed by exact question text."""

    def __init__(self, canned_results: dict[str, list[str]]):
        self._canned_results = canned_results

    async def retrieve(
        self, query: str, knowledge_base_id: str, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        ids = self._canned_results.get(query, [])[:top_k]
        return [
            RetrievedChunk(
                content="irrelevant",
                document_id=doc_id,
                chunk_index=0,
                document_title=None,
                source="stub",
                knowledge_base_id=knowledge_base_id,
                metadata={},
                similarity=1.0 - 0.1 * i,
                distance=0.1 * i,
            )
            for i, doc_id in enumerate(ids)
        ]


def test_evaluate_retrieval_perfect_retriever_scores_max():
    dataset = [
        EvalQuestion(category="simple", question="q1", expected_document_ids=["a"]),
        EvalQuestion(category="simple", question="q2", expected_document_ids=["b"]),
    ]
    retriever = StubRetriever({"q1": ["a", "x", "y"], "q2": ["b", "x", "y"]})

    report = asyncio.run(evaluate_retrieval(retriever, "test-kb", dataset, k_values=[1, 3, 5]))

    assert report.num_questions == 2
    assert report.recall_at[1] == 1.0
    assert report.recall_at[3] == 1.0
    assert report.mrr == 1.0


def test_evaluate_retrieval_distinguishes_recall_at_1_from_recall_at_3():
    dataset = [EvalQuestion(category="simple", question="q1", expected_document_ids=["a"])]
    # Correct answer is ranked 3rd, not 1st.
    retriever = StubRetriever({"q1": ["x", "y", "a"]})

    report = asyncio.run(evaluate_retrieval(retriever, "test-kb", dataset, k_values=[1, 3]))

    assert report.recall_at[1] == 0.0
    assert report.recall_at[3] == 1.0
    assert report.mrr == 1.0 / 3


def test_evaluate_retrieval_total_miss_scores_zero():
    dataset = [EvalQuestion(category="simple", question="q1", expected_document_ids=["a"])]
    retriever = StubRetriever({"q1": ["x", "y", "z"]})

    report = asyncio.run(evaluate_retrieval(retriever, "test-kb", dataset, k_values=[1, 3, 5]))

    assert report.recall_at[1] == 0.0
    assert report.recall_at[5] == 0.0
    assert report.mrr == 0.0


def test_evaluate_retrieval_averages_across_multiple_questions():
    dataset = [
        EvalQuestion(category="simple", question="hit", expected_document_ids=["a"]),
        EvalQuestion(category="simple", question="miss", expected_document_ids=["a"]),
    ]
    retriever = StubRetriever({"hit": ["a"], "miss": ["z"]})

    report = asyncio.run(evaluate_retrieval(retriever, "test-kb", dataset, k_values=[1]))

    assert report.recall_at[1] == 0.5
    assert report.mrr == 0.5


def test_questions_with_no_ground_truth_are_excluded_not_scored_zero():
    """An 'unsupported' question with no valid document has no ground truth
    to recall — it must not silently drag Recall/MRR down as if retrieval
    had failed."""
    dataset = [
        EvalQuestion(category="simple", question="hit", expected_document_ids=["a"]),
        EvalQuestion(category="unsupported", question="out of domain", expected_document_ids=[]),
    ]
    retriever = StubRetriever({"hit": ["a"]})

    report = asyncio.run(evaluate_retrieval(retriever, "test-kb", dataset, k_values=[1]))

    assert report.num_questions == 1
    assert report.num_excluded_no_ground_truth == 1
    assert report.recall_at[1] == 1.0
    assert report.mrr == 1.0
