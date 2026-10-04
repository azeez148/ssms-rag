from dataclasses import dataclass

from app.evaluation.dataset import EvalQuestion
from app.evaluation.metrics import reciprocal_rank, recall_at_k
from app.retrieval.base import Retriever


@dataclass
class EvalReport:
    num_questions: int
    recall_at: dict[int, float]
    mrr: float
    num_excluded_no_ground_truth: int = 0


async def evaluate_retrieval(
    retriever: Retriever,
    knowledge_base_id: str,
    dataset: list[EvalQuestion],
    k_values: list[int] = [1, 3, 5],
) -> EvalReport:
    """Recall/MRR are only meaningful for questions that have a real correct
    document. Questions with an empty expected_document_ids list (e.g. the
    "unsupported" category — genuinely out-of-domain questions) have no
    ground truth to recall, so they're excluded from these metrics rather
    than silently scored as 0 — that would understate retrieval quality for
    a reason that has nothing to do with retrieval. They're still evaluated
    separately by answer-appropriateness metrics in run_rag_eval.py.
    """
    max_k = max(k_values)
    recall_hits = {k: 0.0 for k in k_values}
    reciprocal_ranks = []

    scoreable = [item for item in dataset if item.expected_document_ids]
    excluded = len(dataset) - len(scoreable)

    for item in scoreable:
        results = await retriever.retrieve(item.question, knowledge_base_id, top_k=max_k)
        retrieved_ids = [r.document_id for r in results]
        expected_ids = set(item.expected_document_ids)

        for k in k_values:
            recall_hits[k] += recall_at_k(retrieved_ids, expected_ids, k)
        reciprocal_ranks.append(reciprocal_rank(retrieved_ids, expected_ids))

    num_questions = len(scoreable)
    return EvalReport(
        num_questions=num_questions,
        recall_at={k: hits / num_questions for k, hits in recall_hits.items()},
        mrr=sum(reciprocal_ranks) / num_questions,
        num_excluded_no_ground_truth=excluded,
    )
