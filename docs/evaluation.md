# Evaluation

## What is it?

Measuring whether the system is good and whether a change made it better. Two layers:

1. **Retrieval evaluation** ([app/evaluation/](../app/evaluation/), `scripts/run_retrieval_eval.py`) —
   Recall@K and MRR against DB-verified ground truth. No LLM involved.
2. **End-to-end RAG evaluation** (`scripts/run_rag_eval.py`) — the full `RAGService` pipeline over 52
   categorized questions, run twice (rerank off / on). Output:
   `evaluation/results/rag_evaluation_report.{md,json}`; dataset in
   `evaluation/datasets/sports_store_questions.json`.

## Why did we need it?

Eyeballing a few chat replies can't tell you if a change helped, and a fluent LLM answer can hide a retrieval
failure. Evaluating retrieval on its own isolates the cause.

## How did we implement it?

- **Recall@K**: 1 if any expected document is in the top K, averaged. **MRR**: mean of 1/rank of the first
  correct hit — reveals ranking problems Recall@K hides.
- Questions carry `expected_document_ids` (a list — many questions have several valid answers).
- `unsupported` questions are **excluded** from Recall/MRR (no correct document exists by design).
- Categories: exact-match, semantic, simple, ambiguous, follow-up, multi-document, unsupported — so an
  aggregate can be broken down.
- RAG-level metrics: answer appropriateness (answered vs. declined correctly), groundedness (cited at least
  one source), citation correctness (the *cited* source is a correct document), latency, estimated tokens/cost.
- `InMemoryRetriever` lets chunk-config comparisons run without writing to the real index.

## What alternatives existed?

LLM-as-judge (RAGAS, TruLens, DeepEval), human labeling, click/feedback logs from production, precision@K /
nDCG, synthetic question generation from chunks.

## Why did we choose this approach?

Deterministic, free, and reproducible without an LLM key. The trade-off is that several metrics are **proxies**.

## Results (real runs)

| Metric (52 Qs; 46 scored for retrieval) | No rerank | Rerank |
|---|---|---|
| Recall@1 | 0.565 | 0.652 |
| Recall@3 | 0.696 | 0.696 |
| Recall@5 | 0.804 | 0.739 |
| MRR | 0.652 | 0.681 |
| Avg total latency | 13.7 ms | 136.7 ms |

By category (no rerank, Recall@5): exact-match **1.0**, follow-up **1.0**, multi-document 0.75,
simple 0.667, ambiguous 0.667, **semantic 0.333**. The aggregate 0.804 hides that paraphrase questions are the
weak spot — the actionable finding.

## What problems did we encounter?

- **The LLM is a stub**, so: appropriateness and groundedness are both **1.0** — they're structural proxies
  (right answer/decline decision; "cited something"), *not* evidence the answers are good. Do not quote those
  as answer quality.
- **Citation correctness ≡ Recall@1** here (0.565 / 0.652) because the stub always cites `[1]`. True by
  construction; it only becomes an independent metric with a real LLM.
- **Ground-truth sampling** for `ambiguous`/`simple` (20 of 53 products) can under-count correct results.
- Earlier 22-question set is a different dataset from the 52-question one — don't compare numbers across them
  (e.g. Recall@1 0.682 vs 0.565).
- Latency excludes any real LLM call.

## How did we test it?

`tests/test_evaluation_metrics.py`, `test_evaluation_runner.py`, `test_rag_metrics.py` cover the metric
functions and runner logic with fakes; the reports come from live runs against real ingested Sports Store data.

## What would we change at scale?

Wire a real LLM and add an LLM-as-judge (with human spot checks) for faithfulness and answer relevance;
exhaustive ground truth for broad categories; a hybrid-friendly ID/SKU set; run evaluation in CI with a
regression threshold; sample real production queries from `query_logs` into the dataset; report confidence
intervals (46 questions is small — a 0.09 Recall@1 difference is a handful of questions).

## How would I explain it in an interview?

"I evaluate retrieval separately from generation using Recall@K and MRR on questions whose correct documents I
verified in the database, and I break results down by category. Reranking lifted Recall@1 from 0.565 to 0.652,
and the breakdown showed paraphrase questions are the real weakness (Recall@5 0.333). I'm careful about what
the numbers mean: my LLM is a stub, so groundedness and appropriateness are proxies, and citation correctness
just equals Recall@1. The next step is an LLM judge, not more retrieval metrics."
