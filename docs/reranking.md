# Reranking

## What is it?

A second, more expensive pass that re-scores a small candidate set. `CrossEncoderReranker`
([app/reranking/](../app/reranking/)) scores each `(query, chunk)` pair *jointly* with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. `RerankingRetriever` wraps any `Retriever`, over-fetches
`top_k × RERANK_CANDIDATE_MULTIPLIER` (default 4 → 20 candidates for top 5), reranks, and truncates.
Set `RERANKER_PROVIDER=none` for a no-op reranker.

## Why did we need it?

Bi-encoder vector search embeds query and chunk separately, so it is cheap but coarse. Measured: the right
document was usually in the pool but not first (Recall@5 0.955 vs Recall@1 0.682).

## How did we implement it?

Retrieve wide (cheap) → rerank narrow (expensive) → return top K. Enabled per request with `rerank: true`
(default `false`). The cross-encoder's logit **replaces** `similarity` (and sets `distance = −score`) on the
returned chunks, because it's on a different scale from cosine.

## What alternatives existed?

LLM-as-reranker (flexible, but a full LLM call per rerank — slower, costlier, and a new failure mode:
malformed or hallucinated ordering), Cohere/Voyage rerank APIs, ColBERT-style late interaction, no reranking.

## Why did we choose this approach?

A purpose-built cross-encoder is cheap enough to run on 20 candidates, free/local, and needs no API key.

## What problems did we encounter?

- **Unbounded scores.** The UI multiplied cross-encoder logits by 100 and showed "449.4%" and negative
  percentages. Fixed by displaying the raw score instead of forcing incomparable scales onto one display.
- The weak-retrieval threshold (0.35) assumes cosine similarity, so it runs on the retriever's **pre-rerank**
  candidates, not reranked scores.
- **A confusing result we investigated rather than hid**: in the 52-question evaluation Recall@5 fell
  0.804 → 0.739 with reranking even though Recall@1 rose. We traced part of it to the dataset: the
  `ambiguous`/`simple` questions' ground truth samples only 20 of 53 real Messi products, so a correct reranked
  result outside the sample is scored as a miss. That is a documented dataset limitation, not a proven
  regression — but I have *not* re-run with exhaustive ground truth to confirm that's the whole story.
- **Latency:** average total latency rose from ~13.7 ms to ~136.7 ms per request (~10×), with the stub LLM
  (so real-world latency is dominated by the LLM anyway).

## How did we test it?

`tests/test_reranking.py`, `test_reranking_retriever.py` (wrapper behavior, over-fetch, truncation). Results:

| Eval | Metric | No rerank | Rerank |
|---|---|---|---|
| 22 Qs (Plan Set 15) | Recall@1 | 0.682 | **0.864** |
| | MRR | 0.772 | **0.890** |
| | Recall@5 | 0.955 | 0.955 (unchanged — reordering can't recover what isn't in the pool) |
| 52 Qs (Plan Set 19) | Recall@1 | 0.565 | **0.652** |
| | MRR | 0.652 | **0.681** |
| | Recall@5 | **0.804** | 0.739 |

## What would we change at scale?

Batch/GPU inference or a hosted rerank API; cache rerank results for hot queries; raise candidate pool size
only if Recall@20 shows misses; A/B the reranker per query type (it helped `exact-match`/`follow-up`, not
`ambiguous`/`semantic`); make reranking the default only after fixing the ground-truth sampling.

## How would I explain it in an interview?

"Reranking splits the job in two: the retriever's goal is not to miss the answer, the reranker's goal is to
put it first. It improved Recall@1 and MRR in both of my evaluations, while Recall@5 stayed flat on one and
dropped on the other, because reordering can't add documents. I traced the drop partly to incomplete ground
truth in my dataset, and I say that's likely-but-unconfirmed. The cost is roughly 10× latency on my setup."
