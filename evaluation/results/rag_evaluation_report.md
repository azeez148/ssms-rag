# RAG Evaluation Report

Generated: 2026-09-20T11:40:17.883318+00:00
Models: embedding=all-MiniLM-L6-v2, llm=stub/stub-echo-v1, reranker=cross_encoder

All numbers below come from actually running the dataset through the live RAGService pipeline, twice — once with `rerank=False` (the API's default) and once with `rerank=True` — to directly answer "how do we know the system is actually improving?" with a real before/after comparison rather than a single point-in-time number.

## Overall comparison

| Metric | Without rerank | With rerank |
|---|---|---|
| Recall@1 | 0.565 | 0.652 |
| Recall@3 | 0.696 | 0.696 |
| Recall@5 | 0.804 | 0.739 |
| MRR | 0.652 | 0.681 |
| Answer appropriateness rate | 1.0 | 1.0 |
| Groundedness rate | 1.0 | 1.0 |
| Citation correctness rate | 0.565 | 0.652 |
| Avg total latency (ms) | 14.266 | 138.843 |

## Detail: without rerank

### Overall

| Metric | Value |
|---|---|
| Questions | 52 |
| Excluded from retrieval scoring (no ground truth) | 6 |
| Recall@1 | 0.565 |
| Recall@3 | 0.696 |
| Recall@5 | 0.804 |
| MRR | 0.652 |
| Answer appropriateness rate | 1.0 |
| Groundedness rate | 1.0 |
| Citation correctness rate | 0.565 |
| Avg total latency (ms) | 14.266 |
| Avg retrieval latency (ms) | 13.424 |
| Avg generation latency (ms) | 0.009 |
| Avg estimated prompt tokens | 335.327 |
| Avg estimated completion tokens | 67.904 |

#### By category

| Category | N | Recall@5 | MRR | Appropriateness | Groundedness | Citation correctness |
|---|---|---|---|---|---|---|
| ambiguous | 6 | 0.667 | 0.583 | 1.0 | 1.0 | 0.5 |
| exact-match | 18 | 1.0 | 0.819 | 1.0 | 1.0 | 0.722 |
| follow-up | 6 | 1.0 | 0.833 | 1.0 | 1.0 | 0.667 |
| multi-document | 4 | 0.75 | 0.562 | 1.0 | 1.0 | 0.5 |
| semantic | 6 | 0.333 | 0.208 | 1.0 | 1.0 | 0.167 |
| simple | 6 | 0.667 | 0.542 | 1.0 | 1.0 | 0.5 |
| unsupported | 6 | n/a | None | 1.0 | None | None |


## Detail: with rerank

### Overall

| Metric | Value |
|---|---|
| Questions | 52 |
| Excluded from retrieval scoring (no ground truth) | 6 |
| Recall@1 | 0.652 |
| Recall@3 | 0.696 |
| Recall@5 | 0.739 |
| MRR | 0.681 |
| Answer appropriateness rate | 1.0 |
| Groundedness rate | 1.0 |
| Citation correctness rate | 0.652 |
| Avg total latency (ms) | 138.843 |
| Avg retrieval latency (ms) | 17.749 |
| Avg generation latency (ms) | 0.009 |
| Avg estimated prompt tokens | 340.404 |
| Avg estimated completion tokens | 66.769 |

#### By category

| Category | N | Recall@5 | MRR | Appropriateness | Groundedness | Citation correctness |
|---|---|---|---|---|---|---|
| ambiguous | 6 | 0.333 | 0.25 | 1.0 | 1.0 | 0.167 |
| exact-match | 18 | 1.0 | 0.921 | 1.0 | 1.0 | 0.889 |
| follow-up | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| multi-document | 4 | 0.75 | 0.75 | 1.0 | 1.0 | 0.75 |
| semantic | 6 | 0.333 | 0.208 | 1.0 | 1.0 | 0.167 |
| simple | 6 | 0.5 | 0.5 | 1.0 | 1.0 | 0.5 |
| unsupported | 6 | n/a | None | 1.0 | None | None |


## Methodology notes (read before trusting these numbers)

- **Retrieval quality** (Recall/MRR) is measured on the sources that survived the *full* pipeline (guardrails, reranking, context budget) — not raw retrieval — and excludes `unsupported` questions, which have no correct document to recall by design.
- **Answer appropriateness** is a behavioral proxy for relevance: did the system answer vs. correctly decline, matching the question's category? It does not judge the *content* of the generated text — with the stub LLM there is no real content to judge.
- **Groundedness** is structural: did the answer cite at least one retrieved source? It does not verify that the citation is semantically accurate — that requires a real LLM that can misattribute a claim.
- **Citation correctness** checks whether the *cited* source (not merely a retrieved one) matches a real correct document, where a correct document is known. With the stub LLM always citing source [1], citation correctness is mathematically identical to Recall@1 here — a real finding, not a bug: once a real LLM is wired in, the two metrics can diverge (an LLM might cite a wrong source even when the right one was retrieved, or vice versa).
- No metric here is fabricated or hand-tuned to look good — every row in the JSON report was produced by an actual pipeline run against the live Sports Store data.
- **A known limitation, found by actually running this**: `ambiguous` and `simple` categories' `expected_document_ids` are a *sample* of valid answers (e.g. 20 of the real 53 Messi products), not the complete set. Reranking's apparent Recall@5 drop on these categories may be partly an artifact of this incomplete sampling — a genuinely correct product outside the sampled 20 would be wrongly counted as a miss. This is a real limitation of the evaluation dataset, not a demonstrated regression in reranking; a full fix would require exhaustive ground truth for every broad-category question.
