# Lessons Learned

Each item is something that actually happened in this project, not general advice.

## 1. Measure the output against the invariant you care about
The hand-written chunker had a token-budget overflow (7 of 12 chunks over the limit at chunk_size=500) that
survived 21 plan sets because no test checked real chunk token counts. **Lesson:** test the property, not
just "it returns chunks". → [chunking.md](chunking.md)

## 2. A null result is still a result
Chunk size 300/500/800 gave identical retrieval metrics because every product is one chunk. Hybrid tied
vector on the eval set. Reporting "this experiment can't tell us anything on this corpus" is more useful than
picking a winner. → [chunking.md](chunking.md), [hybrid-search.md](hybrid-search.md)

## 3. Evaluate retrieval separately from generation
`/api/search` never calls an LLM, so a retrieval failure can't hide behind fluent prose. Recall@5 0.955 with
Recall@1 0.682 said "the answer is in the pool; ranking is the problem" — which pointed straight at
reranking. → [evaluation.md](evaluation.md)

## 4. Aggregates hide the interesting failure
Overall Recall@5 was 0.804; by category, `semantic` questions scored 0.333 and `exact-match` scored 1.0. The
breakdown, not the average, told me what to fix. → [evaluation.md](evaluation.md)

## 5. Know what a metric is a proxy for
With a stub LLM, groundedness and appropriateness are 1.0 and citation correctness equals Recall@1 by
construction. Those numbers describe the pipeline's structure, not answer quality. I labelled them as proxies
in the report rather than presenting 1.0 as a result.

## 6. Investigate a surprising number instead of explaining it away
Reranking lowered Recall@5 (0.804 → 0.739). I traced part of it to incomplete ground truth (20 of 53 valid
products sampled). I documented that as a limitation, but I did not re-run with exhaustive labels, so it is
likely, not proven.

## 7. Incomparable scores cause real bugs
Cosine similarity, `ts_rank` and cross-encoder logits are on different scales. The UI showed "449.4%". RRF
uses ranks for the same reason; the weak-retrieval threshold runs on pre-rerank cosine scores.

## 8. Make the important invariant impossible to forget
Cross-knowledge-base isolation as an optional filter would rely on every caller remembering it. A required
`knowledge_base_id` parameter, a NOT NULL column, and a 422 on the API make omission a compile/validation
error. Also why we didn't adopt LangChain's `BaseRetriever` (its signature can't express that parameter).

## 9. Prove abstractions by using them
"Database-agnostic" was an assertion until Plan Set 20 ingested a book library with zero changes under `app/`.
It also exposed that coexistence in shared tables worked only by luck, which led to Plan Set 21.

## 10. Use a library where it's tricky and generic; don't where it's a thin wrapper
LangChain replaced the chunker (subtle, generic, measurably better) but not retrieval (project-specific
invariant), embeddings or reranking (already thin wrappers around the same sentence-transformers models), or
prompts (static strings). Introducing it *after* building things by hand made that call informed.

## 11. Fail early and explicitly
Unknown embedding/LLM provider raises instead of falling back to the stub; ingestion raises if a source omits
`knowledge_base_id`; `chunk_overlap >= chunk_size` fails validation. Silent defaults hide misconfiguration —
and for an LLM provider, could mean unintentionally calling a billed API.

## 12. Observability must never break the request
Persisting `query_logs` is wrapped in try/except that only logs; every return path (blocked, weak, error,
answered) emits an event. Chunk **content** is never logged (the field doesn't exist).

## 13. Bound every loop
The agentic retry loop is limited by `AGENTIC_MAX_RETRIES`, and a test proves the retriever is called exactly
`1 + max_retries` times even if retrying never helps. The adversarial case matters more than the happy path.

## 14. Guardrails have client-side and server-side halves
Excluding blocked turns from history fixed the UI but not the API. A mitigation in one layer isn't a
guarantee. → [security.md](security.md)

## 15. Reviewing my own work found real gaps
Writing these docs surfaced: an unauthenticated ingest endpoint that accepted a filesystem path (path since restricted to `config/`; auth still missing), no delete
detection in ingestion, no ANN/GIN indexes, hybrid not wired into chat, and no real-LLM evaluation. They are
listed, not hidden.

## What I'd do first with more time

1. Real LLM + LLM-as-judge evaluation (faithfulness, answer relevance), with human spot checks.
2. Authentication and authorization; lock down `/admin/ingest`.
3. HNSW index and a GIN/tsvector column; re-measure latency at 100k+ chunks.
4. Exhaustive ground truth; an ID/SKU-heavy eval set for hybrid; confidence intervals.
5. Delete detection and async ingestion jobs.
6. Deploy to AWS for real (design exists; nothing has been deployed).
