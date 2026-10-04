# Resume Evidence (Plan Set 26)

Every number below was measured on 2026-09-20 against the live RAG database, or reproduced from a stored
result file that this session re-ran and matched. Nothing is estimated except where labelled "estimated".
Percentages are not used; differences are given as absolute points and as question counts.

## Scope: what was actually implemented

FastAPI service; PostgreSQL + pgvector store (Alembic migrations); config-driven Postgres source adapter;
idempotent hash-based ingestion; token-based chunking (LangChain splitter); local sentence-transformers
embeddings; vector, full-text, hybrid (RRF) and cross-encoder-reranked retrieval; metadata filters; per-knowledge-base
isolation; grounded prompt with citation extraction; conversational query rewriting; input/content/weak-retrieval
guardrails; structured per-request observability (`query_logs`); a bounded LangGraph agentic retry pipeline;
evaluation harness; Docker image + AWS deployment *design*; Angular test console.

**Not implemented / not measured:** a real LLM (the LLM is a stub), LLM-judged answer quality, authentication,
ANN/GIN indexes, an actual AWS deployment, hybrid search in the chat path, load or scale testing.

## Measurements

### Corpus and index (queried from the RAG DB)

| Item | Value |
|---|---|
| Source documents | **1,020** — 1,000 `sports-store` products + 20 `library` books (two unrelated source databases) |
| Chunks | **1,020** (all with embeddings) — exactly 1 chunk per document |
| Chunk size (tokens) | avg **42** (min 23, max 81); sports-store avg 42, library avg 48 |
| Chunk configuration | API default **500 tokens / 50 overlap**. Every product fits in one chunk, so the configuration is never exercised on this corpus (300/30, 500/50 and 800/100 give identical metrics) |
| Embedding model | `all-MiniLM-L6-v2`, 384 dimensions, local CPU, cosine distance |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 candidates → top 5 |
| Automated tests | **195 passed, 0 skipped** (full suite, DBs running) |
| Idempotent re-ingest | First run 1,000 chunks; unchanged re-run created 0 chunks, ~5 s (from the Plan Set 7 log; not re-measured today) |

### Retrieval quality — retrieval only, raw question, `evaluation/results/latest.json`
46 scorable questions (52 total minus 6 `unsupported`); follow-ups are scored **without** their history here.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Vector | 0.478 (22/46) | 0.587 | 0.696 (32/46) | 0.554 |
| Keyword (full-text) | 0.196 | 0.217 | 0.217 | 0.203 |
| Hybrid (RRF) | 0.457 (21/46) | 0.543 | 0.674 (31/46) | 0.527 |
| Vector + rerank | 0.522 (24/46) | 0.565 | 0.609 (28/46) | 0.551 |

- **Hybrid retrieval improvement: none.** Hybrid was 1 question worse than vector at Recall@1 and Recall@5,
  and slightly lower MRR. Do not claim a hybrid gain.
- **Reranking, retrieval-only:** Recall@1 +0.043 (+2 questions), MRR −0.003 (flat), Recall@5 −0.087 (−4 questions).
  Reranking reordered the top results without adding recall.

### Retrieval quality — end-to-end pipeline, `rag_evaluation_report.md`
Same 46 scorable questions, through the full `RAGService` (guardrails, query rewriting using each question's history,
reranking option, context budget). Retrieval is scored on the sources that reached the prompt.

| Metric | No rerank | Rerank |
|---|---|---|
| Recall@1 | 0.565 (26/46) | **0.652** (30/46) |
| Recall@3 | 0.696 | 0.696 |
| Recall@5 | **0.804** (37/46) | 0.739 (34/46) |
| MRR | 0.652 | **0.681** |

**Reranking improvement (end-to-end):** Recall@1 +0.087 (+4 questions), MRR +0.029, Recall@5 −0.065 (−3 questions).
The Recall@5 drop is partly attributed (not confirmed) to ground truth that samples 20 of 53 valid products for the
`ambiguous`/`simple` categories.

Why the two tables differ (vector Recall@1 0.478 vs 0.565): the end-to-end run applies query rewriting with
conversation history for the 6 follow-up questions; the retrieval-only run does not. They are different
experiments — quote them separately.

By category (no rerank, Recall@5): exact-match **1.000** (n=18), follow-up 1.000 (6), multi-document 0.750 (4),
simple 0.667 (6), ambiguous 0.667 (6), **semantic 0.333** (6).

### Latency (local CPU, stub LLM, 52 questions per configuration)

| | No rerank | Rerank |
|---|---|---|
| Avg retrieval | 13.4 ms | 17.7 ms |
| Avg end-to-end | **14.3 ms** (p50 13.1, p95 17.3) | **138.8 ms** (p50 146.6, p95 187.3) |
| Avg generation | 0.009 ms (stub) | 0.009 ms (stub) |

Reranking added ≈ 125 ms on average. **End-to-end latency excludes any real LLM call**, so it is not a user-facing
latency figure. Run-to-run jitter observed: 13.7 → 14.3 ms and 136.7 → 138.8 ms across two runs.

### Token usage (estimated with tiktoken)

Avg prompt ≈ **335** tokens (no rerank) / **340** (rerank); avg completion ≈ **68** / **67** — the completion figure is
the stub's output, not a real model's. Estimated cost: **$0.00** (all models local). The estimator's hosted-model rate
table is illustrative.

## Resume bullets

1. **Architecture** — Designed and built a database-agnostic RAG platform (FastAPI, PostgreSQL/pgvector, Docker) in which
   all schema knowledge lives in YAML-configured source adapters; validated by ingesting a second, unrelated database
   (20-book library, 1,020 documents total) with zero changes to the core, and enforced per-knowledge-base isolation as a
   required parameter on every retrieval call.

2. **Retrieval / AI engineering** — Implemented vector, full-text, hybrid (Reciprocal Rank Fusion) and cross-encoder
   reranked retrieval with idempotent hash-based incremental ingestion, citation extraction, conversational query
   rewriting, prompt-injection and weak-retrieval guardrails, structured request tracing, and a bounded LangGraph
   retry pipeline; compared each stage empirically rather than assuming it helped.

3. **Measurable results** — Built a 52-question categorized evaluation harness and 195 automated tests; in the
   end-to-end pipeline, cross-encoder reranking raised Recall@1 from 0.565 to 0.652 (26 → 30 of 46 scorable questions) and MRR
   from 0.652 to 0.681 at ≈125 ms added latency, and the per-category breakdown identified paraphrase queries (Recall@5 0.33)
   as the main weakness.

Optional honest suffix if space allows: "(local models; LLM stubbed; 46-question set)".

## Claims to avoid

- Any hybrid-search improvement (measured: none).
- Any chunk-size improvement (corpus is one chunk per document).
- "Reduced hallucination" or answer-quality numbers (stub LLM; groundedness/appropriateness = 1.0 are structural proxies;
  citation correctness equals Recall@1 by construction).
- Production latency, throughput or scale (no real LLM, no load test, ~1k chunks, exact scan, no ANN index).
- "Deployed on AWS" (design and example task definition only).
- Cost savings (cost is $0 because nothing hosted is called).
- Percent improvements: the sample is 46 questions; one question is 2.2 points of recall.

## Reproducing

```bash
docker start / run the pgvector container on port 5433 (or docker compose up rag-postgres)
export PYTHONPATH=.
.venv/bin/python scripts/run_rag_eval.py          # end-to-end; rewrites evaluation/results/rag_evaluation_report.*
set -a; . ./.env; set +a
.venv/bin/python scripts/run_retrieval_eval.py    # retrieval-only; needs SPORTS_STORE_DATABASE_URL for the chunk-config part
.venv/bin/python -m pytest -q
```

Known script quirks found today: both scripts need `PYTHONPATH=.`; `run_retrieval_eval.py` does not load `.env`, so it
crashes (before writing) at the chunk-config section unless the environment is exported first.

## Caveat on earlier numbers

`docs/learning-log.html` quotes a 22-question dataset (vector Recall@1 0.682 → 0.864 with rerank, MRR 0.772 → 0.890).
That dataset is no longer in the repo and cannot be re-run, so it is **not** used here. The current dataset is harder and
different; numbers are not comparable across the two.
