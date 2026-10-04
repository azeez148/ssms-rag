# Retrieval

## What is it?

Given a question, return the top-K most relevant chunks. Everything sits behind the `Retriever` ABC
([app/retrieval/base.py](../app/retrieval/base.py)):
`retrieve(query, knowledge_base_id, top_k, filters) -> list[RetrievedChunk]`.
Implementations: `VectorRetriever`, `KeywordRetriever`, `HybridRetriever`, and `RerankingRetriever`
(a wrapper). Hybrid and reranking are in [hybrid-search.md](hybrid-search.md) and [reranking.md](reranking.md).

## Why did we need it?

The LLM can only see a small context window, so the quality of what we retrieve caps the quality of the answer.
Retrieval is also the part we can measure without an LLM.

## How did we implement it?

**Vector retrieval:** embed the query with the ingestion model, then in SQL:

```text
SELECT … FROM chunks JOIN documents
WHERE documents.knowledge_base_id = :kb   [AND metadata filters]
ORDER BY embedding <=> :query_vector LIMIT :top_k
```

- `similarity = 1 − cosine_distance`; both are returned.
- **`knowledge_base_id` is a required positional argument** on every retriever, not an optional filter
  (see [security.md](security.md)).
- **Pre-filtering**: optional `filters` (a plain `dict`, e.g. `{"category": "Boot-Adult"}`) are SQL `WHERE`
  clauses applied *before* `ORDER BY … LIMIT`, comparing JSON values as strings. Post-filtering the top K in
  Python could return fewer than K results, or none. Live proof: "Argentina" filtered to `Boot-Adult` returned
  only boots, at similarities as low as 18% — the vector index alone would never surface them.
  A filter that matches nothing returns `[]`, not an error.
- `/api/search` (`mode=vector|keyword|hybrid`, `rerank`) never calls an LLM, so retrieval can be inspected on
  its own. `/api/chat` currently uses `VectorRetriever` (+ optional rerank).
- Retrievers are injected through FastAPI dependencies, so tests swap in fakes.

## What alternatives existed?

A dedicated vector DB (Pinecone, Qdrant, Weaviate); ANN indexes (HNSW / IVFFlat); MMR for result diversity;
multi-query / HyDE; parent-document retrieval; LangChain's `BaseRetriever`.

## Why did we choose this approach?

pgvector keeps vectors, metadata filters, joins and bookkeeping in one database — no second store to sync.
We *didn't* adopt LangChain's `BaseRetriever` because its `_get_relevant_documents(query)` signature has no
required knowledge-base argument; adopting it would have made the isolation guarantee optional.

## What problems did we encounter?

- Vector search **always returns neighbors**, however weak — hence the weak-retrieval threshold (0.35) in
  guardrails. Live: "boiling point of tungsten" against a jersey catalog returned weak matches, so the system
  answered "insufficient information" without calling the LLM.
- Tests seed synthetic chunks into a DB that already holds real ingested data; real rows outranked test rows
  for common queries. Fix: a transaction per test that is rolled back, plus distinctive test content.
- Different retrievers produce **incomparable scores** (cosine 0–1, `ts_rank` unbounded, cross-encoder logits) —
  the UI once showed "449.4%". Never compare or threshold scores across methods.

## How did we test it?

`tests/test_retrieval.py`, `test_metadata_filtering.py`, `test_keyword_retriever.py`, `test_search_api.py`
(the DB-backed ones skip when Postgres isn't running); the offline evaluation in
[evaluation.md](evaluation.md). Measured on 22 DB-verified questions (Plan Set 9): Recall@1 **0.682**,
Recall@3 **0.818**, Recall@5 **0.955**, MRR **0.772**.

## What would we change at scale?

Add an HNSW index (currently **none** — exact scan, fine at ~1k chunks, not at millions); tune
`hnsw.ef_search`; put hybrid in the chat path; add MMR for diversity; consider a denormalized/indexed column for
hot metadata filters instead of JSON extraction; pool DB connections properly.

## How would I explain it in an interview?

"Retrieval is nearest-neighbor search in SQL over pgvector, always scoped to a knowledge base and optionally
pre-filtered by metadata. I evaluate it separately from generation: on 22 verified questions, Recall@5 was
0.955 but Recall@1 only 0.682 — the right answer is usually in the pool, just not first, which is what
reranking fixes. I'd be upfront that I have no ANN index; at this size exact search is fine, and HNSW is the
first thing I'd add at scale."
