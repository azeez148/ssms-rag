# Hybrid Search

## What is it?

Running lexical (keyword) and semantic (vector) retrieval independently and merging their rankings.
`KeywordRetriever` uses Postgres full-text search (`to_tsvector` / `plainto_tsquery` / `ts_rank`, English
config); `HybridRetriever` fuses it with `VectorRetriever` using **Reciprocal Rank Fusion**
([app/retrieval/fusion.py](../app/retrieval/fusion.py)).

## Why did we need it?

Vectors are good at meaning and bad at exact tokens — SKUs, model codes, precise names. Keyword search is the
opposite. A unit test seeds a model code `XJ429Q`; keyword search finds it instantly, and returns an empty list
for a query sharing no stemmed words (vector search never returns empty — it returns its nearest neighbors).

## How did we implement it?

- Over-fetch `top_k × 4` candidates from each retriever, fuse, then cut to `top_k`.
- RRF: `score(chunk) = Σ 1 / (k + rank)` over each list containing it (k = 60, rank 1-indexed). A chunk that
  appears in both lists outranks one that is #1 in only one.
- **Rank, not raw score**, because scores are incomparable (bounded cosine vs. unbounded `ts_rank`).
- Chunks are keyed by `(document_id, chunk_index)` — which is why `chunk_index` was added to
  `RetrievedChunk`.
- Available via `POST /api/search` with `mode=hybrid`. **Not wired into `/api/chat`.**

## What alternatives existed?

Weighted score blending (needs score normalization/tuning), learned fusion, BM25 (better than `ts_rank`,
needs an extension/engine), Elasticsearch/OpenSearch, SPLADE/sparse learned vectors, a cross-encoder only.

## Why did we choose this approach?

RRF has essentially one knob, needs no score calibration, and keeps everything in Postgres. Both retrievers
already existed behind the same ABC, so hybrid is a composition, not new infrastructure.

## What problems did we encounter?

- **AND semantics:** `plainto_tsquery` requires every term. "SEGA GLAZE RED boots" keyword-mode returned exactly
  the one RED match, while vector and hybrid also surfaced "SEGA GLAZE GREEN" (missing "RED").
- **Hybrid tied vector; it did not win.** On the 22-question set, keyword-only scored Recall@1 **0.318** vs
  vector **0.682**, and hybrid exactly **tied** vector. The questions are natural-language paraphrases — the
  case where keyword search is weakest. That says the *dataset* doesn't exercise hybrid's strength, not that
  hybrid doesn't work. We did not build an SKU-lookup eval set, so hybrid's benefit is demonstrated only by
  the unit test, not by a metric.
- No GIN index / persisted `tsvector`: `to_tsvector` runs per row at query time.

## How did we test it?

`tests/test_fusion.py` (RRF math and dedup), `test_keyword_retriever.py` (exact-code hit, empty on no match,
isolation), `test_hybrid_retriever.py`, and the retrieval evaluation comparing vector / keyword / hybrid.

## What would we change at scale?

Persisted `tsvector` column + GIN index; BM25; build an eval set with codes and exact names and re-measure;
consider weighted RRF; move hybrid into the chat path only if that eval justifies the extra latency.

## How would I explain it in an interview?

"Hybrid search fuses keyword and vector rankings with RRF. I chose rank-based fusion because the two scores
aren't on the same scale. The honest result: on my paraphrase-heavy eval set hybrid only *tied* vector, because
that data doesn't favor keyword search. I'd need an ID/SKU-heavy set to show the benefit, and until then I'd
say I've verified the mechanism, not the lift."
