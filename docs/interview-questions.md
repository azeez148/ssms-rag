# Interview Questions

Grounded in this project. Answers are short on purpose — expand from the linked doc. Where the honest answer
is "I haven't done that", say so.

## Architecture & design

**1. Walk me through the architecture.**
Source adapter → normalized `Document` → clean/hash → chunk → embed → pgvector store (offline). Query: guard →
optional rewrite → retrieve → guard/budget → prompt → LLM → answer + citations. Domain knowledge lives only in
YAML + the adapter. → [architecture.md](architecture.md)

**2. Why a separate RAG database instead of adding vector columns to the app DB?**
Independent lifecycle and migrations; the source can be swapped; and the chat path holds no source
credentials, which is a structural security guarantee.

**3. How do you know the core is really database-agnostic?**
I swapped a sports store for a book library through the same ingestion endpoint with zero changes under
`app/`. → [database-independence.md](database-independence.md)

**4. Why pgvector rather than a dedicated vector DB?**
One store for vectors, metadata filters, joins and bookkeeping; less to operate at this scale. Limits: I have
no ANN index yet; at millions of vectors I'd add HNSW or reconsider.

## Ingestion & chunking

**5. How do you avoid re-embedding unchanged data?**
sha256 of cleaned content vs. stored `content_hash`. Second run over 1000 chunks created 0 and took ~5 s.

**6. What happens if the source deletes a row?**
Nothing — I don't detect deletions. That's a known gap; I'd add a sweep by `external_id` per run.

**7. How did you choose chunk size?**
I compared 300/500/800 on a long document (20 vs 7 chunks) but retrieval metrics were identical on the product
corpus because each product is one chunk. I said the experiment was uninformative rather than pick a winner.

**8. What was wrong with your chunker?**
It summed piece token counts before joining; joined text can tokenize to more, so 7/12 chunks exceeded the
budget. LangChain's splitter had 0/13 over. → [chunking.md](chunking.md)

## Retrieval

**9. Why can't you just use SQL `LIKE`?**
"national team soccer shirt for supporters" shares no words with "FC ARGENTINA 25-26" yet ranks in the top 3.

**10. Pre-filter or post-filter metadata, and why?**
Pre-filter in the `WHERE` before `ORDER BY … LIMIT`; post-filtering can return fewer than K or none.

**11. Explain Reciprocal Rank Fusion. Why not average scores?**
`Σ 1/(k+rank)`. Scores from cosine, `ts_rank` and cross-encoders aren't on the same scale; ranks are. Also
requires over-fetching candidates from each list.

**12. Did hybrid search help?**
On my eval set it tied vector, because the questions are paraphrases. I've verified the mechanism (exact-code
unit test) but not a metric lift — that needs an ID/SKU-heavy set. → [hybrid-search.md](hybrid-search.md)

**13. What does reranking do and what did it cost?**
Cross-encoder rescoring of ~20 candidates. Recall@1 0.565 → 0.652, MRR 0.652 → 0.681 on 52 questions; latency
~13.7 ms → ~136.7 ms (stub LLM, so excludes real generation). Recall@5 dropped 0.804 → 0.739, which I partly
attribute to incomplete ground truth but haven't confirmed.

**14. How would retrieval scale to 10M chunks?**
HNSW index with tuned `ef_search`, partitioning or per-KB indexes, quantization, persisted tsvector + GIN,
caching, and re-evaluating recall vs. latency. None of that is built.

## Generation, conversation, agentic

**15. How do you reduce hallucination?**
Grounding prompt, numbered context with citations, a similarity threshold that skips the LLM, token budget,
and evaluation. It reduces it; it doesn't eliminate it. I can't quantify it on a real LLM yet.

**16. How do you handle follow-up questions?**
Client sends recent history (capped at 3 turns); `QueryRewriter` produces a standalone retrieval query, while
the LLM still sees the original question. With the stub LLM it falls back to concatenation — cruder than a
real rewrite.

**17. What is agentic RAG here and where is the "decision"?**
A LangGraph state machine: evaluate retrieval → enough context ? generate : retries left ? broaden query and
retrieve again : respond. The decision is a plain Python routing function on state, not an LLM call. Retries
are bounded (default 2) and a test proves it. → [architecture.md](architecture.md)

**18. When is an agent the wrong choice?**
When wording mismatch is rare: it adds latency and LLM calls to every weak query. The plain pipeline is
simpler, cheaper, easier to debug.

**19. Why did you use LangChain only for chunking?**
Chunking was subtle, generic and measurably better; retrieval has a required-KB invariant LangChain's
interface can't express; embeddings/reranking are thin wrappers over the same libraries.

## Evaluation

**20. How do you know it's improving?**
Recall@K and MRR on DB-verified questions, per category, with rerank on/off. Semantic questions are the weak
spot (Recall@5 0.333). → [evaluation.md](evaluation.md)

**21. What are the limits of your evaluation?**
Stub LLM (groundedness/appropriateness are proxies; citation correctness = Recall@1), 46 scored questions
(small), sampled ground truth for broad categories, one domain.

**22. Recall@5 was 0.955 but Recall@1 was 0.682 — what does that tell you?**
The relevant doc is retrieved but poorly ranked → reranking, not a better embedding, is the lever.

## Security

**23. How do you defend against prompt injection?**
Regex input guard before retrieval/LLM, the same check on retrieved chunks, and a weak-retrieval gate.
Bypassable by paraphrase — a classifier is the next step. → [security.md](security.md)

**24. How do you keep one tenant's data from another?**
Required `knowledge_base_id` on every `retrieve()`, NOT NULL indexed column, hard SQL filter, 422 on missing,
tested at three layers. It's app-enforced, not row-level security.

**25. What's the weakest point in your security?**
No authentication on any endpoint, including `/admin/ingest`. (The ingest endpoint used to accept an arbitrary server-side file path; it's now restricted to the `config/` directory, but it's still open to any caller.)

## Operations

**26. How is it deployed?**
Multi-stage Docker image, non-root, model weights baked in; Alembic migrations on startup (toggle with
`RUN_MIGRATIONS`); `/health` liveness and `/health/ready` readiness; designed for ECS Fargate + RDS + Secrets
Manager + CloudWatch. Designed, not deployed. → [deployment.md](deployment.md)

**27. How do you debug a bad answer?**
Walk `query_logs`: were retrieved chunk IDs/scores sensible → was rerank order off → was context truncated
(token estimate vs. sources) → generation latency/tokens. Chunk text isn't logged by design.

**28. What does it cost to run?**
$0.00 currently — every model is local and the LLM is a stub. The cost estimator has an illustrative rate
table for hosted models; it isn't a measured bill.

## Reflection

**29. What would you do differently?**
Build an LLM-judge eval and a chunk-budget regression test earlier; keep eval datasets exhaustive; add auth
before adding endpoints.

**30. What's the most important thing you learned?**
Measure each stage separately and be precise about what each number means — several of my "perfect" scores
are artifacts of a stub LLM. → [lessons-learned.md](lessons-learned.md)
