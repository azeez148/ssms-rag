# Architecture

## One-paragraph summary

A database-agnostic RAG platform. A **source adapter** turns rows from any application database into
normalized `Document`s; an **ingestion pipeline** cleans, chunks, embeds and stores them in a separate
PostgreSQL + pgvector database that the RAG system owns; a **chat pipeline** retrieves the most relevant
chunks, applies guardrails, builds a numbered context, calls an LLM provider, and returns an answer with
citations. The RAG core never learns what a "product" or a "book" is — that knowledge lives only in YAML
source configs.

## Two pipelines, one store

```text
INGESTION (offline, batch, write-heavy)             QUERY (online, read-only, latency-sensitive)

Source DB (Sports Store / Library / ...)            POST /api/chat  (knowledge_base required)
        │ read-only                                          │
        ▼                                                    ▼
PostgresKnowledgeSource  ← config/*.yaml            InputGuard (injection / sensitive-data)
        │  (the ONLY place that knows the schema)            │
        ▼                                                    ▼
Document (id, content, source, metadata)            QueryRewriter (only if history given)
        │                                                    │
        ▼                                                    ▼
clean_content → sha256 content_hash                 Embed query ─► VectorRetriever
        │  unchanged? skip                                   │   (WHERE knowledge_base_id = ?)
        ▼                                                    ▼
LangChainRecursiveChunker (token-based)             [optional] CrossEncoderReranker
        │                                                    │
        ▼                                                    ▼
EmbeddingProvider.embed_batch                       weak-retrieval check → content guard
        │                                                    │  → token budget
        ▼                                                    ▼
        └──────────►  RAG PostgreSQL + pgvector ◄─── build numbered context → LLMProvider
                      documents / chunks /                   │
                      ingestion_runs / query_logs            ▼
                                                    answer + sources + citations
                                                    + one structured log line / query_logs row
```

The two pipelines share only two things: the storage tables and the embedding model. That is deliberate —
ingestion can be slow and expensive; a chat request must never pay for it.

## Layers (all under `app/`)

| Layer | Package | Responsibility | Knows about the domain? |
|---|---|---|---|
| Transport | `api/` | HTTP ↔ Python, Pydantic validation, DI | No |
| Orchestration | `services/` (`RAGService`, `IngestionService`), `agentic/` | Order the steps | No |
| Source adapters | `sources/` + `config/*.yaml` | Rows → `Document` | **Yes — the only layer that does** |
| Ingestion | `ingestion/` | Clean, chunk | No |
| Embeddings | `embeddings/` | Text → vector (`EmbeddingProvider` ABC) | No |
| Retrieval | `retrieval/` | Vector, keyword, hybrid (RRF), reranking wrapper | No |
| Reranking | `reranking/` | Cross-encoder rescoring (`Reranker` ABC) | No |
| Generation | `generation/` | Prompt, context builder, citations, query rewriter, `LLMProvider` ABC | No |
| Guardrails | `guardrails/` | Input guard, content guard, weak-retrieval check, token budget | No |
| Observability | `observability/` | Structured event, cost estimate, `query_logs` | No |
| Evaluation | `evaluation/` | Recall@K, MRR, RAG-level metrics | No |

## Storage (RAG database, owned by this system)

- `documents` — `external_id` (unique), `knowledge_base_id` (indexed, NOT NULL), `source`, `source_type`,
  `title`, `content_hash`, JSON `metadata`.
- `chunks` — `document_id` (FK, `ON DELETE CASCADE`), `content`, `chunk_index`, `token_count`,
  `embedding vector(384)`, JSON `metadata`.
- `ingestion_runs` — per-run created / updated / unchanged counts, status, error.
- `query_logs` — one row per chat request (query, per-stage latency, chunk IDs + scores, models, tokens, outcome).

Schema is managed by Alembic (`alembic/versions/`), run automatically at container start.

## Key design decisions

1. **Domain isolation is physical, not conventional.** `sources/` + YAML is the only code that names tables
   or columns. Proved by switching domains (see [database-independence.md](database-independence.md)).
2. **Separate RAG DB from the source DB.** No vector columns on business tables; the source can change or be
   replaced without touching RAG storage, and the chat path has no credentials to the source at all.
3. **Interfaces at every seam** (`KnowledgeSource`, `ChunkingStrategy`, `EmbeddingProvider`, `Retriever`,
   `Reranker`, `LLMProvider`). Each swap is a factory/config change, not a rewrite.
4. **`knowledge_base_id` is a required parameter of `Retriever.retrieve()`**, not an optional filter — it can't
   be forgotten at a call site.
5. **Guardrails before generation.** Blocked or weak-retrieval questions never reach the LLM.
6. **Stateless server.** Conversation history is sent by the client each turn and capped (`MAX_HISTORY_TURNS`).
7. **Free, local models by default** (MiniLM embeddings, MS-MARCO cross-encoder, stub LLM) so the whole system
   is testable offline at zero cost.

## Two chat pipelines

- `POST /api/chat` — straight line: retrieve once; weak → "insufficient information".
- `POST /api/chat/agentic` — LangGraph state machine that can rewrite the query with broader terms and
  retry (bounded by `AGENTIC_MAX_RETRIES`, default 2) before giving up. Same components, same `ChatResult`.

## Deployment shape

Multi-stage Docker image (non-root, model weights baked in) → ECS Fargate behind an ALB, RDS PostgreSQL with
pgvector, Secrets Manager, CloudWatch Logs. See [deployment.md](deployment.md). The image builds and runs
locally (verified, ~1.75 GB, CPU-only torch); the AWS side is a **design plus example task definition** and has
not been deployed.

## Known gaps (state these up front in an interview)

- The LLM is a **stub** (`LLM_PROVIDER=stub`); no real hosted model has been wired in or evaluated.
- No ANN index on `chunks.embedding` (exact scan) and no GIN index for keyword search — fine at ~1k chunks.
- `/api/chat` uses vector retrieval only; hybrid search exists on `/api/search` but is not wired into chat.
- No authentication on any endpoint, including `/admin/ingest` (which is at least restricted to loading configs from `config/`).
- Knowledge-base isolation is enforced in application code, not by database row-level security.
