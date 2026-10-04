# Ingestion

## What is it?

The offline pipeline that turns source-system rows into searchable vectors:
`Source adapter → Document → clean → hash check → chunk → embed → store`.
Entry point: `POST /admin/ingest` with a source config path; logic in
[app/services/ingestion_service.py](../app/services/ingestion_service.py).

## Why did we need it?

Embedding is slow and (with hosted models) costs money; queries must be fast and read-only. Separating the two
means a chat request never pays ingestion cost, and re-running ingestion must be cheap and safe.

## How did we implement it?

1. **`KnowledgeSource.load_documents()`** (ABC) returns normalized `Document`s. `PostgresKnowledgeSource` is
   driven entirely by YAML: a SQL query per entity, `id_column`, `title_column`, a `content_template`,
   `metadata_columns`, `static_metadata`, and a required `knowledge_base_id`. `Document.id` is
   `"{entity}-{row_id}"` so entities from one source can't collide.
2. **`clean_content`** — generic whitespace normalization only.
3. **Idempotency via content hash.** `sha256(cleaned content)` is compared to `documents.content_hash`.
   Unchanged → skipped entirely (no chunk, no embed, no write). Changed → delete that document's chunks and
   regenerate (full replacement, not diffing). New → insert.
4. **Chunk → `embed_batch` (one model call per document) → insert chunks.**
5. **Commit per document**, so a failure at document 501 of 1000 keeps the first 500. Each run records
   created / updated / unchanged / chunks_created in `ingestion_runs`.
6. `IngestionService` refuses to run if a source doesn't set `knowledge_base_id` (no silent default).

Content design matters more than it looks: raw rows (`selling_price=275, category_id=4`) are poor RAG
content. The template turns them into prose ("Price: 275", "Category: Kids Jersey"), and the SQL itself filters
`is_active AND can_listed` and never selects cost price or customer/order data — business rules live in config.

## What alternatives existed?

| Alternative | Trade-off |
|---|---|
| Re-embed everything every run | Simple, but cost scales with corpus size, not change size |
| Row-level change data capture / timestamps | More efficient; needs source cooperation (`updated_at`) — not assumed |
| Diff and patch individual chunks | Complex and error-prone vs. delete + re-insert |
| Vector columns on the source tables | Couples RAG to the app DB; rejected |

## Why did we choose this approach?

A content hash needs nothing from the source (no `updated_at`, no CDC), and hashing the *cleaned* content means
cosmetic whitespace changes don't trigger re-embedding.

## What problems did we encounter?

- Embedding dimension mismatch: the column was first created as `vector(1536)` (an OpenAI-sized placeholder);
  switching to a 384-dim local model made pgvector reject inserts until the column was resized.
- Early on there was no migration tool — tables were dropped and recreated. Alembic has since been added.
- `PostgresKnowledgeSource.load_documents` is `async` but wraps a sync SQLAlchemy engine in
  `asyncio.to_thread` to avoid blocking the event loop.

## How did we test it?

A throwaway dummy table (the adapter knows nothing about it) proves the adapter has no hardcoded schema;
changing one row and asserting the other is untouched proves incremental behavior; a second ingest over the
real data.

Measured: first run over Sports Store created **1000 chunks**; the immediate re-run created **0** and took
about 5 seconds (vs. re-embedding everything).

## What would we change at scale?

Stream/batch documents instead of `list[Document]`; parallelize embedding; add a background job runner with
progress and retries instead of a synchronous HTTP call; detect **deletions** (source rows that disappear are
currently not removed — a real gap); authenticate `/admin/ingest`; use source `updated_at`/CDC to avoid
reading everything.

## How would I explain it in an interview?

"Ingestion is idempotent and incremental. I hash the cleaned content of each document; if the hash matches,
I skip it, so cost scales with what changed, not corpus size — my second run did zero embeddings. The source
adapter is config-driven, so the pipeline never sees SQL or table names, and I commit per document so a failure
partway through doesn't lose progress. What I didn't build is delete detection and async jobs."
