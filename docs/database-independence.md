# Database Independence

## What is it?

The requirement that the RAG core has no knowledge of any particular business domain, so the data source can be
swapped by changing configuration, not code. All schema knowledge lives in `sources/` + `config/*.yaml`.

## Why did we need it?

The first data source (a sports-store Postgres DB) is a means, not the goal. A platform that hard-codes
"Product" and "Jersey" is a demo; one that ingests any source is reusable (and is the credible interview claim).

## How did we implement it?

- **`Document`**: `id, content, source, source_type, title, metadata` — the only thing downstream code sees.
- **`KnowledgeSource` ABC** with `load_documents()`; `PostgresKnowledgeSource` is generic and config-driven:
  entities with a SQL `query`, `id_column`, `title_column`, `content_template`, `metadata_columns`,
  `static_metadata`, and a required `knowledge_base_id`.
- **Separate RAG database** with its own tables (`documents`, `chunks`, `ingestion_runs`, `query_logs`);
  no vector columns on business tables.
- Metadata filters are a plain `dict[str, str]` — no method like `filter_by_category`.
- Enforced by a test that `Document.model_fields` contains no domain-specific names.

## What alternatives existed?

Vector columns on the app's tables; text-to-SQL at query time; a per-domain RAG service; a heavyweight
connector framework (Airbyte/LlamaHub/LangChain loaders).

## Why did we choose this approach?

The RAG store can be re-created from the source at any time, the source can change independently, and the
chat path needs no source credentials.

## What problems did we encounter?

- **Proof by argument isn't enough**, so Plan Set 20 did the swap for real: a `library_db` (20 books) ingested
  through the *same* `/admin/ingest`, chunker, embedder, retriever, service and guardrails.
  Ledger: created `config/library_source.yaml` and one test; added one env line; touched **zero files under
  `app/`** (in that plan set). "Is there a dystopian novel about a totalitarian surveillance state?" ranked
  *1984* first (similarity 0.634) with no shared words.
- **Coexistence was luck.** Both domains shared the same tables; nothing stopped a cross-domain hit except
  semantic distance. Plan Set 21 fixed it with a required `knowledge_base_id` — which *did* touch ~20 files,
  a necessary, honest cost of making isolation an invariant.
- Adding a source with a different *shape* is still work: the Postgres adapter supports SQL rows only. A
  file/API/web source means a new `KnowledgeSource` implementation (the interface exists; only Postgres is
  implemented).
- Content quality is a config responsibility: raw IDs/flags are poor RAG content; the template must produce
  prose, and the SQL must filter unpublished rows.

## How did we test it?

`tests/test_postgres_source.py` (against a throwaway table the adapter knows nothing about),
`test_sports_store_config.py`, `test_library_source_config.py` (real end-to-end ingest + retrieval on a second
domain; skipped if `library_db` isn't reachable), and the three-layer isolation tests.

## What would we change at scale?

Implement `FileSource`/`APISource`; support joins/related-entity templates in `EntityConfig`; incremental
reads via `updated_at`; delete detection; per-source read-only credentials; row-level security for tenants;
config validation and a dry run (`scripts/preview_source.py` already previews documents).

## How would I explain it in an interview?

"The RAG core only understands documents, content and metadata. Domain knowledge is in YAML that the adapter
interprets. I proved it by switching from a sports store to a book library with zero changes under `app/`,
using the same ingestion endpoint. That test also exposed the weakness — both domains shared tables and only
stayed separate by embedding distance — which led me to add a required knowledge-base ID. What it does *not*
yet do is ingest non-SQL sources; the interface is there but only Postgres is implemented."
