# RAG Platform

Database-agnostic Retrieval-Augmented Generation platform. See `prompt.md` for the full project plan.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
.venv/bin/uvicorn app.main:app --reload
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Chat assistant (Angular)

`frontend/` is an Angular app. Its home page (`/`) is a chat assistant over the
configured knowledge bases:

```bash
docker-compose up -d rag-postgres ollama               # knowledge database + local LLM
docker-compose exec ollama ollama pull qwen2.5:1.5b    # once: model for LLM_PROVIDER=ollama
.venv/bin/uvicorn app.main:app --port 8000             # API (run from the repo root)
cd frontend && npx ng serve                            # http://localhost:4200
```

- **Opening the chat runs ingestion** for the selected knowledge base (idempotent —
  unchanged source rows are skipped), then unlocks the composer. If the source is
  unreachable but data was indexed earlier, you can still chat, with a warning.
- **Answers** come from the engine chosen by `LLM_PROVIDER` / `LLM_MODEL`: `anthropic`,
  `openai`, `gemini`, `qwen`, `ollama` (local, free; e.g. `qwen2.5:1.5b`), or
  `openai_compatible` (any other Chat Completions endpoint via `LLM_BASE_URL`). Switching
  is a `.env` change, not a code change; see `.env.example` for each provider's key and
  default model. If the model fails or refuses, that request is answered by the offline
  extractive provider, which lists the matching records with their fields and `[n]`
  citations (no cost, no key). It's also what `LLM_PROVIDER=auto` uses when no
  `ANTHROPIC_API_KEY` is set.
- **Shop actions (sports store, for staff):** the chat also handles till requests, reading
  live data from the store DB rather than the index snapshot. Type `help` in the chat for the list.
  - Products: `list products argentina`, `show out of stock products`, `list categories`,
    `how many FC ARGENTINA 25-26 are left in size M?`
  - Add / edit products: `add product Argentina Home 26 category Jersey price 799 cost 400 sizes S:2 M:5`,
    `update #41 price 799`, `update #41 offer 699` (`offer none` removes it), `change #41 category to Jersey`,
    `rename #41 to …`, `deactivate #41`
  - Stock: `add 5 #41 size M`, `remove 1 #41 size M`, `set stock of #41 M=5 L=2`
  - Sales: `sell 2 #41 size M to Rahul 9876543210 upi` (also `cash`/`card`, `express delivery`…;
    walk-in if no customer), `sale #1521`, `last 5 sales`, `update sale #1521 payment upi`,
    `change sale #1521 qty 2 size L`, `cancel sale #1521`, `mark sale #1521 as shipped`
  - Reports: `today's sales`, `sales this week`, `sales this month`, `total sales`
  - Day: `start day with 2000`, `add expense 150 for tea`, `day status`, `end day with 5400 reason: …`

  Every write is only *proposed* in the chat, and is made through the store API
  (`SPORTS_STORE_API_URL`, the `ssms-backend-fastapi` app) when staff press the proposal's
  confirm button. Day management needs a store login: set `SPORTS_STORE_API_MOBILE` and
  `SPORTS_STORE_API_PASSWORD` (a staff or admin user) or `SPORTS_STORE_API_TOKEN`.

  Requests are recognised by rules, not the LLM, so they behave the same with any
  `LLM_PROVIDER`; anything else is answered by RAG as before. The logic lives in
  `plugins/sports_store/`, switched on by the `actions:` block in
  `config/sports_store_source.yaml`; `app/` stays domain-free.
- Knowledge bases are discovered from `config/*.yaml` (`GET /api/knowledge-bases`);
  optional `title`, `description` and `suggested_questions` in a config feed the UI.
- The developer test console (health / ingest / search / raw chat) is at `/console`.

## Manual test console (Angular)

The old single-page test console lives at `/console` in the same app. See
`frontend/README.md`.

## Docker / deployment

```bash
docker compose up --build   # FastAPI on :8000 + PostgreSQL/pgvector
```

Health: `GET /health` (liveness), `GET /health/ready` (readiness, checks the DB).
Migrations run on container start (`RUN_MIGRATIONS=false` to skip). The AWS design
(ECS Fargate + RDS + Secrets Manager + CloudWatch) is in `docs/deployment.md`, with an
example task definition in `deploy/ecs-task-definition.json`.
