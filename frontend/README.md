# RAG Platform — Test Console

A minimal Angular app for manually testing the RAG backend (`../app`) as each Plan
Set lands. Not part of the RAG architecture itself — it's a throwaway UI over the
API, not something the RAG core depends on.

Currently has test panels for:

- **Health** (Plan Set 1) — `GET /health`
- **Ingest** (Plan Set 7) — `POST /admin/ingest`
- **Semantic Search** (Plan Set 8) — `POST /api/search`

New panels get added here as later plan sets add endpoints (chat in Plan Set 11,
citations in Plan Set 12, etc.).

## Run it

1. Start the backend (from `rag/`):
   ```bash
   docker-compose up -d
   .venv/bin/uvicorn app.main:app --port 8000
   ```
2. Start this app:
   ```bash
   npm install
   npx ng serve
   ```
3. Open `http://localhost:4200`.

The backend's API base URL is set in `src/environments/environment.ts`
(defaults to `http://localhost:8000`). CORS for `localhost:4200` is already
enabled on the backend in `app/main.py`.


## Routes

- `/` — the chat assistant (`src/app/chat/`). Opening it loads `GET /api/knowledge-bases`, runs
  `POST /admin/ingest` for the selected knowledge base, then enables the composer. State lives in
  `ChatStore` (signals, conversations kept per knowledge base in `localStorage`); answers are rendered by
  `format-answer.ts`, which never produces HTML.
- `/console` — the developer test console (health, ingest, search, raw chat).

Tests: `CHROME_BIN=/usr/bin/google-chrome npx ng test --watch=false --browsers=ChromeHeadless`.
