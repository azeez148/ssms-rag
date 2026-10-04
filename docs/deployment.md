# Deployment (Plan Set 24)

Local: Docker Compose (FastAPI + PostgreSQL/pgvector). Production: AWS, using the
**same image** with configuration injected from the environment.

## Local

```bash
cp .env.example .env
docker compose up --build      # app on :8000, pgvector Postgres on :5433
curl localhost:8000/health/ready
```

## AWS target architecture

```text
Internet → ALB (HTTPS, ACM cert) → ECS Fargate service (rag-app, private subnets)
                                        │
                                        ├─→ RDS PostgreSQL 16 + pgvector (private subnets)
                                        ├─→ Secrets Manager (RAG_DATABASE_URL, future LLM API keys)
                                        └─→ CloudWatch Logs (/ecs/rag-app)
Image: built once → ECR → referenced by the task definition
```

| Concern | Decision |
|---|---|
| Compute | ECS Fargate, ≥2 tasks across AZs. 1 vCPU / 3 GB: the image bakes in sentence-transformers + cross-encoder weights (loaded into memory). |
| Database | RDS PostgreSQL 16 with the `vector` extension (`CREATE EXTENSION vector`, done by the initial migration/`scripts/init_db.py`). Security group allows 5432 only from the ECS task SG. |
| Image | `Dockerfile` multi-stage, non-root user, **CPU-only torch** (installed from PyTorch's CPU index first; the default PyPI build pulls GBs of unused CUDA libraries), model weights baked in so no HuggingFace download at startup. ~1.75 GB. Verified locally: builds, applies migrations from an empty DB, `/health` + `/health/ready` OK, Docker `HEALTHCHECK` reports healthy, runs as user `rag`. Push to ECR, deploy by immutable tag (git SHA), never `latest`. |
| Task definition | `deploy/ecs-task-definition.json` (placeholders in `<...>`). |

## Health checks

| Endpoint | Meaning | Used by |
|---|---|---|
| `GET /health` | **Liveness** — process is up; never touches the DB | Dockerfile `HEALTHCHECK`, ECS container health check |
| `GET /health/ready` | **Readiness** — DB reachable, else `503` | compose service healthcheck; optionally the ALB target group |

Why two: if the DB has a brief outage, a readiness failure takes the task out of
rotation without ECS killing and restarting every task (which a DB-dependent
liveness check would do, turning a blip into an outage). The ALB target group
uses `/health` by default for stable deploys; switch it to `/health/ready` if
you prefer traffic to drain during DB trouble. Use a ≥60 s `startPeriod`
because the model weights load at startup.

## Environment variables

Everything is read by `app/core/config.py` (pydantic-settings); env vars override `.env`.
`.env` is never baked into the image (`.dockerignore`) — in AWS there is no `.env`.

| Variable | Where it comes from in AWS |
|---|---|
| `RAG_DATABASE_URL` | **Secret** (Secrets Manager → task `secrets`) |
| `ANTHROPIC_API_KEY` (enables Claude answers; `LLM_PROVIDER=auto` falls back to the offline provider without it) | **Secret** |
| `APP_ENV`, `LOG_LEVEL`, `CORS_ALLOWED_ORIGINS`, model/guardrail settings | Plain `environment` in the task definition |
| `RUN_MIGRATIONS` | Plain env (see Migrations) |

## Secrets

- Never in the image, the repo, task-definition plain `environment`, or logs.
- Store in Secrets Manager; the task **execution role** needs `secretsmanager:GetSecretValue`
  on those ARNs only (ECS injects them as env vars at container start).
- The task **role** (what the app code can call) stays minimal — no secret access needed.
- RDS master credentials: use RDS-managed rotation; if the URL is rotated,
  redeploy (env vars are read at start).
- compose's `rag/rag` password is a dev default only (`POSTGRES_PASSWORD` overrides it).

## Logging

- Logs go to **stdout** (`app/core/logging.py`) → the `awslogs` driver → CloudWatch Logs
  group `/ecs/rag-app`. No log files, no volumes.
- Format is `key=value` (`level=… logger=… message=…`); the per-request
  observability events (`rag.observability`) are logged at INFO, errors at ERROR —
  queryable with CloudWatch Logs Insights (`parse @message "logger=* message=*"`).
- Set a retention period on the log group (e.g. 30 days).
- `LOG_LEVEL` is an env var — raise to DEBUG temporarily without a rebuild.

## Database migrations

Alembic, run by `docker/entrypoint.sh` **before** uvicorn starts, with retry/backoff
(a fresh RDS or slow DB isn't accepting connections instantly). A task that passes
health checks therefore always has the current schema.

Race hazard: several tasks starting at once (rolling deploy, autoscale) would all run
`alembic upgrade head` against the same DB. Options:

1. **Small/single-task setup:** leave `RUN_MIGRATIONS=true` (default).
2. **Recommended for ≥2 tasks:** run migrations once as a one-off ECS task
   (`aws ecs run-task … --overrides '{"containerOverrides":[{"name":"rag-app","command":["alembic","upgrade","head"]}]}'`)
   as a deploy pipeline step *before* updating the service, and set
   `RUN_MIGRATIONS=false` on the service (as in the example task definition).
   Migrations must be backward compatible with the previous app version
   (expand → deploy → contract) so rolling deploys are safe.

## Startup behavior

1. Entrypoint: (optional) migrate → `exec uvicorn` (PID 1 receives SIGTERM → graceful shutdown).
2. App configures logging, loads settings, registers routes; the embedding/reranker models
   load from the baked-in cache (`HF_HOME`), not the network.
3. ECS waits for `startPeriod`, then health checks gate the service and the ALB.
4. Rolling deploy: min healthy 100 %, max 200 %, so the old tasks keep serving until new ones are healthy.

## Not done / open items

- The Angular chat UI (`frontend/`) is not part of this image or task definition. It builds to static files (`ng build`); host them on S3 + CloudFront (or behind the same ALB) with an SPA fallback to `index.html`, and set `CORS_ALLOWED_ORIGINS` to that origin. The API base URL is in `frontend/src/environments/environment.ts`.
- The chat UI runs ingestion when it opens, so the task role/network must allow the app to reach the source databases, and `/admin/ingest` must be protected (see security.md) before it is reachable from browsers.

- No IaC (Terraform/CDK) written — the table above plus `deploy/ecs-task-definition.json` is the spec.
- Not deployed to a real AWS account; the task definition has not been registered. (The image itself has been built and run locally.)
- Real LLM provider keys don't exist yet (`LLM_PROVIDER=stub`); the secret wiring is shown for `RAG_DATABASE_URL` and would be the same pattern for keys.
- No auth on the API yet (ALB/WAF or an app-level auth layer is needed before exposing publicly). `/admin/ingest` only loads configs from `SOURCE_CONFIG_DIR` (default `config/`), but is still unauthenticated.
- `TORCH_VERSION` build arg (default 2.14.0) must be kept in step with development; `docker build --build-arg TORCH_VERSION=...`.
