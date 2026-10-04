# syntax=docker/dockerfile:1

# ---- builder: compile/install dependencies into a venv -------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch is installed first, from PyTorch's CPU-only index. sentence-transformers
# depends on torch, and left to itself pip pulls the default PyPI build, which
# drags in ~GBs of CUDA libraries (nvidia-*, triton) this CPU-only service
# never uses — a multi-GB image and a very slow build. With the CPU wheel
# already installed, the `-r requirements.txt` step sees torch as satisfied.
# Keep TORCH_VERSION in step with what local development uses.
ARG TORCH_VERSION=2.14.0
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- runtime: slim image with just the venv + app code -------------------
FROM python:3.12-slim AS runtime

# Non-root user — the app never needs root once dependencies are installed.
RUN useradd --create-home --shell /bin/bash rag
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY app/ app/
COPY plugins/ plugins/
COPY alembic/ alembic/
COPY alembic.ini .
COPY config/ config/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Bake the default embedding/reranker model weights into the image at build
# time, matching .env.example's defaults. Without this, the *first* request
# after every fresh container start pays a cold download from the HuggingFace
# Hub — slow, and a hard dependency on an external service being reachable at
# startup. Change EMBEDDING_MODEL/RERANKER_MODEL? Rebuild the image (or accept
# that one slow first request while it downloads on demand instead).
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && chown -R rag:rag /app

USER rag
EXPOSE 8000

# Same /health route (Plan Set 1) used by: this HEALTHCHECK, docker-compose's
# service healthcheck, and — in the AWS design below — the ALB target group
# and ECS task health check. One endpoint, three consumers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
