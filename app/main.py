import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.actions import router as actions_router
from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.knowledge_bases import router as knowledge_bases_router
from app.api.search import router as search_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.embeddings.factory import get_embedding_provider
from app.reranking.factory import get_reranker
from dotenv import load_dotenv

load_dotenv()

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


def _warm_models() -> None:
    try:
        get_embedding_provider(settings)
        get_reranker(settings)
        logger.info("embedding and reranker models loaded")
    except Exception:
        # Not fatal: the first request that needs a model will load it (and report any real error).
        logger.warning("model warm-up failed", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the models in the background so the first chat message doesn't pay
    # the multi-second load, without delaying the app becoming reachable
    # (/health must answer while this runs).
    warmup = asyncio.create_task(asyncio.to_thread(_warm_models))
    yield
    warmup.cancel()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# Defaults to the Angular test console (localhost:4200); production sets
# CORS_ALLOWED_ORIGINS to the real frontend origin(s), no code change needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(admin_router)
app.include_router(search_router)
app.include_router(knowledge_bases_router)
app.include_router(chat_router)
app.include_router(actions_router)
