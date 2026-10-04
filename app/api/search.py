from fastapi import APIRouter, Depends
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.reranking.base import Reranker
from app.reranking.factory import get_reranker
from app.retrieval.base import Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.reranking_retriever import RerankingRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.retrieval import SearchRequest, SearchResponse

router = APIRouter(prefix="/api")


def get_retriever() -> VectorRetriever:
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    return VectorRetriever(embedding_provider)


def get_session_factory() -> sessionmaker:
    return SessionLocal


def get_reranker_dependency() -> Reranker:
    return get_reranker(get_settings())


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    vector_retriever: VectorRetriever = Depends(get_retriever),
    session_factory: sessionmaker = Depends(get_session_factory),
    reranker: Reranker = Depends(get_reranker_dependency),
) -> SearchResponse:
    active_retriever: Retriever

    if request.mode == "vector":
        active_retriever = vector_retriever
    elif request.mode == "keyword":
        active_retriever = KeywordRetriever(session_factory)
    else:
        active_retriever = HybridRetriever(vector_retriever, KeywordRetriever(session_factory))

    if request.rerank:
        settings = get_settings()
        active_retriever = RerankingRetriever(active_retriever, reranker, settings.rerank_candidate_multiplier)

    results = await active_retriever.retrieve(
        request.query, request.knowledge_base, request.top_k, filters=request.filters
    )
    return SearchResponse(results=results)
