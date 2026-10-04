from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.core.database import engine
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness: the process is up. Never touches the database, so a DB blip
    can't make the orchestrator kill otherwise-healthy containers."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_env=settings.app_env,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def ready(response: Response) -> ReadinessResponse:
    """Readiness: the app can actually serve requests (database reachable).
    Returns 503 otherwise, so a load balancer stops routing to this task
    without restarting it."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable", database="unreachable")
    return ReadinessResponse(status="ok", database="ok")
