from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
