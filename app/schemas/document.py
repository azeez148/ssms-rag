from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    source: str
    source_type: str
    title: str | None = None
    metadata: dict = Field(default_factory=dict)
