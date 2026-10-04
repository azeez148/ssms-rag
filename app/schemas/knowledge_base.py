from pydantic import BaseModel


class KnowledgeBaseInfo(BaseModel):
    id: str
    name: str
    description: str | None = None
    suggested_questions: list[str]
    # Send this to POST /admin/ingest to (re)build the knowledge base.
    config_path: str
    documents_indexed: int


class KnowledgeBasesResponse(BaseModel):
    # Which engine writes answers: "anthropic" (Claude), "extractive" (offline) or "stub".
    answer_engine: str
    answer_model: str
    knowledge_bases: list[KnowledgeBaseInfo]
