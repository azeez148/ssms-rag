import json
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ConversationTurn

QuestionCategory = Literal[
    "simple", "semantic", "exact-match", "multi-document", "ambiguous", "unsupported", "follow-up"
]


class EvalQuestion(BaseModel):
    category: QuestionCategory
    question: str
    expected_document_ids: list[str]
    history: list[ConversationTurn] = Field(default_factory=list)


def load_eval_dataset(path: str) -> list[EvalQuestion]:
    with open(path) as f:
        raw = json.load(f)
    return [EvalQuestion.model_validate(item) for item in raw]
