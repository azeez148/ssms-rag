from abc import ABC, abstractmethod

from app.schemas.document import Document


class KnowledgeSource(ABC):
    @abstractmethod
    async def load_documents(self) -> list[Document]:
        ...
