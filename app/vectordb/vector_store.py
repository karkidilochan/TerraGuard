from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

"""Abstract class for vector databases."""


class VectorStore(ABC):

    @abstractmethod
    def store_documents(self, documents: List[Document]) -> None:
        """Persist a list of documents."""
        pass

    @abstractmethod
    def check_if_empty(self) -> bool:
        """Check if the store has been populated."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        """Retrieve top-k relevant documents."""
        pass
