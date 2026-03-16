from abc import ABC, abstractmethod
from xecai.models import Chunk, SearchType


class VectorDBInterface(ABC):
    @abstractmethod
    def sync_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        pass

    @abstractmethod
    async def async_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        pass

    @abstractmethod
    def sync_insert(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    async def async_insert(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def sync_get_num_documents(self) -> int:
        pass

    @abstractmethod
    async def async_get_num_documents(self) -> int:
        pass
