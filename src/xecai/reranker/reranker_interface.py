from abc import ABC, abstractmethod
from xecai.models import Chunk


class RerankerInterface(ABC):
    @abstractmethod
    def sync_rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        pass

    @abstractmethod
    async def async_rerank(
        self, query: str, chunks: list[Chunk], k: int
    ) -> list[Chunk]:
        pass
