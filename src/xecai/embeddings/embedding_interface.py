from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    @abstractmethod
    def sync_get_embeddings(self, text: str, model: str) -> list[float]:
        pass

    @abstractmethod
    async def async_get_embeddings(self, text: str, model: str) -> list[float]:
        pass
