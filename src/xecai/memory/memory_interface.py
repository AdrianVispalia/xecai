from abc import ABC, abstractmethod
from xecai.models import Conversation


class MemoryInterface(ABC):
    @abstractmethod
    def sync_get_conversation(self, conversation_id: str) -> Conversation | None:
        pass

    @abstractmethod
    async def async_get_conversation(self, conversation_id: str) -> Conversation | None:
        pass

    @abstractmethod
    def sync_save_conversation(self, conversation: Conversation) -> None:
        pass

    @abstractmethod
    async def async_save_conversation(self, conversation: Conversation) -> None:
        pass
