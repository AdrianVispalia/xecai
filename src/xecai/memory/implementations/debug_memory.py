import logging
from typing import Dict
from xecai.memory.memory_interface import MemoryInterface
from xecai.models import Conversation


class DebugMemory(MemoryInterface):
    """
    Debug implementation of MemoryInterface that stores conversations in an in-memory dictionary.
    """

    def __init__(self):
        logging.warning(
            "DebugMemory is for debugging purposes only and is not recommended for production usecases."
        )
        self._storage: Dict[str, Conversation] = {}

    def sync_get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._storage.get(str(conversation_id))

    async def async_get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.sync_get_conversation(conversation_id)

    def sync_save_conversation(self, conversation: Conversation) -> None:
        self._storage[str(conversation.conversation_id)] = conversation

    async def async_save_conversation(self, conversation: Conversation) -> None:
        self.sync_save_conversation(conversation)
