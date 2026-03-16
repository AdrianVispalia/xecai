import os
from typing import Final
from xecai.memory.memory_interface import MemoryInterface
from xecai.models import Conversation

try:
    import redis
    import redis.asyncio as redis_async
except ImportError as e:
    raise RuntimeError(
        """Redis memory provider requires the 'redis' extra.
        Install with: uv pip install xecai[redis]"""
    ) from e


class RedisMemory(MemoryInterface):
    def __init__(self):
        self.redis_url: Final[str] = os.environ.get(
            "REDIS_URL", "redis://127.0.0.1:6379/0"
        )
        self.client: Final[redis.Redis] = redis.from_url(self.redis_url)
        self.async_client: Final[redis_async.Redis] = redis_async.from_url(
            self.redis_url
        )
        self.prefix: Final[str] = "conversation:"

    def _get_key(self, conversation_id: str) -> str:
        return f"{self.prefix}{conversation_id}"

    def sync_get_conversation(self, conversation_id: str) -> Conversation | None:
        data = self.client.get(self._get_key(conversation_id))
        if data:
            return Conversation.model_validate_json(data)
        return None

    async def async_get_conversation(self, conversation_id: str) -> Conversation | None:
        data = await self.async_client.get(self._get_key(conversation_id))
        if data:
            return Conversation.model_validate_json(data)
        return None

    def sync_save_conversation(self, conversation: Conversation) -> None:
        key = self._get_key(str(conversation.conversation_id))
        self.client.set(key, conversation.model_dump_json())

    async def async_save_conversation(self, conversation: Conversation) -> None:
        key = self._get_key(str(conversation.conversation_id))
        await self.async_client.set(key, conversation.model_dump_json())
