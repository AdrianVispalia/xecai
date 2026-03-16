import os
from typing import Final
from xecai.memory.memory_interface import MemoryInterface
from xecai.models import Conversation

try:
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
except ImportError as e:
    raise RuntimeError(
        """PostgreSQL memory provider requires the 'postgresql' extra.
        Install with: uv pip install xecai[postgresql]"""
    ) from e


class PostgreSQLMemory(MemoryInterface):
    def __init__(self):
        self.pg_url: Final[str] = os.environ.get(
            "POSTGRESQL_URL", "postgresql://postgres:postgres@127.0.0.1:5432/xecai"
        )
        self._pool: ConnectionPool | None = None
        self._async_pool: AsyncConnectionPool | None = None
        self._ensure_table_exists()

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(self.pg_url)
        return self._pool

    async def _get_async_pool(self) -> AsyncConnectionPool:
        if self._async_pool is None:
            self._async_pool = AsyncConnectionPool(self.pg_url, open=False)
            await self._async_pool.open()
        return self._async_pool

    def _ensure_table_exists(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            data JSONB NOT NULL
        );
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def sync_get_conversation(self, conversation_id: str) -> Conversation | None:
        query = "SELECT data FROM conversations WHERE id = %s;"
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (conversation_id,))
                result = cur.fetchone()
                if result:
                    data = result[0]
                    if isinstance(data, str):
                        return Conversation.model_validate_json(data)
                    return Conversation.model_validate(data)
        return None

    async def async_get_conversation(self, conversation_id: str) -> Conversation | None:
        query = "SELECT data FROM conversations WHERE id = %s;"

        # 2. Await the pool fetching
        pool = await self._get_async_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (conversation_id,))
                result = await cur.fetchone()
                if result:
                    data = result[0]
                    if isinstance(data, str):
                        return Conversation.model_validate_json(data)
                    return Conversation.model_validate(data)
        return None

    def sync_save_conversation(self, conversation: Conversation) -> None:
        query = """
        INSERT INTO conversations (id, data)
        VALUES (%s, %s)
        ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data;
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        str(conversation.conversation_id),
                        conversation.model_dump_json(),
                    ),
                )
            conn.commit()

    async def async_save_conversation(self, conversation: Conversation) -> None:
        query = """
        INSERT INTO conversations (id, data)
        VALUES (%s, %s)
        ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data;
        """

        # 2. Await the pool fetching
        pool = await self._get_async_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (
                        str(conversation.conversation_id),
                        conversation.model_dump_json(),
                    ),
                )
            await conn.commit()
