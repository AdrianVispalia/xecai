import os
from typing import Final
from xecai.embeddings.embedding_interface import EmbeddingInterface
from xecai.models import Chunk, SearchType
from xecai.vector_db.vector_db_interface import VectorDBInterface

try:
    import psycopg
    from pgvector.psycopg import register_vector, register_vector_async
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
except ImportError as e:
    raise RuntimeError(
        """PostgreSQL vector DB provider requires the 'postgresql' extra.
        Install with: uv pip install xecai[postgresql]"""
    ) from e


class PostgreSQLVectorDB(VectorDBInterface):
    """
    PostgreSQL vector database implementation using the pgvector extension.

    Since PostgreSQL doesn't inherently generate embeddings from text queries
    like managed services (e.g. AWS Bedrock Knowledge Bases), you must provide
    an embedding interface and model during initialization.
    """

    def __init__(
        self,
        embedding_interface: EmbeddingInterface | None = None,
        embedding_model: str | None = None,
        table_name: str = "vector_documents",
        vector_dim: int = 1536,
    ):
        if not embedding_interface or not embedding_model:
            raise ValueError(
                "embedding_interface and embedding_model must be provided during PostgreSQLVectorDB initialization."
            )

        self.pg_url: Final[str] = os.environ.get(
            "POSTGRESQL_URL", "postgresql://postgres:postgres@localhost:5432/xecai"
        )
        self.table_name: Final[str] = table_name
        self.vector_dim: Final[int] = vector_dim

        self.embedding_interface = embedding_interface
        self.embedding_model = embedding_model

        self._pool: ConnectionPool | None = None
        self._async_pool: AsyncConnectionPool | None = None
        self._ensure_table_exists()

    def _configure_sync_conn(self, conn: psycopg.Connection):
        register_vector(conn)

    async def _configure_async_conn(self, conn: psycopg.AsyncConnection):
        await register_vector_async(conn)

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self.pg_url, configure=self._configure_sync_conn
            )
        return self._pool

    async def _get_async_pool(self) -> AsyncConnectionPool:
        if self._async_pool is None:
            self._async_pool = AsyncConnectionPool(
                self.pg_url, configure=self._configure_async_conn, open=False
            )
            await self._async_pool.open()
        return self._async_pool

    def _ensure_table_exists(self) -> None:
        query_ext = "CREATE EXTENSION IF NOT EXISTS vector;"
        query_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document TEXT NOT NULL,
            origin TEXT,
            fragment INTEGER DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector({self.vector_dim})
        );
        """
        with psycopg.connect(self.pg_url) as conn:
            # Create extension first before registering the vector type
            with conn.cursor() as cur:
                cur.execute(query_ext)
            conn.commit()

            # Now that the type exists in the database, register it with psycopg
            self._configure_sync_conn(conn)

            # Create the table
            with conn.cursor() as cur:
                cur.execute(query_table)
            conn.commit()

    def _build_semantic_query(self) -> str:
        return f"""
            SELECT document, origin, fragment, content, 1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

    def _build_hybrid_query(self) -> str:
        # Standard Reciprocal Rank Fusion (RRF) for Hybrid Search
        return f"""
            WITH semantic_search AS (
                SELECT id, document, origin, fragment, content,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            ),
            keyword_search AS (
                SELECT id, document, origin, fragment, content,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC) AS rank
                FROM {self.table_name}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY rank
                LIMIT 100
            )
            SELECT COALESCE(s.document, k.document) as document,
                   COALESCE(s.origin, k.origin) as origin,
                   COALESCE(s.fragment, k.fragment) as fragment,
                   COALESCE(s.content, k.content) as content,
                   COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
            FROM semantic_search s
            FULL OUTER JOIN keyword_search k ON s.id = k.id
            ORDER BY rrf_score DESC
            LIMIT %s;
        """

    def _map_row_to_chunk(self, row: tuple) -> Chunk:
        return Chunk(
            document=row[0],
            origin=row[1],
            fragment=row[2] or 0,
            content=row[3],
            score=row[4] if len(row) > 4 else None,
        )

    def sync_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        query_embedding = self.embedding_interface.sync_get_embeddings(
            query, self.embedding_model
        )

        results = []
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                if search_type == SearchType.SEMANTIC:
                    cur.execute(
                        self._build_semantic_query(),
                        (query_embedding, query_embedding, k),
                    )
                    for row in cur.fetchall():
                        results.append(self._map_row_to_chunk(row))
                elif search_type == SearchType.HYBRID:
                    cur.execute(
                        self._build_hybrid_query(),
                        (query_embedding, query_embedding, query, query, k),
                    )
                    for row in cur.fetchall():
                        results.append(self._map_row_to_chunk(row))

        return results

    async def async_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        query_embedding = await self.embedding_interface.async_get_embeddings(
            query, self.embedding_model
        )

        results = []
        pool = await self._get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if search_type == SearchType.SEMANTIC:
                    await cur.execute(
                        self._build_semantic_query(),
                        (query_embedding, query_embedding, k),
                    )
                    for row in await cur.fetchall():
                        results.append(self._map_row_to_chunk(row))
                elif search_type == SearchType.HYBRID:
                    await cur.execute(
                        self._build_hybrid_query(),
                        (query_embedding, query_embedding, query, query, k),
                    )
                    for row in await cur.fetchall():
                        results.append(self._map_row_to_chunk(row))

        return results

    def sync_insert(self, chunks: list[Chunk]) -> None:
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    embedding = self.embedding_interface.sync_get_embeddings(
                        chunk.content, self.embedding_model
                    )
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name} (document, origin, fragment, content, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """,
                        (
                            chunk.document,
                            chunk.origin,
                            chunk.fragment,
                            chunk.content,
                            embedding,
                        ),
                    )
            conn.commit()

    async def async_insert(self, chunks: list[Chunk]) -> None:
        pool = await self._get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk in chunks:
                    embedding = await self.embedding_interface.async_get_embeddings(
                        chunk.content, self.embedding_model
                    )
                    await cur.execute(
                        f"""
                        INSERT INTO {self.table_name} (document, origin, fragment, content, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """,
                        (
                            chunk.document,
                            chunk.origin,
                            chunk.fragment,
                            chunk.content,
                            embedding,
                        ),
                    )
            await conn.commit()

    def sync_get_num_documents(self) -> int:
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]

    async def async_get_num_documents(self) -> int:
        pool = await self._get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return (await cur.fetchone())[0]
