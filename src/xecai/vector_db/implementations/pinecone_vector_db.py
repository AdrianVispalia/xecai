import os
import uuid
from typing import Final
from xecai.models import Chunk, SearchType
from xecai.vector_db.vector_db_interface import VectorDBInterface

try:
    import pinecone
except ImportError as e:
    raise RuntimeError(
        """Pinecone vector DB provider requires the 'pinecone' extra.
        Install with: uv pip install xecai[pinecone]"""
    ) from e


class PineconeVectorDB(VectorDBInterface):
    """
    Pinecone vector database implementation.

    Pinecone is a managed vector database service. It requires an API key and
    an index name. This implementation uses Pinecone's Integrated Inference
    so you do not need to provide an embedding interface.
    """

    def __init__(
        self,
        index_name: str = "index",
        namespace: str = "",
    ):
        self.api_key: Final[str] = os.environ.get("PINECONE_API_KEY", "")
        self.index_name: Final[str] = os.environ.get("PINECONE_INDEX_NAME", index_name)
        # Using "__default__" for empty namespace per new SDK requirements
        self.namespace: Final[str] = namespace if namespace else "__default__"

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")

        self.pc = pinecone.Pinecone(api_key=self.api_key)

        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            raise ValueError(
                f"Pinecone index '{self.index_name}' does not exist. Please create it in the Pinecone console or via the API."
            )

        self.index = self.pc.Index(self.index_name)

    def sync_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        # Using integrated inference search
        response = self.index.search(
            namespace=self.namespace, query={"inputs": {"text": query}, "top_k": k}
        )

        results = []
        hits = response.get("result", {}).get("hits", [])
        for hit in hits:
            fields = hit.get("fields", {})
            results.append(
                Chunk(
                    document=fields.get("document", ""),
                    origin=fields.get("origin", ""),
                    fragment=int(fields.get("fragment", 0)),
                    content=fields.get("text", ""),
                    score=hit.get("_score"),
                )
            )

        return results

    async def async_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> list[Chunk]:
        # Pinecone SDK is primarily synchronous
        return self.sync_retrieve(query, k, search_type)

    def sync_insert(self, chunks: list[Chunk]) -> None:
        records_to_upsert = []
        for chunk in chunks:
            record_id = str(uuid.uuid4())

            record = {
                "_id": record_id,
                "text": chunk.content,
                "document": chunk.document,
                "origin": chunk.origin or "",
                "fragment": chunk.fragment,
            }

            records_to_upsert.append(record)

        # Upsert in batches of 96 as recommended for integrated inference models
        batch_size = 96
        for i in range(0, len(records_to_upsert), batch_size):
            batch = records_to_upsert[i : i + batch_size]
            self.index.upsert_records(namespace=self.namespace, records=batch)

    async def async_insert(self, chunks: list[Chunk]) -> None:
        self.sync_insert(chunks)

    def sync_get_num_documents(self) -> int:
        stats = self.index.describe_index_stats()
        if self.namespace:
            return (
                stats.get("namespaces", {})
                .get(self.namespace, {})
                .get("vector_count", 0)
            )
        return stats.get("total_vector_count", 0)

    async def async_get_num_documents(self) -> int:
        return self.sync_get_num_documents()
