import os
from typing import Final, List, Dict, Any

from xecai.models import Chunk
from xecai.reranker.reranker_interface import RerankerInterface
from xecai.error_handlers.aws_error_handler import (
    sync_error_decorator,
    async_error_decorator,
)

try:
    import boto3
    from aiobotocore.session import AioSession, get_session
except ImportError as e:
    raise RuntimeError(
        "AWS Bedrock provider requires the 'aws' extra.\n"
        "Install with: uv pip install xecai[aws]"
    ) from e


class AWSReranker(RerankerInterface):
    _SERVICE_NAME: Final[str] = "bedrock-agent-runtime"

    def __init__(
        self,
        region_name: str | None = None,
        model_arn: str | None = None,
    ) -> None:
        self.region_name: Final[str] = region_name or os.getenv("AWS_REGION", "us-east-1")
        default_model = (
            f"arn:aws:bedrock:{self.region_name}"
            "::foundation-model/cohere.rerank-v3-5:0"
        )
        self.model_arn: Final[str] = model_arn or os.getenv(
            "AWS_RERANK_MODEL_ARN",
            default_model,
        )
        self.client: Final = boto3.client(
            self._SERVICE_NAME,
            region_name=self.region_name,
        )
        self._aiosession: Final[AioSession] = get_session()

    @staticmethod
    def _prepare_sources(chunks: List[Chunk]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": chunk.content},
                },
            }
            for chunk in chunks
        ]

    @staticmethod
    def _map_response(response: Dict[str, Any], chunks: List[Chunk]) -> List[Chunk]:
        results = response.get("results", [])
        reranked: List[Chunk] = []

        for result in results:
            idx = result.get("index")
            score = result.get("relevanceScore")

            if idx is not None and 0 <= idx < len(chunks):
                reranked.append(
                    chunks[idx].model_copy(update={"score": score})
                )

        return reranked

    def _build_request(
        self,
        query: str,
        chunks: List[Chunk],
        k: int,
    ) -> Dict[str, Any]:

        return {
            "queries": [
                {
                    "type": "TEXT",
                    "textQuery": {"text": query},
                }
            ],
            "sources": self._prepare_sources(chunks),
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(k, len(chunks)),
                    "modelConfiguration": {"modelArn": self.model_arn},
                },
            },
        }


    @sync_error_decorator
    def sync_rerank(
        self,
        query: str,
        chunks: List[Chunk],
        k: int,
    ) -> List[Chunk]:
        if not chunks:
            return []

        request = self._build_request(query, chunks, k)
        response = self.client.rerank(**request)
        return self._map_response(response, chunks)


    @async_error_decorator
    async def async_rerank(
        self,
        query: str,
        chunks: List[Chunk],
        k: int,
    ) -> List[Chunk]:

        if not chunks:
            return []

        request = self._build_request(query, chunks, k)
        async with self._aiosession.create_client(
            self._SERVICE_NAME,
            region_name=self.region_name,
        ) as client:
            response = await client.rerank(**request)

        return self._map_response(response, chunks)
