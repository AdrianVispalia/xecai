import json
import os
from typing import Any, Final
from xecai.embeddings.embedding_interface import EmbeddingInterface
from xecai.error_handlers.aws_error_handler import async_error_decorator, sync_error_decorator

try:
    import boto3
    from aiobotocore.session import AioSession, get_session
except ImportError as e:
    raise RuntimeError(
        """AWS Bedrock provider requires the 'aws' extra.
        Install with: uv pip install xecai[aws]"""
    ) from e


class AWSEmbedding(EmbeddingInterface):
    def __init__(self):
        self.region_name: Final[str] = os.environ.get("AWS_REGION", "us-east-1")
        self.client: Final[boto3.session.Session.client] = boto3.client(
            "bedrock-runtime", region_name=self.region_name
        )
        self.aiosession: Final[AioSession] = get_session()

    def _invoke_sync(self, model: str, body: dict) -> dict[str, Any]:
        response = self.client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response.get("body").read())

    async def _invoke_async(
        self, client: Any, model: str, body: dict
    ) -> dict[str, Any]:
        response = await client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(await response.get("body").read())

    @sync_error_decorator
    def sync_get_embeddings(self, text: str, model: str) -> list[float]:
        if "cohere" in model.lower():
            body = {"texts": [text], "input_type": "search_document"}
            response_body = self._invoke_sync(model, body)
            embeddings = response_body.get("embeddings", [])
            return embeddings[0] if embeddings else []

        elif "titan" in model.lower():
            body = {"inputText": text}
            response_body = self._invoke_sync(model, body)
            return response_body.get("embedding", [])

        else:
            raise ValueError(f"Unsupported AWS embedding model family: {model}")

    @async_error_decorator
    async def async_get_embeddings(self, text: str, model: str) -> list[float]:
        async with self.aiosession.create_client(
            "bedrock-runtime", region_name=self.region_name
        ) as client:
            if "cohere" in model.lower():
                body = {"texts": [text], "input_type": "search_document"}
                response_body = await self._invoke_async(client, model, body)
                embeddings = response_body.get("embeddings", [])
                return embeddings[0] if embeddings else []

            elif "titan" in model.lower():
                body = {"inputText": text}
                response_body = await self._invoke_async(client, model, body)
                return response_body.get("embedding", [])

            else:
                raise ValueError(f"Unsupported AWS embedding model family: {model}")
