import os
from typing import Final
from xecai.embeddings.embedding_interface import EmbeddingInterface

try:
    from google import genai
except ImportError as e:
    raise RuntimeError(
        """Google provider requires the 'google' extra.
        Install with: uv pip install xecai[google]"""
    ) from e


class GoogleEmbedding(EmbeddingInterface):
    def __init__(self):
        self.client: Final[genai.Client] = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY")
        )

    def sync_get_embeddings(self, text: str, model: str) -> list[float]:
        response = self.client.models.embed_content(
            model=model,
            contents=text,
        )
        return response.embeddings[0].values if response.embeddings else []

    async def async_get_embeddings(self, text: str, model: str) -> list[float]:
        response = await self.client.aio.models.embed_content(
            model=model,
            contents=text,
        )
        return response.embeddings[0].values if response.embeddings else []
