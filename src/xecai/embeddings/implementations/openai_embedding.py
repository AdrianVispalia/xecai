import os
from typing import Final
from xecai.embeddings.embedding_interface import EmbeddingInterface

try:
    import openai
except ImportError as e:
    raise RuntimeError(
        """OpenAI provider requires the 'openai' extra.
        Install with: uv pip install xecai[openai]"""
    ) from e


class OpenAIEmbedding(EmbeddingInterface):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client: Final[openai.OpenAI] = openai.OpenAI(api_key=api_key)
        self.aclient: Final[openai.AsyncOpenAI] = openai.AsyncOpenAI(api_key=api_key)

    def sync_get_embeddings(self, text: str, model: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding if response.data else []

    async def async_get_embeddings(self, text: str, model: str) -> list[float]:
        response = await self.aclient.embeddings.create(input=text, model=model)
        return response.data[0].embedding if response.data else []
