import asyncio

from dotenv import load_dotenv

from xecai.embeddings.embedding_interface import EmbeddingInterface
from xecai.embeddings.implementations.aws_embedding import AWSEmbedding
from xecai.embeddings.implementations.google_embedding import GoogleEmbedding
from xecai.embeddings.implementations.openai_embedding import OpenAIEmbedding


async def tester(embedding_provider: EmbeddingInterface, model: str):
    text = "This is a test document to embed."

    print(f"Generating embedding for text: '{text}' using model: {model}")

    # Test sync method
    print("\n--- Sync ---")
    sync_embedding = embedding_provider.sync_get_embeddings(text, model)
    print(f"Sync embedding generated. Length: {len(sync_embedding)}")
    print(f"First 5 values: {sync_embedding[:5]}")

    # Test async method
    print("\n--- Async ---")
    async_embedding = await embedding_provider.async_get_embeddings(text, model)
    print(f"Async embedding generated. Length: {len(async_embedding)}")
    print(f"First 5 values: {async_embedding[:5]}")


if __name__ == "__main__":
    load_dotenv()
    provider = input("Test provider (aws/google/openai): ").lower().strip()

    match provider:
        case "aws":
            embedding = AWSEmbedding()
            # Valid models include titan or cohere
            model = "amazon.titan-embed-text-v2:0"
        case "google":
            embedding = GoogleEmbedding()
            model = "gemini-embedding-001"
        case "openai":
            embedding = OpenAIEmbedding()
            model = "text-embedding-3-small"
        case _:
            print(f"Unknown provider '{provider}', defaulting to openai.")
            embedding = OpenAIEmbedding()
            model = "text-embedding-3-small"

    asyncio.run(tester(embedding, model))
