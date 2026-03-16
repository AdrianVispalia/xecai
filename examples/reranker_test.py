import asyncio
from dotenv import load_dotenv
from xecai.models import Chunk
from xecai.reranker.implementations.aws_reranker import AWSReranker


async def main():
    reranker = AWSReranker()
    query = "What is the capital of France?"
    k = 3

    chunks = [
        Chunk(document="doc1", origin="Geography", fragment=0, content="Paris is the capital and most populous city of France."),
        Chunk(document="doc2", origin="Geography", fragment=0, content="London is the capital and largest city of England and the United Kingdom."),
        Chunk(document="doc3", origin="Geography", fragment=0, content="Berlin is the capital and largest city of Germany by both area and population."),
        Chunk(document="doc4", origin="History", fragment=0, content="The French Revolution was a period of radical political and societal change in France."),
        Chunk(document="doc5", origin="Cuisine", fragment=0, content="A baguette is a long, thin loaf of French bread that is commonly made from basic lean dough."),
        Chunk(document="doc6", origin="Sports", fragment=0, content="The Tour de France is an annual men's multiple-stage bicycle race primarily held in France."),
        Chunk(document="doc7", origin="Geography", fragment=0, content="The Seine is a 777-kilometre-long river in northern France."),
        Chunk(document="doc8", origin="Art", fragment=0, content="The Louvre is the world's most-visited museum, and a historic monument in Paris, France."),
        Chunk(document="doc9", origin="Landmarks", fragment=0, content="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."),
        Chunk(document="doc10", origin="Geography", fragment=0, content="Madrid is the capital and most populous city of Spain."),
    ]

    print(f"Query: '{query}'. Total input chunks: {len(chunks)}\n. Testing async_rerank for top {k} results...")

    async_reranked_chunks = await reranker.async_rerank(query=query, chunks=chunks, k=k)
    sync_reranked_chunks = reranker.sync_rerank(query=query, chunks=chunks, k=k)

    print("\n" + "="*50 + "\n")
    for case, reranked_chunks in [("async", async_reranked_chunks), ("sync", sync_reranked_chunks)]:
        print(f"\nTop {k} Reranked Chunks ({case}):")
        for i, chunk in enumerate(reranked_chunks, 1):
            score = chunk.score if chunk.score is not None else 0.0
            print(f"{i}. Score: {score:.4f} | Document: {chunk.document} | Content: {chunk.content}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
