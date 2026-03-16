from xecai.models import Chunk


def make_rag_prompt(query: str, chunks: list[Chunk]) -> str:
    context_str_parts: list[str] = []
    for chunk in chunks:
        part = f"Document: {chunk.document}\nContent: {chunk.content}"
        if chunk.score is not None:
            part += f"\nScore: {chunk.score}"
        context_str_parts.append(part)
    context_str = "\n\n".join(context_str_parts)
    return f"Context information:\n{context_str}\n\nGiven the context information, answer the following query: {query}"
