# Before running: uv pip install uvicorn fastapi

from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from xecai.chat.implementations.openai_chat import OpenAIChat
from xecai.embeddings.implementations.openai_embedding import OpenAIEmbedding
from xecai.memory.implementations.debug_memory import DebugMemory
from xecai.models import Chunk, Conversation, Message, MessageType, SearchType
from xecai.utils.condenser import async_condense_question
from xecai.utils.rag_prompt import make_rag_prompt
from xecai.vector_db.implementations.debug_vector_db import DebugVectorDB

load_dotenv()

chat = OpenAIChat()
vector_db = DebugVectorDB(
    embedding_interface=OpenAIEmbedding(), embedding_model="text-embedding-3-small"
)
memory = DebugMemory()


class ChatRequest(BaseModel):
    query: str
    conversationId: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = await vector_db.async_get_num_documents()
    if count == 0:
        print("Inserting dummy chunks...")
        chunks_to_insert = [
            Chunk(
                document="Meaning of Life",
                origin="Philosophy",
                fragment=0,
                content="42 is the meaning of life. According to the supercomputer Deep Thought, after calculating for 7.5 million years, this is the ultimate answer to life, the universe, and everything. Though the question itself remains unknown.",
            ),
            Chunk(
                document="Moon Satellite",
                origin="Astronomy",
                fragment=0,
                content="The moon is our satellite. It is the fifth largest satellite in the solar system and the largest relative to the size of its planet. It has a significant effect on Earth's tides and has been the only celestial body visited by humans.",
            ),
            Chunk(
                document="Komodo Dragon",
                origin="Biology",
                fragment=0,
                content="The Komodo dragon is the largest existing reptile species. Found in the Indonesian islands, these massive lizards can grow up to 3 meters in length and weigh over 70 kilograms. They are apex predators in their ecosystem with a venomous bite.",
            ),
        ]
        await vector_db.async_insert(chunks_to_insert)
        print("Chunks inserted!")

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Get previous conversation messages or create a new conversation
    conversation = (
        request.conversationId
        and await memory.async_get_conversation(request.conversationId)
    ) or Conversation()

    llm_model = "gpt-4o"
    # Condense question, add context from previous messages
    question = await async_condense_question(
        chat, llm_model, conversation.messages, request.query
    )

    chunks = await vector_db.async_retrieve(
        query=question,
        k=3,
        search_type=SearchType.HYBRID,
    )

    response = await chat.async_invoke(
        model_name=llm_model,
        system_prompt="You are a helpful and informative assistant.",
        messages=[
            Message(
                content=make_rag_prompt(question, chunks), message_type=MessageType.USER
            )
        ],
    )

    conversation.messages.extend(
        [
            Message(content=request.query, message_type=MessageType.USER),
            Message(content=response.text, message_type=MessageType.BOT),
        ]
    )
    await memory.async_save_conversation(conversation)

    return {
        "answer": response.text,
        "chunks": [chunk.model_dump() for chunk in chunks],
        "conversationId": str(conversation.conversation_id),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
