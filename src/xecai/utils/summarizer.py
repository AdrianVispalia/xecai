from xecai.chat.chat_interface import ChatInterface
from xecai.memory.memory_interface import MemoryInterface
from xecai.models import Conversation, Message, MessageType

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation. Focus on the main topics and context."
)


def _prepare_summary_payload(conversation: Conversation) -> list[Message]:
    """Prepares the message payload for the summary LLM. Returns None if no conversation exists."""
    conversation_text = "\n".join(
        [msg.to_prompt_text() for msg in conversation.messages]
    )
    return [Message(message_type=MessageType.USER, content=conversation_text)]


def sync_summarize_conversation(
    memory: MemoryInterface,
    chat: ChatInterface,
    conversation_id: str,
    summary_model_name: str,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
) -> None:
    conversation = memory.sync_get_conversation(conversation_id)
    if not conversation or not conversation.messages:
        return

    messages = _prepare_summary_payload(conversation)
    response = chat.invoke(
        model_name=summary_model_name,
        system_prompt=summary_prompt,
        messages=messages,
    )

    conversation.messages = [
        Message(message_type=MessageType.BOT, content=response.text)
    ]
    memory.sync_save_conversation(conversation)


async def async_summarize_conversation(
    memory: MemoryInterface,
    chat: ChatInterface,
    conversation_id: str,
    summary_model_name: str,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
) -> None:
    conversation = await memory.async_get_conversation(conversation_id)
    if not conversation or not conversation.messages:
        return

    messages = _prepare_summary_payload(conversation)
    response = await chat.async_invoke(
        model_name=summary_model_name,
        system_prompt=summary_prompt,
        messages=messages,
    )

    conversation.messages = [
        Message(message_type=MessageType.BOT, content=response.text)
    ]
    await memory.async_save_conversation(conversation)
