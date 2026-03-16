from xecai.chat.chat_interface import ChatInterface
from xecai.models import Message, MessageType

DEFAULT_CONDENSE_PROMPT = """
    You are a summary assistant.
    You will receive a conversation between a human and a bot, and a new question.
    Your task is to create a new, independent question that takes all of the information
    from the context and does not miss key information needed in the question.
    """


def _prepare_condense_payload(
    previous_messages: list[Message], new_question: str
) -> list[Message] | None:
    """Prepares the message payload for the LLM. Returns None if no context exists."""
    if not previous_messages:
        return None

    condense_content = f"""
        Conversation:

        {"\n".join([msg.to_prompt_text() for msg in previous_messages])}

        Question: {new_question}
        """
    return [Message(message_type=MessageType.USER, content=condense_content)]


def sync_condense_question(
    condenser_chat: ChatInterface,
    condenser_model_name: str,
    previous_messages: list[Message],
    new_question: str,
    CONDENSE_PROMPT: str = DEFAULT_CONDENSE_PROMPT,
) -> str:
    messages = _prepare_condense_payload(previous_messages, new_question)
    if not messages:
        return new_question

    response = condenser_chat.invoke(condenser_model_name, CONDENSE_PROMPT, messages)
    return response.text


async def async_condense_question(
    condenser_chat: ChatInterface,
    condenser_model_name: str,
    previous_messages: list[Message],
    new_question: str,
    CONDENSE_PROMPT: str = DEFAULT_CONDENSE_PROMPT,
) -> str:
    messages = _prepare_condense_payload(previous_messages, new_question)
    if not messages:
        return new_question

    response = await condenser_chat.async_invoke(
        condenser_model_name, CONDENSE_PROMPT, messages
    )
    return response.text
