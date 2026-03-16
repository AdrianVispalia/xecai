from xecai.models import Message, MessageType


def trim_conversation(messages: list[Message], max_chars: int) -> list[Message]:
    result_messages: list[Message] = []
    result_chars = 0
    for msg in reversed(messages):
        msg_len = len(msg.content)
        if result_chars + msg_len > max_chars:
            break

        result_chars += msg_len
        result_messages.insert(0, msg)

    while result_messages and result_messages[0].message_type != MessageType.USER:
        result_messages.pop(0)

    return result_messages
