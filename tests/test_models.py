import pytest
from xecai.models import Message, MessageType

def test_message_creation():
    msg = Message(content="hello", message_type=MessageType.USER)
    assert msg.content == "hello"
    assert msg.message_type == MessageType.USER
