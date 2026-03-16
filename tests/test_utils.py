import pytest
from xecai.utils.trim_conversation import trim_conversation
from xecai.utils.rag_prompt import make_rag_prompt
from xecai.utils.condenser import sync_condense_question
from xecai.utils.summarizer import _prepare_summary_payload, sync_summarize_conversation
from xecai.models import Message, MessageType, Chunk, Conversation

def test_trim_conversation_basic():
    conversation = ["hello", "world"]
    trimmed = trim_conversation(conversation, max_length=1)
    assert isinstance(trimmed, list)
    assert len(trimmed) <= 1

def test_trim_conversation_with_max_chars():
    messages = [Message(content="hello", message_type=MessageType.USER), Message(content="world", message_type=MessageType.BOT)]
    trimmed = trim_conversation(messages, max_chars=5)
    assert isinstance(trimmed, list)
    assert len(trimmed) == 1
    assert trimmed[0].content == "hello"

def test_make_rag_prompt():
    chunk = Chunk(document="doc1", content="some content", score=0.9)
    prompt = make_rag_prompt("What is this?", [chunk])
    assert "doc1" in prompt
    assert "some content" in prompt
    assert "Score: 0.9" in prompt
    assert "What is this?" in prompt

def test_sync_condense_question_returns_new_question(monkeypatch):
    class DummyChat:
        def invoke(self, model, prompt, messages):
            class Response:
                text = "condensed question"
            return Response()
    previous = [Message(content="context", message_type=MessageType.USER)]
    new_question = "What is up?"
    result = sync_condense_question(DummyChat(), "model", previous, new_question)
    assert result == "condensed question"

def test_prepare_summary_payload():
    conv = Conversation(messages=[Message(content="hello", message_type=MessageType.USER)])
    payload = _prepare_summary_payload(conv)
    assert isinstance(payload, list)
    assert payload[0].content == "hello"

def test_sync_summarize_conversation(monkeypatch):
    class DummyMemory:
        def sync_get_conversation(self, conversation_id):
            return Conversation(messages=[Message(content="hi", message_type=MessageType.USER)])
        def sync_save_conversation(self, conversation):
            self.saved = conversation
    class DummyChat:
        def invoke(self, model_name, system_prompt, messages):
            class Response:
                text = "summary"
            return Response()
    memory = DummyMemory()
    chat = DummyChat()
    sync_summarize_conversation(memory, chat, "id", "model")
    assert hasattr(memory, "saved")
    assert memory.saved.messages[0].content == "summary"
