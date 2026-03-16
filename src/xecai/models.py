from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MessageType(Enum):
    DEVELOPER = "DEVELOPER"
    USER = "USER"
    BOT = "BOT"


class ReasoningOptions(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StopReason(Enum):
    END = "end"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class SearchType(Enum):
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"


class Message(BaseModel):
    content: str
    message_type: MessageType

    def to_prompt_text(self) -> str:
        return f"{self.message_type.value}: {self.content}"


class Stats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    reasoning_tokens: int = 0


class ChatResponse(BaseModel):
    text: str
    stats: Stats | None = None
    stop_reason: StopReason | None = None


class Chunk(BaseModel):
    document: str
    origin: str | None = None
    fragment: int = 0
    content: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class Conversation(BaseModel):
    user: str | None = None
    conversation_id: UUID = Field(default_factory=uuid4)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CredentialsError(Exception):
    pass


class RateLimitError(Exception):
    pass


class BadRequestError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass
