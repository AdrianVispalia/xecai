from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any
from xecai.models import ChatResponse, Message, ReasoningOptions


class ChatInterface(ABC):
    @abstractmethod
    def check_model(self, model_name: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def messages_to_custom_messages(system_prompt: str, messages: list[Message]) -> Any:
        pass

    @abstractmethod
    def invoke(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> ChatResponse:
        pass

    @abstractmethod
    async def async_invoke(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> ChatResponse:
        pass

    @abstractmethod
    def stream(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> Iterator[ChatResponse]:
        yield ChatResponse(text="")

    @abstractmethod
    async def async_stream(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> AsyncIterator[ChatResponse]:
        yield ChatResponse(text="")
