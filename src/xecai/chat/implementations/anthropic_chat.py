import os
from collections.abc import AsyncIterator, Iterator
from typing import Final
from xecai.chat.chat_interface import ChatInterface
from xecai.models import (
    ChatResponse,
    Message,
    MessageType,
    ReasoningOptions,
    Stats,
    StopReason,
    ModelNotFoundError,
)
from xecai.error_handlers.anthropic_error_handler import (
    async_error_decorator,
    async_generator_error_decorator,
    sync_error_decorator,
    sync_generator_error_decorator,
)

try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError as e:
    raise RuntimeError(
        """Anthropic provider requires the 'anthropic' extra.
        Install with: uv pip install xecai[anthropic]"""
    ) from e


class AnthropicChat(ChatInterface):
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client: Final[Anthropic] = Anthropic(api_key=api_key)
        self.aclient: Final[AsyncAnthropic] = AsyncAnthropic(api_key=api_key)

    @sync_error_decorator
    def check_model(self, model_name: str) -> None:
        available_models = self.client.models.list()
        if not any(m.id == model_name for m in available_models.data):
            raise ModelNotFoundError(f"Model '{model_name}' not found.")

    @staticmethod
    def messages_to_custom_messages(
        system_prompt: str, messages: list[Message]
    ) -> list:
        contents = []
        for msg in messages:
            if msg.message_type == MessageType.BOT:
                role = "assistant"
            else:
                role = "user"

            contents.append({"role": role, "content": msg.content})
        return contents

    def _prepare_kwargs(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None,
        temperature: float | None,
        stream: bool = False,
        retries: int = 3,
    ) -> dict:

        kwargs = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": self.messages_to_custom_messages(system_prompt, messages),
            "system": system_prompt,
        }

        if stream:
            kwargs["stream"] = True

        if temperature is not None:
            kwargs["temperature"] = temperature

        if reasoning and reasoning != ReasoningOptions.NONE:
            budget_map = {
                ReasoningOptions.LOW: 1024,
                ReasoningOptions.MEDIUM: 4096,
                ReasoningOptions.HIGH: 16384,
            }
            budget = budget_map[reasoning]
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = budget + 4096
            kwargs["temperature"] = 1.0

        return kwargs

    def _map_stop_reason(self, stop_reason: str | None) -> StopReason:
        if stop_reason == "end_turn":
            return StopReason.END
        elif stop_reason == "max_tokens":
            return StopReason.MAX_TOKENS
        elif stop_reason == "stop_sequence":
            return StopReason.STOP_SEQUENCE
        elif stop_reason == "tool_use":
            return StopReason.TOOL_USE
        elif stop_reason == "content_filtered":
            return StopReason.CONTENT_FILTER
        elif stop_reason is None:
            return StopReason.END
        return StopReason.OTHER

    @sync_error_decorator
    def invoke(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> ChatResponse:
        kwargs = self._prepare_kwargs(
            model_name,
            system_prompt,
            messages,
            reasoning,
            temperature,
        )
        response = self.client.messages.create(**kwargs)
        usage = response.usage
        stats = (
            Stats(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=(getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0),
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                reasoning_tokens=0,
            )
            if usage
            else Stats()
        )
        text = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    text += block.text

        return ChatResponse(
            text=text,
            stats=stats,
            stop_reason=self._map_stop_reason(getattr(response, "stop_reason", None)),
        )

    @async_error_decorator
    async def async_invoke(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> ChatResponse:
        kwargs = self._prepare_kwargs(
            model_name,
            system_prompt,
            messages,
            reasoning,
            temperature,
        )
        response = await self.aclient.messages.create(**kwargs)
        usage = response.usage
        stats = (
            Stats(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=(getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0),
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                reasoning_tokens=0,
            )
            if usage
            else Stats()
        )
        text = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    text += block.text

        return ChatResponse(
            text=text,
            stats=stats,
            stop_reason=self._map_stop_reason(getattr(response, "stop_reason", None)),
        )

    @sync_generator_error_decorator
    def stream(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> Iterator[ChatResponse]:
        kwargs = self._prepare_kwargs(
            model_name,
            system_prompt,
            messages,
            reasoning,
            temperature,
            stream=True,
        )
        response = self.client.messages.create(**kwargs)

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        stop_reason = None

        for event in response:
            if event.type == "message_start":
                usage = getattr(event.message, "usage", None)
                if usage:
                    input_tokens += getattr(usage, "input_tokens", 0) or 0
                    cached_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens += getattr(usage, "output_tokens", 0) or 0
                if hasattr(event.delta, "stop_reason"):
                    stop_reason = event.delta.stop_reason
            elif (
                event.type == "content_block_delta" and event.delta.type == "text_delta"
            ):
                yield ChatResponse(text=event.delta.text)
            elif event.type == "message_stop":
                stats = Stats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cached_input_tokens=cached_tokens,
                    reasoning_tokens=0,
                )
                yield ChatResponse(
                    text="", stats=stats, stop_reason=self._map_stop_reason(stop_reason)
                )

    @async_generator_error_decorator
    async def async_stream(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        retries: int = 3,
    ) -> AsyncIterator[ChatResponse]:
        kwargs = self._prepare_kwargs(
            model_name,
            system_prompt,
            messages,
            reasoning,
            temperature,
            stream=True,
        )
        response = await self.aclient.messages.create(**kwargs)

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        stop_reason = None

        async for event in response:
            if event.type == "message_start":
                usage = getattr(event.message, "usage", None)
                if usage:
                    input_tokens += getattr(usage, "input_tokens", 0) or 0
                    cached_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens += getattr(usage, "output_tokens", 0) or 0
                if hasattr(event.delta, "stop_reason"):
                    stop_reason = event.delta.stop_reason
            elif (
                event.type == "content_block_delta" and event.delta.type == "text_delta"
            ):
                yield ChatResponse(text=event.delta.text)
            elif event.type == "message_stop":
                stats = Stats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    cached_input_tokens=cached_tokens,
                    reasoning_tokens=0,
                )
                yield ChatResponse(
                    text="", stats=stats, stop_reason=self._map_stop_reason(stop_reason)
                )
