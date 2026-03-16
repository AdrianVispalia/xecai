import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Final
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
from xecai.error_handlers.openai_error_handler import (
    async_error_decorator,
    async_generator_error_decorator,
    sync_error_decorator,
    sync_generator_error_decorator,
)

try:
    import openai
except ImportError as e:
    raise RuntimeError(
        """OpenAI provider requires the 'openai' extra.
        Install with: uv pip install xecai[openai]"""
    ) from e


class OpenAIChat(ChatInterface):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client: Final[openai.OpenAI] = openai.OpenAI(api_key=api_key)
        self.aclient: Final[openai.AsyncOpenAI] = openai.AsyncOpenAI(api_key=api_key)

    @staticmethod
    def messages_to_custom_messages(
        system_prompt: str, messages: list[Message]
    ) -> list[dict]:
        inputs = []

        if system_prompt:
            inputs.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        for msg in messages:
            if msg.message_type == MessageType.BOT:
                role = "assistant"
            elif msg.message_type == MessageType.DEVELOPER:
                role = "developer"
            else:
                role = "user"

            inputs.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": msg.content}],
                }
            )

        return inputs

    @sync_error_decorator
    def check_model(self, model_name: str) -> None:
        available_models = self.client.models.list()
        if not any(m.id == model_name for m in available_models.data):
            raise ModelNotFoundError(f"Model '{model_name}' not found.")

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
            "input": self.messages_to_custom_messages(system_prompt, messages),
        }

        if stream:
            kwargs["stream"] = True

        if temperature is not None:
            kwargs["temperature"] = temperature

        if reasoning and reasoning != ReasoningOptions.NONE:
            kwargs["reasoning"] = {"effort": reasoning.value}

        return kwargs

    def _map_stop_reason(
        self, status: str | None, incomplete_details: Any | None
    ) -> StopReason:
        if status == "completed":
            return StopReason.END
        elif status == "incomplete":
            reason = (
                getattr(incomplete_details, "reason", None)
                if incomplete_details
                else None
            )
            if reason == "max_output_tokens":
                return StopReason.MAX_TOKENS
            elif reason == "content_filter":
                return StopReason.CONTENT_FILTER
            elif reason == "stop":
                return StopReason.STOP_SEQUENCE
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
        response = self.client.responses.create(**kwargs)
        usage = response.usage
        stats = (
            Stats(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cached_input_tokens=getattr(
                    getattr(usage, "input_tokens_details", None), "cached_tokens", 0
                )
                or 0,
                reasoning_tokens=getattr(
                    getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0
                )
                or 0,
            )
            if usage
            else Stats()
        )

        status = response.status
        if status == "completed" and response.output:
            status = response.output[0].status

        stop_reason = self._map_stop_reason(
            status, getattr(response, "incomplete_details", None)
        )

        return ChatResponse(
            text=response.output_text or "", stats=stats, stop_reason=stop_reason
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
        response = await self.aclient.responses.create(**kwargs)

        usage = response.usage
        stats = (
            Stats(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cached_input_tokens=getattr(
                    getattr(usage, "input_tokens_details", None), "cached_tokens", 0
                )
                or 0,
                reasoning_tokens=getattr(
                    getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0
                )
                or 0,
            )
            if usage
            else Stats()
        )

        status = response.status
        if status == "completed" and response.output:
            status = response.output[0].status

        stop_reason = self._map_stop_reason(
            status, getattr(response, "incomplete_details", None)
        )

        return ChatResponse(
            text=response.output_text or "", stats=stats, stop_reason=stop_reason
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
        response = self.client.responses.create(**kwargs)

        for event in response:
            if event.type == "response.output_text.delta":
                yield ChatResponse(text=event.delta)
            elif event.type == "response.completed" and hasattr(event, "response"):
                usage = getattr(event.response, "usage", None)
                stats = (
                    Stats(
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        cached_input_tokens=getattr(
                            getattr(usage, "input_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0,
                        reasoning_tokens=getattr(
                            getattr(usage, "output_tokens_details", None),
                            "reasoning_tokens",
                            0,
                        )
                        or 0,
                    )
                    if usage
                    else Stats()
                )

                status = getattr(event.response, "status", None)
                stop_reason = self._map_stop_reason(
                    status, getattr(event.response, "incomplete_details", None)
                )

                yield ChatResponse(text="", stats=stats, stop_reason=stop_reason)

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
        response = await self.aclient.responses.create(**kwargs)

        async for event in response:
            if event.type == "response.output_text.delta":
                yield ChatResponse(text=event.delta)
            elif event.type == "response.completed" and hasattr(event, "response"):
                usage = getattr(event.response, "usage", None)
                stats = (
                    Stats(
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        cached_input_tokens=getattr(
                            getattr(usage, "input_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0,
                        reasoning_tokens=getattr(
                            getattr(usage, "output_tokens_details", None),
                            "reasoning_tokens",
                            0,
                        )
                        or 0,
                    )
                    if usage
                    else Stats()
                )

                status = getattr(event.response, "status", None)
                stop_reason = self._map_stop_reason(
                    status, getattr(event.response, "incomplete_details", None)
                )

                yield ChatResponse(text="", stats=stats, stop_reason=stop_reason)
