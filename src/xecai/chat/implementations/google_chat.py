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

from xecai.error_handlers.google_error_handler import (
    async_error_decorator,
    async_generator_error_decorator,
    sync_error_decorator,
    sync_generator_error_decorator,
)

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise RuntimeError(
        """Google provider requires the 'google' extra.
        Install with: uv pip install xecai[google]"""
    ) from e


class GoogleChat(ChatInterface):
    def __init__(self):
        self.client: Final[genai.Client] = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY")
        )

    @sync_error_decorator
    def check_model(self, model_name: str) -> None:
        available_models = list(self.client.models.list())
        # Google model names usually have a "models/" prefix
        if not any(
            m.name == model_name or m.name == f"models/{model_name}"
            for m in available_models
        ):
            raise ModelNotFoundError(f"Model '{model_name}' not found.")

    @staticmethod
    def messages_to_custom_messages(
        system_prompt: str, messages: list[Message]
    ) -> list:
        inputs = []
        for msg in messages:
            if msg.message_type == MessageType.BOT:
                role = "model"
            else:
                role = "user"

            inputs.append({"role": role, "parts": [{"text": msg.content}]})
        return inputs

    def _prepare_config(
        self,
        system_prompt: str,
        reasoning: ReasoningOptions | None,
        temperature: float | None,
        retries: int = 3,
    ) -> types.GenerateContentConfig:
        kwargs = {"system_instruction": system_prompt}

        if temperature is not None:
            kwargs["temperature"] = temperature

        if reasoning and reasoning != ReasoningOptions.NONE:
            budget_map = {
                ReasoningOptions.LOW: 1024,
                ReasoningOptions.MEDIUM: 4096,
                ReasoningOptions.HIGH: 16384,
            }
            kwargs["thinking_config"] = {"thinking_budget": budget_map[reasoning]}

        return types.GenerateContentConfig(**kwargs)

    def _map_stop_reason(self, finish_reason: str | int | None) -> StopReason:
        if finish_reason is None:
            return StopReason.END

        # Convert to string to handle enum names or raw values
        reason_str = str(finish_reason).upper()

        if "STOP" in reason_str:
            return StopReason.END
        elif "MAX_TOKENS" in reason_str:
            return StopReason.MAX_TOKENS
        elif (
            "SAFETY" in reason_str
            or "BLOCKLIST" in reason_str
            or "PROHIBITED_CONTENT" in reason_str
            or "SPII" in reason_str
        ):
            return StopReason.CONTENT_FILTER
        elif "UNEXPECTED_TOOL_CALL" in reason_str:
            return StopReason.TOOL_USE

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
        response = self.client.models.generate_content(
            model=model_name,
            contents=self.messages_to_custom_messages(system_prompt, messages),
            config=self._prepare_config(
                system_prompt, reasoning, temperature
            ),
        )

        usage = response.usage_metadata
        stats = (
            Stats(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage, "total_token_count", 0) or 0,
                cached_input_tokens=getattr(usage, "cached_content_token_count", 0)
                or 0,
                reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            )
            if usage
            else Stats()
        )

        candidates = getattr(response, "candidates", [])
        finish_reason = None
        if candidates and len(candidates) > 0:
            finish_reason = getattr(candidates[0], "finish_reason", None)

        return ChatResponse(
            text=response.text or "",
            stats=stats,
            stop_reason=self._map_stop_reason(finish_reason),
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
        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=self.messages_to_custom_messages(system_prompt, messages),
            config=self._prepare_config(
                system_prompt, reasoning, temperature
            ),
        )

        usage = response.usage_metadata
        stats = (
            Stats(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage, "total_token_count", 0) or 0,
                cached_input_tokens=getattr(usage, "cached_content_token_count", 0)
                or 0,
                reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
            )
            if usage
            else Stats()
        )

        candidates = getattr(response, "candidates", [])
        finish_reason = None
        if candidates and len(candidates) > 0:
            finish_reason = getattr(candidates[0], "finish_reason", None)

        return ChatResponse(
            text=response.text or "",
            stats=stats,
            stop_reason=self._map_stop_reason(finish_reason),
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
        response = self.client.models.generate_content_stream(
            model=model_name,
            contents=self.messages_to_custom_messages(system_prompt, messages),
            config=self._prepare_config(
                system_prompt, reasoning, temperature
            ),
        )

        final_stats = None
        finish_reason = None
        for chunk in response:
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
                final_stats = Stats(
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(usage, "total_token_count", 0) or 0,
                    cached_input_tokens=getattr(usage, "cached_content_token_count", 0)
                    or 0,
                    reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                )

            candidates = getattr(chunk, "candidates", [])
            if candidates and len(candidates) > 0:
                chunk_reason = getattr(candidates[0], "finish_reason", None)
                if (
                    chunk_reason is not None
                    and str(chunk_reason) != "FINISH_REASON_UNSPECIFIED"
                ):
                    finish_reason = chunk_reason

            if chunk.text:
                yield ChatResponse(text=chunk.text)

        if final_stats or finish_reason is not None:
            yield ChatResponse(
                text="",
                stats=final_stats,
                stop_reason=self._map_stop_reason(finish_reason),
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
        response = await self.client.aio.models.generate_content_stream(
            model=model_name,
            contents=self.messages_to_custom_messages(system_prompt, messages),
            config=self._prepare_config(
                system_prompt, reasoning, temperature
            ),
        )

        final_stats = None
        finish_reason = None
        async for chunk in response:
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
                final_stats = Stats(
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(usage, "total_token_count", 0) or 0,
                    cached_input_tokens=getattr(usage, "cached_content_token_count", 0)
                    or 0,
                    reasoning_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                )

            candidates = getattr(chunk, "candidates", [])
            if candidates and len(candidates) > 0:
                chunk_reason = getattr(candidates[0], "finish_reason", None)
                if (
                    chunk_reason is not None
                    and str(chunk_reason) != "FINISH_REASON_UNSPECIFIED"
                ):
                    finish_reason = chunk_reason

            if chunk.text:
                yield ChatResponse(text=chunk.text)

        if final_stats or finish_reason is not None:
            yield ChatResponse(
                text="",
                stats=final_stats,
                stop_reason=self._map_stop_reason(finish_reason),
            )
