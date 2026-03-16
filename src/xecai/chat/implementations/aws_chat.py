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

from xecai.error_handlers.aws_error_handler import (
    async_error_decorator,
    async_generator_error_decorator,
    sync_error_decorator,
    sync_generator_error_decorator,
)

try:
    import boto3
    from aiobotocore.session import AioSession, get_session
except ImportError as e:
    raise RuntimeError(
        """AWS Bedrock provider requires the 'aws' extra.
        Install with: uv pip install xecai[aws]"""
    ) from e


class AWSChat(ChatInterface):
    def __init__(self):
        self.region_name: Final[str] = os.environ.get("AWS_REGION", "us-east-1")
        self.client: Final[boto3.session.Session.client] = boto3.client(
            "bedrock-runtime", region_name=self.region_name
        )
        self.bedrock_client: Final[boto3.session.Session.client] = boto3.client(
            "bedrock", region_name=self.region_name
        )
        self.aiosession: Final[AioSession] = get_session()

    @sync_error_decorator
    def check_model(self, model_name: str) -> None:
        response = self.bedrock_client.list_foundation_models()
        available_models = response.get("modelSummaries", [])
        if any(m.get("modelId") == model_name for m in available_models):
            return

        try:
            profiles_response = self.bedrock_client.list_inference_profiles()
            available_profiles = profiles_response.get("inferenceProfileSummaries", [])
            if any(p.get("inferenceProfileId") == model_name for p in available_profiles):
                return
        except Exception:
            pass

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

            contents.append({"role": role, "content": [{"text": msg.content}]})
        return contents

    def _prepare_kwargs(
        self,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        reasoning: ReasoningOptions | None,
        temperature: float | None,
        retries: int = 3,
    ) -> dict:

        kwargs = {
            "modelId": model_name,
            "messages": self.messages_to_custom_messages(system_prompt, messages),
            "system": [{"text": system_prompt}],
        }

        if temperature is not None:
            kwargs["inferenceConfig"] = {"temperature": temperature}

        if reasoning and reasoning != ReasoningOptions.NONE:
            budget_map = {
                ReasoningOptions.LOW: 1024,
                ReasoningOptions.MEDIUM: 4096,
                ReasoningOptions.HIGH: 16384,
            }
            budget = budget_map[reasoning]
            kwargs["additionalModelRequestFields"] = {
                "reasoning_config": {"type": "enabled", "budget_tokens": budget}
            }

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
        response = self.client.converse(**kwargs)

        usage = response.get("usage", {})
        stats = Stats(
            input_tokens=usage.get("inputTokens", 0) or 0,
            output_tokens=usage.get("outputTokens", 0) or 0,
            total_tokens=usage.get("totalTokens", 0) or 0,
        )

        text = (
            response.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
        )

        stop_reason = response.get("stopReason")

        return ChatResponse(
            text=text, stats=stats, stop_reason=self._map_stop_reason(stop_reason)
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
        async with self.aiosession.create_client(
            "bedrock-runtime", region_name=self.region_name
        ) as client:
            response = await client.converse(**kwargs)

            usage = response.get("usage", {})
            stats = Stats(
                input_tokens=usage.get("inputTokens", 0) or 0,
                output_tokens=usage.get("outputTokens", 0) or 0,
                total_tokens=usage.get("totalTokens", 0) or 0,
            )

            text = (
                response.get("output", {})
                .get("message", {})
                .get("content", [{}])[0]
                .get("text", "")
            )

            stop_reason = response.get("stopReason")

            return ChatResponse(
                text=text, stats=stats, stop_reason=self._map_stop_reason(stop_reason)
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
        )
        response = self.client.converse_stream(**kwargs)
        for chunk in response.get("stream", []):
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield ChatResponse(text=delta["text"])
            elif "messageStop" in chunk:
                stop_reason = chunk["messageStop"].get("stopReason")
                yield ChatResponse(
                    text="", stop_reason=self._map_stop_reason(stop_reason)
                )
            elif "metadata" in chunk:
                usage = chunk["metadata"].get("usage", {})
                if usage:
                    stats = Stats(
                        input_tokens=usage.get("inputTokens", 0) or 0,
                        output_tokens=usage.get("outputTokens", 0) or 0,
                        total_tokens=usage.get("totalTokens", 0) or 0,
                    )
                    yield ChatResponse(text="", stats=stats)

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
        )
        async with self.aiosession.create_client(
            "bedrock-runtime", region_name=self.region_name
        ) as client:
            response = await client.converse_stream(**kwargs)
            stream = response.get("stream")
            if stream:
                async for chunk in stream:
                    if "contentBlockDelta" in chunk:
                        delta = chunk["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            yield ChatResponse(text=delta["text"])
                    elif "messageStop" in chunk:
                        stop_reason = chunk["messageStop"].get("stopReason")
                        yield ChatResponse(
                            text="", stop_reason=self._map_stop_reason(stop_reason)
                        )
                    elif "metadata" in chunk:
                        usage = chunk["metadata"].get("usage", {})
                        if usage:
                            stats = Stats(
                                input_tokens=usage.get("inputTokens", 0) or 0,
                                output_tokens=usage.get("outputTokens", 0) or 0,
                                total_tokens=usage.get("totalTokens", 0) or 0,
                            )
                            yield ChatResponse(text="", stats=stats)
