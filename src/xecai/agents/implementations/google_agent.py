import asyncio
from typing import Any

from xecai.agents.agent_interface import AgentInterface, Tool, WebSearchTool
from xecai.models import (
    ChatResponse,
    Message,
    MessageType,
    ReasoningOptions,
    StopReason,
)

try:
    from google import genai
    from google.genai.types import FunctionDeclaration
    from google.genai.types import Tool as GenAITool
except ImportError as e:
    raise RuntimeError(
        "Google GenAI SDK required. Install with: uv pip install google-genai"
    ) from e

from xecai.error_handlers.google_error_handler import (
    async_error_decorator,
    sync_error_decorator,
)


def _convert_tools(tools: list[Tool] | None) -> list[Any] | None:
    if not tools:
        return None

    converted = []
    for t in tools:
        if t is WebSearchTool or t.name == "web_search":
            converted.append({"google_search": {}})
            continue

        converted.append(
            GenAITool(
                function_declarations=[
                    FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters={
                            "type": "object",
                            "properties": {},
                        },
                    )
                ]
            )
        )
    return converted


def _prepare_messages(messages: list[Message]) -> list[dict[str, Any]]:
    role_map = {
        MessageType.USER: "user",
        MessageType.BOT: "model",
        MessageType.DEVELOPER: "user",
    }

    return [
        {
            "role": role_map.get(m.message_type, "user"),
            "parts": [{"text": m.content}],
        }
        for m in messages
    ]


class GoogleAgent(AgentInterface):
    def __init__(self) -> None:
        self.client = genai.Client()
        self._tool_cache: dict[tuple, list[GenAITool] | None] = {}

    def parse_tool_call(self, response):
        return None

    def format_tool_result(self, result):
        return Message(content=str(result.output), message_type=MessageType.DEVELOPER)

    def get_tools_prompt(self, tools):
        return ""

    def _get_tools(self, tools: list[Tool] | None) -> list[GenAITool] | None:
        key = tuple((t.name, id(t.func)) for t in (tools or []))

        if key not in self._tool_cache:
            self._tool_cache[key] = _convert_tools(tools)

        return self._tool_cache[key]

    def _build_tool_map(self, tools: list[Tool] | None) -> dict[str, Tool]:
        return {t.name: t for t in (tools or [])}

    def _extract_text(self, response: Any) -> str:
        return getattr(response, "text", "") or str(response)

    def _extract_tool_calls(self, response: Any):
        try:
            parts = response.candidates[0].content.parts
        except Exception:
            return []

        return [p.function_call for p in parts if getattr(p, "function_call", None)]

    def _append_tool_result(self, contents, name: str, result: Any):
        contents.append(
            {
                "role": "tool",
                "parts": [
                    {
                        "function_response": {
                            "name": name,
                            "response": {"result": str(result)},
                        }
                    }
                ],
            }
        )

    async def _loop(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[Tool] | None,
        temperature: float | None,
        max_steps: int,
        run_tool,
    ) -> ChatResponse:

        contents = _prepare_messages(messages)
        genai_tools = self._get_tools(tools)
        tool_map = self._build_tool_map(tools)

        for _ in range(max_steps):
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    "system_instruction": system_prompt,
                    "temperature": temperature,
                    "tools": genai_tools,
                },
            )

            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                return ChatResponse(
                    text=self._extract_text(response),
                    stop_reason=StopReason.END,
                )

            for call in tool_calls:
                tool = tool_map.get(call.name)
                if not tool:
                    continue

                try:
                    result = await run_tool(tool, call.args or {})
                except Exception as e:
                    result = f"Tool error: {e}"

                self._append_tool_result(contents, call.name, result)

        return ChatResponse(
            text="Max steps reached without resolution",
            stop_reason=StopReason.MAX_TOKENS,
        )

    @sync_error_decorator
    def run(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        max_steps: int = 10,
        retries: int = 3,
    ) -> ChatResponse:
        async def run_tool(tool, args):
            return tool.run(args)

        return asyncio.run(
            self._loop(
                model_name=model_name,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_steps=max_steps,
                run_tool=run_tool,
            )
        )

    @async_error_decorator
    async def async_run(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        reasoning: ReasoningOptions | None = None,
        temperature: float | None = None,
        max_steps: int = 10,
        retries: int = 3,
    ) -> ChatResponse:
        async def run_tool(tool, args):
            return await tool.arun(args)

        return await self._loop(
            model_name=model_name,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_steps=max_steps,
            run_tool=run_tool,
        )
