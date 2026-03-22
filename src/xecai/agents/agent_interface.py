import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from xecai.models import ChatResponse, Message, ReasoningOptions, StopReason


class Tool(BaseModel):
    name: str
    description: str
    func: Callable[[dict[str, Any]], Any]
    async_func: Callable[[dict[str, Any]], Awaitable[Any]] | None = None

    def run(self, args: dict[str, Any]) -> Any:
        return self.func(args)

    async def arun(self, args: dict[str, Any]) -> Any:
        return await self.async_func(args) if self.async_func else self.func(args)


def tool(func: Callable) -> Tool:
    """Decorator to turn a function into a Tool."""
    return Tool(
        name=func.__name__,
        description=inspect.getdoc(func) or "",
        func=(lambda _: None) if inspect.iscoroutinefunction(func) else func,
        async_func=func if inspect.iscoroutinefunction(func) else None,
    )


WebSearchTool = Tool(
    name="web_search",
    description="Perform a web search using the provider's built-in capabilities. Provide a 'query' string.",
    func=lambda args: (
        "WebSearchTool is a stub and must be replaced by the specific agent implementation."
    ),
)


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    output: Any


class AgentInterface(ABC):
    @abstractmethod
    def parse_tool_call(self, response: ChatResponse) -> ToolCall | None: ...

    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Message: ...

    @abstractmethod
    def get_tools_prompt(self, tools: list[Tool] | None) -> str: ...

    async def _loop(
        self,
        invoke,
        tool_runner,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[Tool] | None,
        reasoning: ReasoningOptions | None,
        temperature: float | None,
        max_steps: int,
        retries: int,
    ) -> ChatResponse:

        tool_prompt = self.get_tools_prompt(tools)
        system_prompt = (
            f"{system_prompt}\n\n{tool_prompt}" if tool_prompt else system_prompt
        )

        msgs = list(messages)

        for _ in range(max_steps):
            response = await invoke(
                model_name=model_name,
                system_prompt=system_prompt,
                messages=msgs,
                reasoning=reasoning,
                temperature=temperature,
                retries=retries,
            )

            if response.stop_reason != StopReason.TOOL_USE:
                return response

            call = self.parse_tool_call(response)
            if not call:
                return ChatResponse(text=response.text, stop_reason=StopReason.OTHER)

            tool_map = {t.name: t for t in tools or []}
            tool = tool_map.get(call.name)
            if not tool:
                return ChatResponse(
                    text=f"Tool '{call.name}' not found",
                    stop_reason=StopReason.OTHER,
                )

            try:
                output = await tool_runner(tool, call.arguments)
            except Exception as e:
                output = f"Tool execution error: {str(e)}"

            msgs.append(
                self.format_tool_result(ToolResult(call_id=call.call_id, output=output))
            )

        return ChatResponse(text="Max steps reached", stop_reason=StopReason.MAX_TOKENS)

    def run(self, **kwargs) -> ChatResponse:
        return self._run_sync(**kwargs)

    def _run_sync(self, **kwargs) -> ChatResponse:
        async def invoke(**k):
            return self.chat.invoke(**k)

        async def tool_runner(tool, args):
            return tool.run(args)

        return self._run_blocking(invoke, tool_runner, **kwargs)

    async def async_run(self, **kwargs) -> ChatResponse:
        async def invoke(**k):
            return await self.chat.async_invoke(**k)

        async def tool_runner(tool, args):
            return await tool.arun(args)

        return await self._loop(invoke, tool_runner, **kwargs)

    def _run_blocking(self, invoke, tool_runner, **kwargs):
        return asyncio.run(self._loop(invoke, tool_runner, **kwargs))
