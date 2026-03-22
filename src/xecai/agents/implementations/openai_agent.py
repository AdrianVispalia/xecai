import json
from typing import Any

from xecai.agents.agent_interface import AgentInterface, Tool, WebSearchTool
from xecai.models import (
    ChatResponse,
    Message,
    ReasoningOptions,
    StopReason,
)

try:
    from agents import Agent, FunctionTool, Runner
    from agents import WebSearchTool as OpenAIWebSearchTool
except ImportError as e:
    raise RuntimeError(
        """OpenAI Agents provider requires the packages 'openai-agents' and 'openai'.
        Install with: uv add openai-agents openai"""
    ) from e

from xecai.error_handlers.openai_error_handler import (
    async_error_decorator,
    sync_error_decorator,
)


def _make_tool_invoker(tool: Tool):
    """Creates an async invoker compatible with OpenAI Agents SDK."""

    async def on_invoke(ctx: Any, args_str: str) -> Any:
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}

        return await tool.arun(args)

    return on_invoke


class OpenAIAgent(AgentInterface):
    def parse_tool_call(self, response):
        return None

    def format_tool_result(self, result):
        return Message(content="")  # never used

    def get_tools_prompt(self, tools):
        return ""

    def _build_agent(
        self,
        model_name: str,
        system_prompt: str,
        tools: list[Tool] | None,
    ) -> Agent:
        sdk_tools: list[Any] = []

        if tools:
            for t in tools:
                if t is WebSearchTool or t.name == "web_search":
                    sdk_tools.append(OpenAIWebSearchTool())
                    continue

                sdk_tools.append(
                    FunctionTool(
                        name=t.name,
                        description=t.description,
                        params_json_schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        },
                        on_invoke_tool=_make_tool_invoker(t),
                        strict_json_schema=False,
                    )
                )

        return Agent(
            name="OpenAIAgent",
            instructions=system_prompt,
            model=model_name,
            tools=sdk_tools,
        )

    def _build_input(self, messages: list[Message]) -> str:
        return "\n".join(m.to_prompt_text() for m in messages)

    def _format_output(self, result: Any) -> ChatResponse:
        final_output = (
            result.final_output if hasattr(result, "final_output") else result
        )

        return ChatResponse(
            text=str(final_output) if final_output is not None else "",
            stop_reason=StopReason.END,
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
        agent = self._build_agent(model_name, system_prompt, tools)
        input_text = self._build_input(messages)

        try:
            result = Runner.run_sync(agent, input_text, max_turns=max_steps)
            return self._format_output(result)
        except Exception as e:
            return ChatResponse(
                text=f"Agent error: {e}",
                stop_reason=StopReason.OTHER,
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
        agent = self._build_agent(model_name, system_prompt, tools)
        input_text = self._build_input(messages)

        try:
            result = await Runner.run(agent, input_text, max_turns=max_steps)
            return self._format_output(result)
        except Exception as e:
            return ChatResponse(
                text=f"Agent error: {e}",
                stop_reason=StopReason.OTHER,
            )
