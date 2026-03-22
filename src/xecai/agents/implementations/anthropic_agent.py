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
    import claude_agent_sdk
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
except ImportError as e:
    raise RuntimeError(
        "Anthropic Agent SDK requires 'claude-agent-sdk'. "
        "Install with: uv pip install claude-agent-sdk"
    ) from e

from xecai.error_handlers.anthropic_error_handler import async_error_decorator


def _wrap_tool(tool: Tool):
    @claude_agent_sdk.tool(
        name=tool.name,
        description=tool.description,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    )
    async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await tool.arun(args)
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "is_error": True,
            }

    return wrapped


class AnthropicAgent(AgentInterface):
    def parse_tool_call(self, response):
        return None

    def format_tool_result(self, result):
        return Message(content="", message_type=MessageType.DEVELOPER)

    def get_tools_prompt(self, tools):
        return ""

    def _build_input(self, messages: list[Message]) -> str:
        return "\n".join(m.to_prompt_text() for m in messages)

    def run(self, **kwargs) -> ChatResponse:
        raise NotImplementedError(
            "Anthropic's claude-agent-sdk does not implement synchronous methods"
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
        sdk_tools = []
        regular_tools = []

        if tools:
            for t in tools:
                if t is WebSearchTool or t.name == "web_search":
                    sdk_tools.append("WebSearch")
                else:
                    regular_tools.append(t)

        mcp_servers = {}
        if regular_tools:
            wrapped = [_wrap_tool(t) for t in regular_tools]
            server = create_sdk_mcp_server("tools", tools=wrapped)
            mcp_servers = {"tools": server}

        try:
            async with ClaudeSDKClient(
                options=ClaudeAgentOptions(
                    model=model_name,
                    system_prompt=system_prompt,
                    mcp_servers=mcp_servers,
                    tools=sdk_tools if sdk_tools else None,
                    permission_mode="bypassPermissions",
                    include_partial_messages=False,
                )
            ) as client:
                input_text = self._build_input(messages)
                await client.query(input_text)

                async for msg in client.receive_response():
                    if isinstance(msg, claude_agent_sdk.ResultMessage):
                        return ChatResponse(
                            text=str(msg.result or ""),
                            stop_reason=StopReason.END,
                        )

        except Exception as e:
            return ChatResponse(
                text=f"Agent error: {e}",
                stop_reason=StopReason.OTHER,
            )

        return ChatResponse(text="", stop_reason=StopReason.END)
