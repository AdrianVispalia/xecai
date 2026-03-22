import asyncio
from typing import Any

from dotenv import load_dotenv

from xecai.agents.agent_interface import AgentInterface, tool
from xecai.agents.implementations.anthropic_agent import AnthropicAgent
from xecai.agents.implementations.google_agent import GoogleAgent
from xecai.agents.implementations.openai_agent import OpenAIAgent
from xecai.models import Message, MessageType


@tool
def multiply(args: dict[str, Any]) -> Any:
    """Multiply two numbers. Requires 'a' and 'b'."""
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    result = a * b
    print(f"\n  [Tool used: multiply(a={a}, b={b}) -> {result}]")
    return result


@tool
def divide(args: dict[str, Any]) -> Any:
    """Divide number 'a' by number 'b'. Requires 'a' and 'b'."""
    a = float(args.get("a", 0))
    b = float(args.get("b", 1))
    if b == 0:
        result = "Error: Division by zero"
    else:
        result = a / b
    print(f"\n  [Tool used: divide(a={a}, b={b}) -> {result}]")
    return result


async def run_agent(agent: AgentInterface, model: str):
    tools = [multiply, divide]
    messages: list[Message] = []
    system_prompt = (
        "You are a helpful mathematical assistant. "
        "Always use the tools provided to do the mathematical operations, "
        "otherwise say you can't do it. Don't say approximately, use the exact tool result value."
    )

    print(f"Agent is ready. Using model: {model}")
    print(f"Available tools: {', '.join([tool.name for tool in tools])}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        messages.append(Message(content=user_input, message_type=MessageType.USER))
        response = await agent.async_run(
            model_name=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )
        print(" " * 20, end="\r")
        print(f"Agent: {response.text}\n")
        messages.append(Message(content=response.text, message_type=MessageType.BOT))


if __name__ == "__main__":
    load_dotenv()
    provider = input("Test provider (anthropic/openai/google): ").strip().lower()

    match provider:
        case "anthropic":
            agent, model = AnthropicAgent(), "claude-sonnet-4-6"
        case "google":
            agent, model = GoogleAgent(), "gemini-3-flash-preview"
        case _:
            agent, model = OpenAIAgent(), "gpt-4o"

    asyncio.run(run_agent(agent, model))
