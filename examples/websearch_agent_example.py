import asyncio

from dotenv import load_dotenv

from xecai.agents.agent_interface import AgentInterface, WebSearchTool
from xecai.agents.implementations.anthropic_agent import AnthropicAgent
from xecai.agents.implementations.google_agent import GoogleAgent
from xecai.agents.implementations.openai_agent import OpenAIAgent
from xecai.models import Message, MessageType


async def run_agent(agent: AgentInterface, model: str):
    tools = [WebSearchTool]
    messages: list[Message] = []
    system_prompt = (
        "You are a helpful search assistant. "
        "Always use the tools provided to get the latest and accurate information, "
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
            agent, model = GoogleAgent(), "gemini-2.5-flash"
        case _:
            agent, model = OpenAIAgent(), "gpt-4o"

    asyncio.run(run_agent(agent, model))
