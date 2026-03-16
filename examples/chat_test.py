import asyncio
from dotenv import load_dotenv
from xecai.chat.chat_interface import ChatInterface
from xecai.chat.implementations.anthropic_chat import AnthropicChat
from xecai.chat.implementations.aws_chat import AWSChat
from xecai.chat.implementations.google_chat import GoogleChat
from xecai.chat.implementations.openai_chat import OpenAIChat
from xecai.models import Message, MessageType, ReasoningOptions


async def tester(chat: ChatInterface, model: str):
    messages = [Message(content="what model are you?", message_type=MessageType.USER)]
    prompt = "you are a helpful bot"
    chat.check_model(model)

    response = await chat.async_invoke(model, prompt, messages)
    print(response.text)
    print("Stats:", response.stats)

    async for response in chat.async_stream(model, prompt, messages):
        if response.text:
            print(response.text, end="", flush=True)

        if response.stats:
            print(f"\nStats: {response.stats}")

        if response.stop_reason:
            print(f"\nStop Reason: {response.stop_reason}")


if __name__ == "__main__":
    load_dotenv()
    provider = input("Test provider: ")
    match provider:
        case "aws":
            chat = AWSChat()
            # amazon.nova-lite-v1:0 does not support reasoning
            model = "us.anthropic.claude-sonnet-4-6"
        case "google":
            chat = GoogleChat()
            model = "gemini-3-flash-preview"
        case "anthropic":
            chat = AnthropicChat()
            model = "claude-sonnet-4-6"
        case _:
            chat = OpenAIChat()
            model = "gpt-4o"
            # model = "o3-mini"
    asyncio.run(tester(chat, model))
