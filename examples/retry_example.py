import asyncio
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from xecai.chat.implementations.openai_chat import OpenAIChat
from xecai.models import Message, MessageType, RateLimitError

load_dotenv()


async def main():
    chat_client = OpenAIChat()

    messages = [
        Message(
            message_type=MessageType.USER,
            content="Hello! Can you give me a brief explanation of how black holes work?",
        )
    ]

    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        ):
            with attempt:
                print(f"Attempt {attempt.retry_state.attempt_number}...")
                response = await chat_client.async_invoke(
                    model_name="gpt-4o",
                    system_prompt="You are a helpful and educational assistant.",
                    messages=messages,
                )
                print(response.text + "\n" + "-" * 50)

    except RateLimitError as e:
        print(f"\nFailed: Rate limit exceeded and max retries reached. Error: {e}")
    except Exception as e:
        print(f"\nFailed: An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
