import functools
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Callable, NoReturn, ParamSpec, TypeVar
from xecai.models import CredentialsError, BadRequestError, RateLimitError

try:
    import botocore.exceptions
except ImportError as e:
    raise RuntimeError(
        """AWS Bedrock provider requires the 'aws' extra.
        Install with: uv pip install xecai[aws]"""
    ) from e

P = ParamSpec("P")
T = TypeVar("T")


def handle_client_error(e: botocore.exceptions.ClientError) -> NoReturn:
    error_code = e.response.get("Error", {}).get("Code", "")
    message = e.response.get("Error", {}).get("Message", str(e))

    if error_code == "ThrottlingException" or error_code == "InternalServerException":
        raise RateLimitError(message) from e
    elif error_code == "ValidationException":
        raise BadRequestError(message) from e
    elif error_code in ("UnrecognizedClientException", "AccessDeniedException"):
        raise CredentialsError(message) from e

    raise e


def handle_botocore_error(e: botocore.exceptions.BotoCoreError) -> NoReturn:
    if isinstance(e, botocore.exceptions.NoCredentialsError):
        raise CredentialsError(str(e)) from e

    raise e


def sync_error_decorator(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except botocore.exceptions.ClientError as e:
            handle_client_error(e)
        except botocore.exceptions.BotoCoreError as e:
            handle_botocore_error(e)

    return wrapper


def async_error_decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except botocore.exceptions.ClientError as e:
            handle_client_error(e)
        except botocore.exceptions.BotoCoreError as e:
            handle_botocore_error(e)

    return wrapper


def sync_generator_error_decorator(
    func: Callable[P, Iterator[T]],
) -> Callable[P, Iterator[T]]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[T]:
        try:
            yield from func(*args, **kwargs)
        except botocore.exceptions.ClientError as e:
            handle_client_error(e)
        except botocore.exceptions.BotoCoreError as e:
            handle_botocore_error(e)

    return wrapper


def async_generator_error_decorator(
    func: Callable[P, AsyncIterator[T]],
) -> Callable[P, AsyncIterator[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[T]:
        try:
            async for item in func(*args, **kwargs):
                yield item
        except botocore.exceptions.ClientError as e:
            handle_client_error(e)
        except botocore.exceptions.BotoCoreError as e:
            handle_botocore_error(e)

    return wrapper
