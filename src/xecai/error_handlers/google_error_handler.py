import functools
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Callable, NoReturn, ParamSpec, TypeVar
from xecai.models import BadRequestError, RateLimitError


try:
    from google.genai.errors import APIError
except ImportError as e:
    raise RuntimeError(
        """Google provider requires the 'google' extra.
        Install with: uv pip install xecai[google]"""
    ) from e

P = ParamSpec("P")
T = TypeVar("T")


def handle_api_error(e: APIError) -> NoReturn:
    code = None
    if isinstance(e.code, int):
        code = e.code
    elif getattr(e, "details", None) and isinstance(e.details, dict):
        error_details = e.details.get("error", {})
        if isinstance(error_details, dict):
            code = error_details.get("code")

    message = getattr(e, "message", str(e))
    if code == 400:
        raise BadRequestError(message) from e
    elif code == 429 or code == 500 or code == 503:
        raise RateLimitError(message) from e

    raise e


def sync_error_decorator(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except APIError as e:
            handle_api_error(e)

    return wrapper


def async_error_decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except APIError as e:
            handle_api_error(e)

    return wrapper


def sync_generator_error_decorator(
    func: Callable[P, Iterator[T]],
) -> Callable[P, Iterator[T]]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[T]:
        try:
            yield from func(*args, **kwargs)
        except APIError as e:
            handle_api_error(e)

    return wrapper


def async_generator_error_decorator(
    func: Callable[P, AsyncIterator[T]],
) -> Callable[P, AsyncIterator[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[T]:
        try:
            async for item in func(*args, **kwargs):
                yield item
        except APIError as e:
            handle_api_error(e)

    return wrapper
