import functools
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Callable, ParamSpec, TypeVar
from xecai.models import BadRequestError, RateLimitError


try:
    import openai
except ImportError as e:
    raise RuntimeError(
        """OpenAI provider requires the 'openai' extra.
        Install with: uv pip install xecai[openai]"""
    ) from e


P = ParamSpec("P")
T = TypeVar("T")


def sync_error_decorator(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except openai.BadRequestError as e:
            raise BadRequestError(str(e)) from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.InternalServerError as e:
            raise RateLimitError(str(e)) from e

    return wrapper


def async_error_decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except openai.BadRequestError as e:
            raise BadRequestError(str(e)) from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.InternalServerError as e:
            raise RateLimitError(str(e)) from e

    return wrapper


def sync_generator_error_decorator(
    func: Callable[P, Iterator[T]],
) -> Callable[P, Iterator[T]]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[T]:
        try:
            yield from func(*args, **kwargs)
        except openai.BadRequestError as e:
            raise BadRequestError(str(e)) from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.InternalServerError as e:
            raise RateLimitError(str(e)) from e

    return wrapper


def async_generator_error_decorator(
    func: Callable[P, AsyncIterator[T]],
) -> Callable[P, AsyncIterator[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[T]:
        try:
            async for item in func(*args, **kwargs):
                yield item
        except openai.BadRequestError as e:
            raise BadRequestError(str(e)) from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.InternalServerError as e:
            raise RateLimitError(str(e)) from e

    return wrapper
