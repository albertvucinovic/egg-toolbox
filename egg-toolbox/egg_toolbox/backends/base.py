from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Iterator

from ..types import CompiledRequest


class Tokenizer(abc.ABC):
    """Minimal tokenizer interface that backends must expose."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str: ...

    @abc.abstractmethod
    def decode_single(self, token_id: int) -> str: ...

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> int: ...

    @property
    @abc.abstractmethod
    def bos_token_id(self) -> int | None: ...

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int: ...


class StepBackend(abc.ABC):
    """Backend where we control the generation loop token-by-token.

    Suitable for: tinygrad, llama-cpp-python.
    """

    @abc.abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer: ...

    @abc.abstractmethod
    def chat_template_source(self) -> str:
        """Return the raw Jinja2 chat template string from model metadata."""
        ...

    @abc.abstractmethod
    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        """Yield token IDs one at a time.

        The caller (orchestrator) is responsible for:
        - Decoding each token
        - Feeding it to the parser
        - Checking stop conditions
        - Breaking the loop when done

        The backend handles:
        - KV cache management
        - Sampling with the given parameters
        """
        ...

    @abc.abstractmethod
    def model_name(self) -> str: ...


class ConstraintBackend(abc.ABC):
    """Backend that owns the generation loop but accepts constraints.

    Suitable for: vLLM, SGLang.
    """

    @abc.abstractmethod
    async def load_model(self, model_path: str, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer: ...

    @abc.abstractmethod
    def chat_template_source(self) -> str: ...

    @abc.abstractmethod
    async def generate_stream(self, request: CompiledRequest) -> AsyncIterator[str]:
        """Yield text chunks (not individual tokens).

        The backend applies constraints (grammar, structured_outputs, stop
        strings) internally. The caller receives decoded text chunks and
        feeds them to the parser.
        """
        ...
        yield  # pragma: no cover

    @abc.abstractmethod
    def model_name(self) -> str: ...


# Union type for any backend
Backend = StepBackend | ConstraintBackend
