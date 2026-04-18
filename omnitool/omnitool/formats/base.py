from __future__ import annotations

import abc
from typing import Any

from ..types import FormatAnalysis, SemanticEvent, Tool


class FormatHandler(abc.ABC):
    """Abstract base for tool-call format handlers.

    A FormatHandler is constructed once per model load from the
    FormatAnalysis produced by the detector. It is then used for
    every request to that model.
    """

    def __init__(self, analysis: FormatAnalysis):
        self.analysis = analysis

    @abc.abstractmethod
    def stop_strings(self) -> tuple[str, ...]:
        """Additional stop strings the backend should watch for."""
        ...

    @abc.abstractmethod
    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        """Additional stop token IDs (resolved from strings via tokenizer)."""
        ...

    @abc.abstractmethod
    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        """Create a fresh, per-request parser state machine."""
        ...

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        """Generate a GBNF grammar constraining output to valid tool calls.

        Returns None if this format does not support grammar constraints.
        Default implementation returns None.
        """
        return None

    def generate_json_schema(self, tools: list[Tool]) -> dict[str, Any] | None:
        """Generate a JSON schema for structured_outputs backends.

        Returns None if this format does not support structured outputs.
        Default implementation returns None.
        """
        return None


class FormatParserState(abc.ABC):
    """Per-request mutable state for streaming tool call extraction.

    The parser state machine processes tokens one at a time (for step
    backends) or chunks of text (for constraint backends). It emits
    SemanticEvents describing what it found.

    The state machine lifecycle:
    1. Created fresh per request via FormatHandler.create_parser_state()
    2. fed tokens/text via feed_token() or feed_text()
    3. finalized via finish()
    4. Discarded after the request completes
    """

    @abc.abstractmethod
    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        """Process a single decoded token. Returns zero or more events."""
        ...

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        """Process a chunk of text (for constraint backends that deliver
        text in larger chunks). Default implementation calls feed_token
        with the full text and token_id=-1.

        Subclasses SHOULD override for more efficient parsing when
        processing large text chunks from constraint backends.
        """
        return self.feed_token(-1, new_text)

    @abc.abstractmethod
    def finish(self) -> list[SemanticEvent]:
        """Signal end of generation. Flush any pending tool calls
        and emit TOOL_CALL_COMMIT / DONE events."""
        ...

    @abc.abstractmethod
    def has_pending_tool_call(self) -> bool:
        """Is a tool call currently being assembled?"""
        ...
