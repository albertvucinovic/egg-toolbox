from __future__ import annotations

from .types import SemanticEvent, EventKind, Tool
from .formats.base import FormatHandler, FormatParserState


class StreamingParser:
    """Top-level parser that wraps a FormatHandler's parser state.

    Handles:
    1. Generation prompt stripping (the template's generation_prompt
       prefix is not part of the model's output)
    2. Lazy grammar activation (for step backends)
    3. Collecting all events into a final message structure
    """

    def __init__(
        self,
        handler: FormatHandler,
        tools: list[Tool] | None = None,
        generation_prompt_suffix: str = "",
    ):
        self._handler = handler
        self._state: FormatParserState = handler.create_parser_state(tools)
        self._gen_prompt_suffix = generation_prompt_suffix
        self._gen_prompt_consumed = not bool(generation_prompt_suffix)
        self._pending_text = ""

        # Accumulated results (for non-streaming consumers)
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: list[dict] = []  # list of {id, type, function: {name, arguments}}

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        """Feed a single token into the parser. Returns events to emit."""
        # Strip generation prompt prefix from first tokens
        if not self._gen_prompt_consumed:
            self._pending_text += token_text
            if len(self._pending_text) >= len(self._gen_prompt_suffix):
                remainder = self._pending_text[len(self._gen_prompt_suffix):]
                self._gen_prompt_consumed = True
                if remainder:
                    events = self._state.feed_token(token_id, remainder)
                    self._accumulate(events)
                    return events
                return []
            return []

        events = self._state.feed_token(token_id, token_text)
        self._accumulate(events)
        return events

    def feed_text(self, text: str) -> list[SemanticEvent]:
        """Feed a chunk of text (for constraint backends)."""
        events = self._state.feed_text(text)
        self._accumulate(events)
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._state.finish()
        self._accumulate(events)
        return events

    def _accumulate(self, events: list[SemanticEvent]) -> None:
        for ev in events:
            if ev.kind == EventKind.CONTENT_DELTA and ev.text:
                self._content_parts.append(ev.text)
            elif ev.kind == EventKind.REASONING_DELTA and ev.text:
                self._reasoning_parts.append(ev.text)
            elif ev.kind == EventKind.TOOL_CALL_COMMIT:
                self._tool_calls.append({
                    "id": ev.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": ev.tool_name,
                        "arguments": ev.tool_arguments,
                    },
                })

    # -- Accessors for final message construction --

    @property
    def content(self) -> str:
        return "".join(self._content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._tool_calls)

    def trigger_detected(self) -> bool:
        """Has the parser detected a tool-call trigger word?
        Used by step backends to decide whether to activate grammar constraints."""
        return self._state.has_pending_tool_call()
