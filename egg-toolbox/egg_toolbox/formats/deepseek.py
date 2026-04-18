"""DeepSeek V3 / R1 tool-call format handler.

DeepSeek's chat template wraps tool calls in a stack of fullwidth-
vertical-bar markers with SentencePiece-style block separators::

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>search
    ```json
    {"q": "foo"}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Notes:
- ``｜`` is U+FF5C (FULLWIDTH VERTICAL LINE); ``▁`` is U+2581
  (LOWER ONE EIGHTH BLOCK) used by SentencePiece as the word-start
  marker.  Both are single characters in the tokenizer vocabulary.
- The JSON body is fenced inside ``` ```json ... ``` ``` fences; the
  handler strips the fences before parsing.
- Each inner ``<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>`` block is
  ONE tool call.  Parallel calls nest inside the outer
  ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>``.
"""
from __future__ import annotations

import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


_OUTER_BEGIN = "<｜tool▁calls▁begin｜>"
_OUTER_END = "<｜tool▁calls▁end｜>"
_INNER_BEGIN = "<｜tool▁call▁begin｜>"
_INNER_END = "<｜tool▁call▁end｜>"
_SEP = "<｜tool▁sep｜>"

# DeepSeek fences arguments with ```json ... ``` (or just ``` ... ```).
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


class DeepSeekHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        # Let the parser detect the outer end marker via its own state
        # machine; returning it as a stop string would cut generation
        # short when the model still had content to emit after tool
        # calls.
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return DeepSeekParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _DSState(enum.Enum):
    CONTENT    = "content"
    REASONING  = "reasoning"
    IN_OUTER   = "in_outer"    # inside <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
    IN_INNER   = "in_inner"    # inside <｜tool▁call▁begin｜>...<｜tool▁call▁end｜>
    AFTER      = "after"


class DeepSeekParserState(FormatParserState):
    _ALL_MARKERS = (_OUTER_BEGIN, _OUTER_END, _INNER_BEGIN, _INNER_END)

    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _DSState.CONTENT
        self._buffer = ""
        self._inner_buffer = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            if self._state == _DSState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _DSState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _DSState.IN_OUTER:
                events.extend(self._process_in_outer())
            elif self._state == _DSState.IN_INNER:
                events.extend(self._process_in_inner())
            elif self._state == _DSState.AFTER:
                events.extend(self._process_after())
            if self._buffer == prev:
                break
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def _process_content(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []

        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _DSState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        if _OUTER_BEGIN in self._buffer:
            idx = self._buffer.index(_OUTER_BEGIN)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(_OUTER_BEGIN):]
            self._state = _DSState.IN_OUTER
            return events

        for i in range(1, len(_OUTER_BEGIN)):
            if self._buffer.endswith(_OUTER_BEGIN[:i]):
                safe = self._buffer[:-i]
                if safe:
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
                self._buffer = self._buffer[-i:]
                return events

        events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
        self._buffer = ""
        return events

    def _process_reasoning(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._reason_end and self._reason_end in self._buffer:
            idx = self._buffer.index(self._reason_end)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._reason_end):]
            self._state = _DSState.CONTENT
            return events
        if self._reason_end:
            for i in range(1, len(self._reason_end)):
                if self._buffer.endswith(self._reason_end[:i]):
                    safe = self._buffer[:-i]
                    if safe:
                        events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=safe))
                    self._buffer = self._buffer[-i:]
                    return events
        events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
        self._buffer = ""
        return events

    def _process_in_outer(self) -> list[SemanticEvent]:
        """Look for the next <｜tool▁call▁begin｜> or the closing
        <｜tool▁calls▁end｜>."""
        events: list[SemanticEvent] = []
        if _INNER_BEGIN in self._buffer:
            idx = self._buffer.index(_INNER_BEGIN)
            self._buffer = self._buffer[idx + len(_INNER_BEGIN):]
            self._inner_buffer = ""
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
            ))
            self._state = _DSState.IN_INNER
            return events
        if _OUTER_END in self._buffer:
            idx = self._buffer.index(_OUTER_END)
            self._buffer = self._buffer[idx + len(_OUTER_END):]
            self._state = _DSState.AFTER
            return events
        # Hold back any potential partial marker suffix.
        max_partial = 0
        for marker in (_INNER_BEGIN, _OUTER_END):
            for i in range(1, len(marker)):
                if self._buffer.endswith(marker[:i]):
                    max_partial = max(max_partial, i)
        if max_partial > 0:
            self._buffer = self._buffer[-max_partial:]
        else:
            # Whitespace / stray text between calls gets discarded.
            self._buffer = ""
        return events

    def _process_in_inner(self) -> list[SemanticEvent]:
        """Accumulate the single-call body until <｜tool▁call▁end｜>."""
        events: list[SemanticEvent] = []
        if _INNER_END in self._buffer:
            idx = self._buffer.index(_INNER_END)
            self._inner_buffer += self._buffer[:idx]
            self._buffer = self._buffer[idx + len(_INNER_END):]
            events.extend(self._commit_inner())
            self._state = _DSState.IN_OUTER
            return events
        max_partial = 0
        for i in range(1, len(_INNER_END)):
            if self._buffer.endswith(_INNER_END[:i]):
                max_partial = max(max_partial, i)
        if max_partial > 0:
            self._inner_buffer += self._buffer[:-max_partial]
            self._buffer = self._buffer[-max_partial:]
            return events
        self._inner_buffer += self._buffer
        self._buffer = ""
        return events

    def _process_after(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        stripped = self._buffer.lstrip()
        if not stripped:
            self._buffer = ""
            return events
        self._buffer = stripped
        self._state = _DSState.CONTENT
        return events

    def _commit_inner(self) -> list[SemanticEvent]:
        name, args = _parse_deepseek_inner(self._inner_buffer)
        self._inner_buffer = ""
        tool_call_id = f"call_{self._tool_index}"
        events = [
            SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name),
            SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=args),
            SemanticEvent(
                kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
                tool_name=name,
                tool_arguments=args,
            ),
        ]
        self._tool_index += 1
        return events

    def finish(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._state == _DSState.IN_INNER:
            self._inner_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_inner())
        elif self._state == _DSState.IN_OUTER:
            # Opened the outer block but no inner calls were recognised.
            # Synthesize a marker-only tool call so finish_reason fires.
            self._buffer = ""
            self._inner_buffer = ""
            events.extend(self._commit_inner())
        elif self._buffer:
            if self._state in (_DSState.CONTENT, _DSState.AFTER):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _DSState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state in (_DSState.IN_INNER, _DSState.IN_OUTER)


def _parse_deepseek_inner(body: str) -> tuple[str, str]:
    """Extract (name, args_json_string) from a single tool-call body.

    Expected shape::

        function<｜tool▁sep｜>fn_name\n```json\n{...}\n```

    But models sometimes skip the ``function`` keyword or the json
    fence.  We try each in turn.
    """
    # Strip leading "function" keyword if present.
    text = body
    if _SEP in text:
        text = text.split(_SEP, 1)[1]
    text = text.lstrip()

    # Name is the first token up to a newline or a fence.
    name = ""
    if "\n" in text:
        name, rest = text.split("\n", 1)
    elif "```" in text:
        name, rest = text.split("```", 1)
        rest = "```" + rest
    else:
        name, rest = text.strip(), ""
    name = name.strip()

    # Arguments: prefer the first json fence, else the rest verbatim.
    m = _FENCE_RE.search(rest)
    raw_args = m.group(1) if m else rest.strip()

    # Re-serialise if it's valid JSON to normalise whitespace; else
    # keep as-is.
    try:
        parsed = json.loads(raw_args)
        args = json.dumps(parsed)
    except json.JSONDecodeError:
        args = raw_args

    return name, args
