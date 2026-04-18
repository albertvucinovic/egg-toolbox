"""Functionary tool-call format handler (v3 and v3.1 variants).

Functionary models emit tool calls in one of two shapes depending
on the version of the fine-tune:

**v3** -- HTML-ish inline tag::

    <function=get_weather>{"city": "Paris"}</function>

**v3.1** -- ``>>>`` routing before the function name::

    >>>get_weather
    {"city": "Paris"}

Multiple calls may appear back-to-back in either format.  The
handler auto-detects which variant is in play based on whichever
marker appears first in the stream; a single model will use only
one variant so this is safe.
"""
from __future__ import annotations

import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


_V3_OPEN_RE = re.compile(r"<function=([A-Za-z_][\w\-.]*)>")
_V3_CLOSE = "</function>"
_V3_1_ROUTE = ">>>"


class FunctionaryHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return FunctionaryParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _FState(enum.Enum):
    CONTENT    = "content"
    REASONING  = "reasoning"
    IN_V3_BODY = "in_v3_body"
    IN_V31_NAME = "in_v31_name"   # after >>>, reading the function name until \n
    IN_V31_BODY = "in_v31_body"   # after name\n, reading JSON body


class FunctionaryParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _FState.CONTENT
        self._buffer = ""
        self._body_buffer = ""
        self._current_name = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._saw_open = False

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            if self._state == _FState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _FState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _FState.IN_V3_BODY:
                events.extend(self._process_in_v3_body())
            elif self._state == _FState.IN_V31_NAME:
                events.extend(self._process_in_v31_name())
            elif self._state == _FState.IN_V31_BODY:
                events.extend(self._process_in_v31_body())
            if self._buffer == prev:
                break
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def _process_content(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []

        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _FState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        # V3: <function=name>  -- requires a closing >.
        v3_match = _V3_OPEN_RE.search(self._buffer)
        # V3.1: >>>
        v31_idx = self._buffer.find(_V3_1_ROUTE)
        v3_idx = v3_match.start() if v3_match else -1

        # Complete v3 opener wins if it appears first.
        if v3_idx >= 0 and (v31_idx < 0 or v3_idx <= v31_idx):
            assert v3_match is not None
            if v3_idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:v3_idx]))
            self._current_name = v3_match.group(1)
            self._buffer = self._buffer[v3_match.end():]
            self._body_buffer = ""
            self._depth = 0
            self._in_string = False
            self._escape = False
            self._saw_open = False
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
            ))
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_NAME,
                tool_index=self._tool_index,
                tool_name=self._current_name,
            ))
            self._state = _FState.IN_V3_BODY
            return events

        if v31_idx >= 0:
            if v31_idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:v31_idx]))
            self._buffer = self._buffer[v31_idx + len(_V3_1_ROUTE):]
            self._current_name = ""
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
            ))
            self._state = _FState.IN_V31_NAME
            return events

        # In-progress v3 opener: buffer contains "<function=" but no
        # closing ">" yet.  Hold everything from "<function=" onward.
        v3_open_idx = self._buffer.find("<function=")
        if v3_open_idx >= 0:
            safe = self._buffer[:v3_open_idx]
            if safe:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
            self._buffer = self._buffer[v3_open_idx:]
            return events

        # Partial opener suffix (buffer ending with a prefix of the
        # marker, but the marker hasn't fully landed yet).
        max_partial = 0
        for marker in ("<function=", _V3_1_ROUTE):
            for i in range(1, len(marker)):
                if self._buffer.endswith(marker[:i]):
                    max_partial = max(max_partial, i)
        if max_partial > 0:
            safe = self._buffer[:-max_partial]
            if safe:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
            self._buffer = self._buffer[-max_partial:]
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
            self._state = _FState.CONTENT
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

    def _process_in_v3_body(self) -> list[SemanticEvent]:
        """Read JSON body until </function>."""
        events: list[SemanticEvent] = []
        if _V3_CLOSE in self._buffer:
            idx = self._buffer.index(_V3_CLOSE)
            self._body_buffer += self._buffer[:idx]
            self._buffer = self._buffer[idx + len(_V3_CLOSE):]
            events.extend(self._commit_call())
            self._state = _FState.CONTENT
            return events
        for i in range(1, len(_V3_CLOSE)):
            if self._buffer.endswith(_V3_CLOSE[:i]):
                self._body_buffer += self._buffer[:-i]
                self._buffer = self._buffer[-i:]
                return events
        self._body_buffer += self._buffer
        self._buffer = ""
        return events

    def _process_in_v31_name(self) -> list[SemanticEvent]:
        """Collect the function name up to the first newline."""
        events: list[SemanticEvent] = []
        if "\n" in self._buffer:
            idx = self._buffer.index("\n")
            self._current_name = (self._current_name + self._buffer[:idx]).strip()
            self._buffer = self._buffer[idx + 1:]
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_NAME,
                tool_index=self._tool_index,
                tool_name=self._current_name,
            ))
            self._body_buffer = ""
            self._depth = 0
            self._in_string = False
            self._escape = False
            self._saw_open = False
            self._state = _FState.IN_V31_BODY
            return events
        # No newline yet -- keep accumulating into name.
        self._current_name += self._buffer
        self._buffer = ""
        return events

    def _process_in_v31_body(self) -> list[SemanticEvent]:
        """Collect the JSON body via bracket counting.

        V3.1 has no explicit close marker; the call ends at the matching
        close bracket of the top-level JSON object (or at another
        ``>>>`` routing marker, or at EOS).
        """
        events: list[SemanticEvent] = []
        buf = self._buffer

        # Check for a new routing marker BEFORE entering bracket scan
        # -- some variants emit >>>name\n{...}>>>next without proper
        # JSON closing on the first call.
        if _V3_1_ROUTE in buf and self._depth == 0 and not self._in_string:
            route_idx = buf.index(_V3_1_ROUTE)
            self._body_buffer += buf[:route_idx]
            self._buffer = buf[route_idx:]
            events.extend(self._commit_call())
            self._state = _FState.CONTENT
            return events

        end_idx = -1
        for i, ch in enumerate(buf):
            if self._escape:
                self._escape = False
                continue
            if self._in_string:
                if ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                continue
            if ch == '"':
                self._in_string = True
                continue
            if ch in "[{":
                if self._depth == 0 and not self._saw_open:
                    self._saw_open = True
                self._depth += 1
            elif ch in "]}":
                self._depth -= 1
                if self._depth == 0 and self._saw_open:
                    end_idx = i
                    break

        if end_idx >= 0:
            self._body_buffer += buf[: end_idx + 1]
            self._buffer = buf[end_idx + 1 :]
            events.extend(self._commit_call())
            self._state = _FState.CONTENT
            return events

        self._body_buffer += buf
        self._buffer = ""
        return events

    def _commit_call(self) -> list[SemanticEvent]:
        args = _normalise_args(self._body_buffer)
        self._body_buffer = ""
        name = self._current_name
        self._current_name = ""
        tool_call_id = f"call_{self._tool_index}"
        events = [
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
        if self._state in (_FState.IN_V3_BODY, _FState.IN_V31_BODY):
            self._body_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_call())
        elif self._state == _FState.IN_V31_NAME:
            # EOS mid-name.  Commit with whatever we have, empty args.
            self._current_name = (self._current_name + self._buffer).strip()
            self._buffer = ""
            events.extend(self._commit_call())
        elif self._buffer:
            if self._state == _FState.CONTENT:
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _FState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state in (_FState.IN_V3_BODY, _FState.IN_V31_NAME, _FState.IN_V31_BODY)


def _normalise_args(body: str) -> str:
    """Re-serialise the body as canonical JSON if parseable, else
    return the stripped original."""
    stripped = body.strip()
    for cand in (stripped, re.sub(r",(\s*[}\]])", r"\1", stripped)):
        try:
            parsed = json.loads(cand)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            continue
    return stripped
