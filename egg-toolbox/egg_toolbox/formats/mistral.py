"""Mistral tool-call format handler.

Mistral Instruct v0.3+ emits tool calls as::

    [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "Paris"}}]

- ``[TOOL_CALLS]`` is usually a single control token in the tokenizer.
- The body is a **JSON array** of tool-call objects (0..N entries); each
  has ``name`` and ``arguments`` fields.
- The array closes with the matching ``]``; natural EOS follows.

Unlike Hermes/Llama3 (one ``<tool_call>`` tag per call), Mistral batches
parallel calls in a single array.  We walk the body with a bracket-aware
scanner so we can detect the array close reliably even if it's split
across token boundaries and even if nested objects contain ``]`` inside
string values.
"""
from __future__ import annotations

import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


class MistralHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        # Mistral closes a tool-call turn with natural EOS (</s>).
        # We don't return stop strings here because the closing "]"
        # inside the array shouldn't halt generation -- the parser
        # detects array end via bracket counting and transitions
        # itself back to CONTENT.
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return MistralParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _MistralState(enum.Enum):
    CONTENT   = "content"
    REASONING = "reasoning"
    IN_ARRAY  = "in_array"     # seen [TOOL_CALLS], accumulating array body
    AFTER     = "after"        # array closed; back to content


class MistralParserState(FormatParserState):
    _TRIGGER = "[TOOL_CALLS]"

    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _MistralState.CONTENT
        self._buffer = ""
        self._body_buffer = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end

        # Bracket scanner state (only consulted in IN_ARRAY).
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._saw_open = False   # have we seen the opening "[" of the array yet?

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            if self._state == _MistralState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _MistralState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _MistralState.IN_ARRAY:
                events.extend(self._process_in_array())
            elif self._state == _MistralState.AFTER:
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
            self._state = _MistralState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        if self._TRIGGER in self._buffer:
            idx = self._buffer.index(self._TRIGGER)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._TRIGGER):]
            self._body_buffer = ""
            self._depth = 0
            self._in_string = False
            self._escape = False
            self._saw_open = False
            self._state = _MistralState.IN_ARRAY
            return events

        for i in range(1, len(self._TRIGGER)):
            if self._buffer.endswith(self._TRIGGER[:i]):
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
            self._state = _MistralState.CONTENT
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

    def _process_in_array(self) -> list[SemanticEvent]:
        """Consume chars one at a time, tracking JSON bracket depth.

        Emits nothing until the outer array ``]`` is found (bracketed
        bodies mix every tool's name and arguments together, so no
        sensible partial projection exists).  When the array closes,
        parse the full body and emit NAME/ARGS_DELTA/COMMIT for each
        element.
        """
        events: list[SemanticEvent] = []
        i = 0
        end_idx = -1
        buf = self._buffer
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
                if ch == "[" and not self._saw_open and self._depth == 0:
                    self._saw_open = True
                self._depth += 1
            elif ch in "]}":
                self._depth -= 1
                if self._saw_open and self._depth == 0:
                    end_idx = i
                    break

        if end_idx >= 0:
            self._body_buffer += buf[: end_idx + 1]
            self._buffer = buf[end_idx + 1 :]
            events.extend(self._commit_array())
            self._state = _MistralState.AFTER
            return events

        # Array isn't closed yet -- absorb what we have and wait.
        self._body_buffer += buf
        self._buffer = ""
        return events

    def _process_after(self) -> list[SemanticEvent]:
        """After the array closes, trailing text (rare) goes to content."""
        events: list[SemanticEvent] = []
        stripped = self._buffer.lstrip()
        if not stripped:
            self._buffer = ""
            return events
        self._buffer = stripped
        self._state = _MistralState.CONTENT
        return events

    def _commit_array(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        calls = _parse_tool_calls_array(self._body_buffer, self._analysis.args_field or "arguments")
        self._body_buffer = ""
        if not calls:
            # Model clearly tried to emit tool calls (opened [TOOL_CALLS])
            # but we couldn't recover any.  Synthesize a single empty
            # tool call so finish_reason=tool_calls still fires.
            calls = [("", "")]
        for name, args in calls:
            tool_call_id = f"call_{self._tool_index}"
            events.extend([
                SemanticEvent(
                    kind=EventKind.TOOL_CALL_START,
                    tool_index=self._tool_index,
                    tool_call_id=tool_call_id,
                ),
                SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name),
                SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=args),
                SemanticEvent(
                    kind=EventKind.TOOL_CALL_COMMIT,
                    tool_index=self._tool_index,
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    tool_arguments=args,
                ),
            ])
            self._tool_index += 1
        return events

    def finish(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._state == _MistralState.IN_ARRAY:
            # Truncated body (EOS before ]).  Commit whatever we can.
            self._body_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_array())
        elif self._buffer:
            if self._state in (_MistralState.CONTENT, _MistralState.AFTER):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _MistralState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _MistralState.IN_ARRAY


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _parse_tool_calls_array(body: str, args_field: str) -> list[tuple[str, str]]:
    """Return a list of (name, args_json_string) from a Mistral body.

    Accepts:
    - a well-formed JSON array of objects
    - a single JSON object (some Mistral fine-tunes emit this)
    - malformed text (regex fallback)
    """
    stripped = body.strip()
    candidates = [stripped]
    candidates.append(re.sub(r",(\s*[}\]])", r"\1", stripped))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue
        out: list[tuple[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "") or ""
            raw_args = item.get(args_field, item.get("arguments", {}))
            args = json.dumps(raw_args) if not isinstance(raw_args, str) else raw_args
            out.append((str(name), args))
        if out:
            return out

    # Regex fallback: harvest names only, keep body as args blob.
    names = _NAME_RE.findall(body)
    if names:
        return [(n, body.strip()) for n in names]
    return []
