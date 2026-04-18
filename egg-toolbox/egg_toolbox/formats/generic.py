"""Generic fallback tool-call format handler.

Used when a model has no recognisable structural marker but its
chat template claims tool support.  Heuristic:

- If the generation (after any reasoning block) starts with ``{`` or
  ``[`` at the first non-whitespace position, attempt to parse it as
  a tool call (object) or tool-call array.
- Otherwise treat everything as content.

Accepts both shapes::

    {"name": "fn", "arguments": {...}}                      # single call
    [{"name": "a", "arguments": {...}}, {"name": "b", ...}] # parallel calls

This is intentionally permissive -- it is the last-resort fallback.
Content that merely *contains* JSON (e.g. a prose paragraph with a
code block) is NOT interpreted as a tool call; only leading JSON is.
"""
from __future__ import annotations

import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


class GenericHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return GenericParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _GenericState(enum.Enum):
    START     = "start"       # decide if leading token is JSON or text
    REASONING = "reasoning"
    IN_JSON   = "in_json"     # bracket-counting through leading JSON
    CONTENT   = "content"     # everything after is prose content


class GenericParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _GenericState.START
        self._buffer = ""
        self._json_buffer = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end
        self._depth = 0
        self._in_string = False
        self._escape = False

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            if self._state == _GenericState.START:
                events.extend(self._process_start())
            elif self._state == _GenericState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _GenericState.IN_JSON:
                events.extend(self._process_in_json())
            elif self._state == _GenericState.CONTENT:
                events.extend(self._process_content())
            if self._buffer == prev:
                break
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def _process_start(self) -> list[SemanticEvent]:
        """Look at the first non-whitespace character.  If it's { or [,
        enter IN_JSON.  Otherwise, if a reasoning marker is next, enter
        REASONING.  Otherwise fall through to CONTENT."""
        events: list[SemanticEvent] = []

        # Reasoning entry (before any JSON check).
        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _GenericState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        stripped = self._buffer.lstrip()
        # We need at least one non-whitespace char to decide.
        if not stripped:
            # Preserve the whitespace -- it may become prose if JSON
            # doesn't materialise.  For now, hold it until we see
            # the first real char.
            return events

        first = stripped[0]
        if first in "{[":
            # Promote the leading whitespace to discarded (tool call
            # prefix) and begin JSON accumulation.
            self._buffer = stripped
            self._state = _GenericState.IN_JSON
            self._json_buffer = ""
            self._depth = 0
            self._in_string = False
            self._escape = False
            return events

        # Not JSON.  Switch to CONTENT and reprocess.
        self._state = _GenericState.CONTENT
        return events

    def _process_reasoning(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._reason_end and self._reason_end in self._buffer:
            idx = self._buffer.index(self._reason_end)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._reason_end):]
            # After reasoning, re-check for JSON at start of rest.
            self._state = _GenericState.START
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

    def _process_in_json(self) -> list[SemanticEvent]:
        """Scan leading JSON object/array with bracket counting."""
        events: list[SemanticEvent] = []
        buf = self._buffer
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
                self._depth += 1
            elif ch in "]}":
                self._depth -= 1
                if self._depth == 0:
                    end_idx = i
                    break

        if end_idx >= 0:
            self._json_buffer += buf[: end_idx + 1]
            self._buffer = buf[end_idx + 1 :]
            events.extend(self._commit_json())
            self._state = _GenericState.CONTENT
            return events

        self._json_buffer += buf
        self._buffer = ""
        return events

    def _process_content(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        # Once we're in CONTENT, we may still need to emit a reasoning
        # block (unlikely but possible).
        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _GenericState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer.endswith(self._reason_start[:i]):
                    safe = self._buffer[:-i]
                    if safe:
                        events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
                    self._buffer = self._buffer[-i:]
                    return events
        events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
        self._buffer = ""
        return events

    def _commit_json(self) -> list[SemanticEvent]:
        calls = _parse_generic_body(self._json_buffer, self._analysis.args_field or "arguments")
        self._json_buffer = ""
        if not calls:
            return []
        events: list[SemanticEvent] = []
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
        if self._state == _GenericState.IN_JSON:
            self._json_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_json())
        elif self._buffer:
            if self._state in (_GenericState.START, _GenericState.CONTENT):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _GenericState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _GenericState.IN_JSON


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _parse_generic_body(body: str, args_field: str) -> list[tuple[str, str]]:
    """Parse a generic JSON tool-call body (object or array)."""
    stripped = body.strip()
    for cand in (stripped, re.sub(r",(\s*[}\]])", r"\1", stripped)):
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
            # Accept {name, arguments}, {name, parameters}, and
            # OpenAI-style {type:"function", function:{name, arguments}}.
            if "function" in item and isinstance(item["function"], dict):
                inner = item["function"]
                name = inner.get("name", "") or ""
                raw = inner.get(args_field, inner.get("arguments", {}))
            else:
                name = item.get("name", "") or ""
                raw = item.get(args_field, item.get("arguments", item.get("parameters", {})))
            args = json.dumps(raw) if not isinstance(raw, str) else raw
            if name:
                out.append((str(name), args))
        if out:
            return out

    names = _NAME_RE.findall(body)
    if names:
        return [(n, body.strip()) for n in names]
    return []
