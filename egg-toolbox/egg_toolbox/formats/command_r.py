"""Cohere Command-R / R+ tool-call format handler.

Command-R wraps tool calls in a single pair of markers with a JSON
array body that uses ``tool_name`` / ``parameters`` (not
``name`` / ``arguments``)::

    <|START_ACTION|>[
      {"tool_name": "get_weather", "parameters": {"location": "Zagreb"}}
    ]<|END_ACTION|>

Multiple calls can share one array.  The body is always a JSON array;
we apply the same bracket-aware scanning used for Mistral to detect
the true end of the array (relevant if the model emits the markers
but truncates early).
"""
from __future__ import annotations

import enum
import json
import re
import uuid
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


_BEGIN = "<|START_ACTION|>"
_END = "<|END_ACTION|>"


class CommandRHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return CommandRParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _CRState(enum.Enum):
    CONTENT   = "content"
    REASONING = "reasoning"
    IN_ACTION = "in_action"
    AFTER     = "after"


class CommandRParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _CRState.CONTENT
        self._buffer = ""
        self._body_buffer = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            if self._state == _CRState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _CRState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _CRState.IN_ACTION:
                events.extend(self._process_in_action())
            elif self._state == _CRState.AFTER:
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
            self._state = _CRState.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        if _BEGIN in self._buffer:
            idx = self._buffer.index(_BEGIN)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(_BEGIN):]
            self._body_buffer = ""
            self._state = _CRState.IN_ACTION
            return events

        for i in range(1, len(_BEGIN)):
            if self._buffer.endswith(_BEGIN[:i]):
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
            self._state = _CRState.CONTENT
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

    def _process_in_action(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if _END in self._buffer:
            idx = self._buffer.index(_END)
            self._body_buffer += self._buffer[:idx]
            self._buffer = self._buffer[idx + len(_END):]
            events.extend(self._commit_action())
            self._state = _CRState.AFTER
            return events

        for i in range(1, len(_END)):
            if self._buffer.endswith(_END[:i]):
                safe = self._buffer[:-i]
                self._body_buffer += safe
                self._buffer = self._buffer[-i:]
                return events

        self._body_buffer += self._buffer
        self._buffer = ""
        return events

    def _process_after(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        stripped = self._buffer.lstrip()
        if not stripped:
            self._buffer = ""
            return events
        self._buffer = stripped
        self._state = _CRState.CONTENT
        return events

    def _commit_action(self) -> list[SemanticEvent]:
        calls = _parse_command_r_body(self._body_buffer)
        self._body_buffer = ""
        events: list[SemanticEvent] = []
        if not calls:
            calls = [("", "")]
        for name, args in calls:
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
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
        if self._state == _CRState.IN_ACTION:
            self._body_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_action())
        elif self._buffer:
            if self._state in (_CRState.CONTENT, _CRState.AFTER):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _CRState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _CRState.IN_ACTION


_NAME_RE = re.compile(r'"tool_name"\s*:\s*"([^"]+)"')


def _parse_command_r_body(body: str) -> list[tuple[str, str]]:
    """Parse a Command-R action body.

    Expects a JSON array of objects with ``tool_name`` and
    ``parameters``.  Also tolerates a single object (some fine-tunes)
    and falls back to regex name extraction on malformed bodies.
    """
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
            name = item.get("tool_name", item.get("name", "")) or ""
            raw = item.get("parameters", item.get("arguments", {}))
            args = json.dumps(raw) if not isinstance(raw, str) else raw
            if name:
                out.append((str(name), args))
        if out:
            return out

    names = _NAME_RE.findall(body)
    if names:
        return [(n, body.strip()) for n in names]
    return []
