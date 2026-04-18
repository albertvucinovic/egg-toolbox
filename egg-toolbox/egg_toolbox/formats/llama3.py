"""Llama 3.x tool-call format handler.

Llama 3.1+ Instruct emits tool calls after a ``<|python_tag|>`` marker,
terminated by ``<|eom_id|>`` (or EOS).  Two body shapes are supported:

1. **JSON custom-tool form** (most common with user-provided tools)::

       <|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>

2. **Python call syntax** (built-in / fine-tuned tools)::

       <|python_tag|>get_weather(city="Paris")<|eom_id|>

Both collapse onto the same SemanticEvent stream: ``TOOL_CALL_START`` ->
``TOOL_CALL_NAME`` -> ``TOOL_ARGS_DELTA(json_string)`` -> ``TOOL_CALL_COMMIT``.
"""
from __future__ import annotations

import ast
import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


_PYCALL_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", re.DOTALL)


def _parse_llama3_body(body: str, args_field: str) -> tuple[str, str]:
    """Extract (name, arguments_json_string) from a <|python_tag|> body."""
    stripped = body.strip()

    # Try JSON first (custom tool form).
    candidates = [stripped]
    if stripped.startswith("{{") and stripped.endswith("}}"):
        candidates.append("{" + stripped[2:-2] + "}")
    candidates.append(re.sub(r",(\s*[}\]])", r"\1", stripped))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name", "") or ""
        raw_args = parsed.get(args_field, parsed.get("parameters", parsed.get("arguments", {})))
        args = json.dumps(raw_args) if not isinstance(raw_args, str) else raw_args
        return str(name), args

    # Python call syntax: func_name(k=v, k2=v2)
    m = _PYCALL_RE.match(stripped)
    if m:
        name = m.group(1)
        arg_src = m.group(2).strip()
        args_dict = _parse_kwargs(arg_src)
        return name, json.dumps(args_dict)

    return "", stripped


def _parse_kwargs(src: str) -> dict[str, Any]:
    """Parse a comma-separated ``k=v, k2=v2`` argument list.

    Uses ``ast.literal_eval`` so only JSON-compatible literals are accepted
    (no function calls, no attribute access, no arbitrary expressions).
    Unknown / non-literal values fall through as the raw string slice so
    information is never silently lost.
    """
    if not src:
        return {}
    # Wrap as a dummy call expression so ast can tokenise keyword args.
    try:
        node = ast.parse(f"_f({src})", mode="eval")
    except SyntaxError:
        return {"_raw": src}
    if not isinstance(node.body, ast.Call):
        return {"_raw": src}
    out: dict[str, Any] = {}
    for kw in node.body.keywords:
        if kw.arg is None:
            continue
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            out[kw.arg] = ast.unparse(kw.value)
    # Positional args (rare for tool calls) land in a list for completeness.
    if node.body.args:
        try:
            out["_args"] = [ast.literal_eval(a) for a in node.body.args]
        except (ValueError, SyntaxError):
            out["_args"] = [ast.unparse(a) for a in node.body.args]
    return out


class Llama3Handler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        # Llama 3 uses <|eom_id|> to close a tool-call message and
        # <|eot_id|> for turn end.  Either terminates generation.
        return ("<|eom_id|>", "<|eot_id|>")

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return Llama3ParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _Llama3State(enum.Enum):
    CONTENT   = "content"
    REASONING = "reasoning"
    IN_TOOL   = "in_tool"       # seen <|python_tag|>, accumulating body
    AFTER     = "after"         # post tool body


class Llama3ParserState(FormatParserState):
    _TRIGGER = "<|python_tag|>"
    _END_TAGS = ("<|eom_id|>", "<|eot_id|>")

    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _Llama3State.CONTENT
        self._buffer = ""
        self._body_buffer = ""
        self._tool_index = 0
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end
        self._args_field = analysis.args_field or "parameters"

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text

        while self._buffer:
            prev = self._buffer
            if self._state == _Llama3State.CONTENT:
                events.extend(self._process_content())
            elif self._state == _Llama3State.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _Llama3State.IN_TOOL:
                events.extend(self._process_in_tool())
            elif self._state == _Llama3State.AFTER:
                events.extend(self._process_after())
            if self._buffer == prev:
                break
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def _process_content(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []

        # Reasoning entry
        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _Llama3State.REASONING
            return events
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    return events

        # Full trigger match
        if self._TRIGGER in self._buffer:
            idx = self._buffer.index(self._TRIGGER)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._TRIGGER):]
            self._body_buffer = ""
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
            ))
            self._state = _Llama3State.IN_TOOL
            return events

        # Partial trigger suffix
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
            self._state = _Llama3State.CONTENT
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

    def _process_in_tool(self) -> list[SemanticEvent]:
        """Accumulate body until one of the end tags appears.

        We mirror Hermes' "buffer-then-emit" discipline: raw body chunks
        are not streamed as TOOL_ARGS_DELTA because for JSON bodies they
        mix the ``name`` field into arguments, and for Python-call bodies
        they carry the function name at the head.
        """
        events: list[SemanticEvent] = []

        first_end_idx = -1
        matched_tag = ""
        for tag in self._END_TAGS:
            idx = self._buffer.find(tag)
            if idx >= 0 and (first_end_idx < 0 or idx < first_end_idx):
                first_end_idx = idx
                matched_tag = tag

        if first_end_idx >= 0:
            self._body_buffer += self._buffer[:first_end_idx]
            self._buffer = self._buffer[first_end_idx + len(matched_tag):]
            events.extend(self._commit_tool())
            self._state = _Llama3State.AFTER
            return events

        # Partial end tag suffix
        max_partial = 0
        for tag in self._END_TAGS:
            for i in range(1, len(tag)):
                if self._buffer.endswith(tag[:i]):
                    max_partial = max(max_partial, i)
        if max_partial > 0:
            safe = self._buffer[:-max_partial]
            self._body_buffer += safe
            self._buffer = self._buffer[-max_partial:]
            return events

        self._body_buffer += self._buffer
        self._buffer = ""
        return events

    def _process_after(self) -> list[SemanticEvent]:
        """After a tool call, treat the rest as content unless another
        <|python_tag|> starts (parallel tool calls on separate lines)."""
        events: list[SemanticEvent] = []
        stripped = self._buffer.lstrip()
        if not stripped:
            self._buffer = ""
            return events

        if self._TRIGGER in stripped:
            idx = stripped.index(self._TRIGGER)
            self._buffer = stripped[idx + len(self._TRIGGER):]
            self._body_buffer = ""
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
            ))
            self._state = _Llama3State.IN_TOOL
            return events

        for i in range(1, len(self._TRIGGER)):
            if stripped.endswith(self._TRIGGER[:i]):
                self._buffer = stripped[-i:]
                return events

        self._buffer = stripped
        self._state = _Llama3State.CONTENT
        return events

    def _commit_tool(self) -> list[SemanticEvent]:
        name, args = _parse_llama3_body(self._body_buffer, self._args_field)
        self._body_buffer = ""
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

        if self._state == _Llama3State.IN_TOOL:
            self._body_buffer += self._buffer
            self._buffer = ""
            events.extend(self._commit_tool())
        elif self._buffer:
            if self._state in (_Llama3State.CONTENT, _Llama3State.AFTER):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _Llama3State.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _Llama3State.IN_TOOL
