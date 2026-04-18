"""Harmony multi-channel tool-call format handler.

Harmony models (OpenAI gpt-oss and related) partition their output
across named channels delimited by ``<|channel|>`` markers:

- **analysis**   -> internal chain-of-thought  (REASONING_DELTA)
- **commentary** -> tool calls as TypeScript  (TOOL_CALL_* events)
- **final**      -> user-facing answer         (CONTENT_DELTA)

This handler accepts two dialects of Harmony markers:

1. **Simplified channel markers** (used by the reference design in
   ``docs/egg-toolbox-architecture.md``)::

       <|analysis|>reasoning here<|commentary|>functions.get_weather({...})<|final|>answer here

2. **Full gpt-oss protocol** (OpenAI's public spec, with start/end/
   message/call/return tokens)::

       <|start|>assistant<|channel|>analysis<|message|>...<|end|>
       <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>
       <|start|>assistant<|channel|>final<|message|>...<|return|>

   In dialect 2 the function name is embedded in the channel header
   (``to=functions.NAME``) and the JSON arguments are the message
   body closed by ``<|call|>``.

Tool calls in commentary are written either as a JSON-mode message
(dialect 2) or as a TypeScript-style function invocation::

    namespace.functionName({ arg: "value", ... })

The regex used for dialect 1 captures the unqualified name (drops
the ``namespace.`` prefix) and the ``({...})`` body as the arguments.
"""
from __future__ import annotations

import enum
import json
import re
import uuid
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


# Dialect 1 — simplified channel markers.
_SIMPLE_CHANNEL_RE = re.compile(r"<\|(analysis|commentary|final)\|>")

# Dialect 2 — full gpt-oss:
#   <|start|>assistant<|channel|>NAME [to=...] [<|constrain|>json] <|message|>
_GPTOSS_HEADER_RE = re.compile(
    r"<\|start\|>assistant<\|channel\|>(analysis|commentary|final)"
    r"(?:\s+to=([A-Za-z_][\w.]*))?"
    r"(?:\s*<\|constrain\|>\w+)?"
    r"\s*<\|message\|>",
    re.DOTALL,
)
_GPTOSS_CLOSE_RE = re.compile(r"<\|(end|call|return)\|>")

# TypeScript-style call used inside simplified commentary channel:
#   namespace.functionName({ ... })  or  functionName({ ... })
_TS_CALL_RE = re.compile(r"([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+|[A-Za-z_][\w]*)\s*\(\s*(\{.*?\})\s*\)", re.DOTALL)


class HarmonyHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        # Harmony relies on natural EOS and its own channel markers;
        # returning stop strings here would truncate multi-channel
        # output prematurely.
        return ()

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return HarmonyParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        return None


class _HState(enum.Enum):
    IDLE       = "idle"
    ANALYSIS   = "analysis"       # dialect 1 channel
    COMMENTARY = "commentary"     # dialect 1 channel
    FINAL      = "final"          # dialect 1 channel
    # Dialect 2: each message is its own block between <|message|> and <|end|>/<|call|>/<|return|>
    GPTOSS_MESSAGE = "gptoss_message"


class HarmonyParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _HState.IDLE
        self._buffer = ""
        self._commentary_buffer = ""   # accumulates commentary prose for TS-call extraction
        self._gptoss_channel = ""       # active channel in dialect 2
        self._gptoss_tool_name = ""     # tool name extracted from channel header (dialect 2)
        self._gptoss_message = ""       # message body (dialect 2)
        self._tool_index = 0

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text
        while self._buffer:
            prev = self._buffer
            events.extend(self._step())
            if self._buffer == prev:
                break
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def _step(self) -> list[SemanticEvent]:
        if self._state == _HState.GPTOSS_MESSAGE:
            return self._process_gptoss_message()

        # Look for whichever opener appears first in the buffer.
        gptoss = _GPTOSS_HEADER_RE.search(self._buffer)
        simple = _SIMPLE_CHANNEL_RE.search(self._buffer)
        gptoss_idx = gptoss.start() if gptoss else -1
        simple_idx = simple.start() if simple else -1

        # Pick the earlier one.
        if gptoss_idx >= 0 and (simple_idx < 0 or gptoss_idx <= simple_idx):
            return self._enter_gptoss(gptoss)
        if simple_idx >= 0:
            return self._enter_simple(simple)

        # No marker in buffer.  Emit/buffer prefix in current dialect-1
        # channel (or swallow in IDLE).  Hold back partial markers.
        return self._flush_current(hold_partial=True)

    # ---------------- Dialect 1 (simplified channel markers) ----------------

    def _enter_simple(self, match: re.Match[str]) -> list[SemanticEvent]:
        events = self._flush_prefix(match.start())
        channel = match.group(1)
        events.extend(self._transition_to(channel))
        self._buffer = self._buffer[match.end():]
        return events

    def _flush_prefix(self, idx: int) -> list[SemanticEvent]:
        """Emit everything in the buffer up to index ``idx`` in the
        current channel's projection, then drop that prefix."""
        events: list[SemanticEvent] = []
        prefix = self._buffer[:idx]
        if prefix:
            events.extend(self._emit_text(prefix))
        self._buffer = self._buffer[idx:]
        return events

    def _flush_current(self, hold_partial: bool) -> list[SemanticEvent]:
        """No complete marker in buffer.  Emit what's safe in the
        current channel, hold back anything that could still resolve
        to a Harmony marker.

        We hold from the FIRST ``<|`` onward -- once ``<|`` has
        arrived, the next several tokens could complete either a
        dialect-1 channel marker or a dialect-2 header.  We also hold
        a single trailing ``<`` since it could be the start of
        ``<|``.
        """
        events: list[SemanticEvent] = []
        buf = self._buffer
        if not buf:
            return events

        if hold_partial:
            idx = buf.find("<|")
            if idx >= 0:
                safe = buf[:idx]
                if safe:
                    events.extend(self._emit_text(safe))
                self._buffer = buf[idx:]
                return events
            if buf.endswith("<"):
                safe = buf[:-1]
                if safe:
                    events.extend(self._emit_text(safe))
                self._buffer = buf[-1:]
                return events

        events.extend(self._emit_text(buf))
        self._buffer = ""
        return events

    def _emit_text(self, text: str) -> list[SemanticEvent]:
        if self._state == _HState.ANALYSIS:
            return [SemanticEvent(kind=EventKind.REASONING_DELTA, text=text)]
        if self._state == _HState.FINAL:
            return [SemanticEvent(kind=EventKind.CONTENT_DELTA, text=text)]
        if self._state == _HState.COMMENTARY:
            self._commentary_buffer += text
            return []
        # IDLE -- discard
        return []

    def _transition_to(self, channel: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._state == _HState.COMMENTARY and self._commentary_buffer:
            events.extend(self._flush_commentary())
        self._state = {
            "analysis": _HState.ANALYSIS,
            "commentary": _HState.COMMENTARY,
            "final": _HState.FINAL,
        }[channel]
        return events

    def _flush_commentary(self) -> list[SemanticEvent]:
        """Extract TypeScript-style function calls from the commentary
        buffer and emit TOOL_CALL events."""
        events: list[SemanticEvent] = []
        for match in _TS_CALL_RE.finditer(self._commentary_buffer):
            raw_name = match.group(1)
            name = raw_name.split(".")[-1]
            args_src = match.group(2)
            args = _normalise_args(args_src)
            events.extend(self._emit_tool_call(name, args))
        self._commentary_buffer = ""
        return events

    # ---------------- Dialect 2 (full gpt-oss) ----------------

    def _enter_gptoss(self, match: re.Match[str]) -> list[SemanticEvent]:
        events = self._flush_prefix(match.start())
        self._gptoss_channel = match.group(1)
        self._gptoss_tool_name = match.group(2) or ""
        self._gptoss_message = ""
        self._buffer = self._buffer[match.end():]
        # On commentary entry, pre-emit TOOL_CALL_START+NAME; the body
        # will be args and we commit on <|call|>.
        if self._gptoss_channel == "commentary" and self._gptoss_tool_name:
            name = self._gptoss_tool_name.split(".")[-1]
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
            ))
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_NAME,
                tool_index=self._tool_index,
                tool_name=name,
            ))
        self._state = _HState.GPTOSS_MESSAGE
        return events

    def _process_gptoss_message(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        close = _GPTOSS_CLOSE_RE.search(self._buffer)
        if close:
            self._gptoss_message += self._buffer[:close.start()]
            self._buffer = self._buffer[close.end():]
            events.extend(self._commit_gptoss_message())
            self._state = _HState.IDLE
            return events

        # Hold back a partial close marker suffix.
        for marker in ("<|end|>", "<|call|>", "<|return|>", "<|"):
            for i in range(1, len(marker)):
                if self._buffer.endswith(marker[:i]):
                    self._gptoss_message += self._buffer[:-i]
                    self._buffer = self._buffer[-i:]
                    return events

        self._gptoss_message += self._buffer
        self._buffer = ""
        return events

    def _commit_gptoss_message(self) -> list[SemanticEvent]:
        body = self._gptoss_message
        self._gptoss_message = ""
        if self._gptoss_channel == "analysis":
            return [SemanticEvent(kind=EventKind.REASONING_DELTA, text=body)] if body else []
        if self._gptoss_channel == "final":
            return [SemanticEvent(kind=EventKind.CONTENT_DELTA, text=body)] if body else []
        if self._gptoss_channel == "commentary":
            name = (self._gptoss_tool_name or "").split(".")[-1]
            self._gptoss_tool_name = ""
            if name:
                args = _normalise_args(body)
                return self._finalise_started_tool_call(args)
            # No function name -- fall through as prose content.
            return [SemanticEvent(kind=EventKind.CONTENT_DELTA, text=body)] if body else []
        return []

    def _emit_tool_call(self, name: str, args: str) -> list[SemanticEvent]:
        """Emit a fresh tool call (START + NAME + ARGS_DELTA + COMMIT)."""
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        events = [
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
        ]
        self._tool_index += 1
        return events

    def _finalise_started_tool_call(self, args: str) -> list[SemanticEvent]:
        """Close a tool call whose START+NAME was emitted eagerly
        (dialect 2 path).  Emits only the trailing ARGS_DELTA + COMMIT."""
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        name_field = ""  # ignored here; COMMIT just needs consistent id
        events = [
            SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=args),
            SemanticEvent(
                kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
                tool_name=name_field,
                tool_arguments=args,
            ),
        ]
        self._tool_index += 1
        return events

    def finish(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._state == _HState.GPTOSS_MESSAGE:
            self._gptoss_message += self._buffer
            self._buffer = ""
            events.extend(self._commit_gptoss_message())
        elif self._buffer:
            events.extend(self._flush_current(hold_partial=False))
        if self._commentary_buffer:
            events.extend(self._flush_commentary())

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return (
            self._state == _HState.COMMENTARY
            or (self._state == _HState.GPTOSS_MESSAGE and self._gptoss_channel == "commentary")
        )


def _normalise_args(body: str) -> str:
    stripped = body.strip()
    for cand in (stripped, re.sub(r",(\s*[}\]])", r"\1", stripped)):
        try:
            parsed = json.loads(cand)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            continue
    return stripped
