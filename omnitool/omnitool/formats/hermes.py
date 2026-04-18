import enum
import json
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


class HermesHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        end = self.analysis.tool_call_end or "</tool_call>"
        return (end,)

    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools: list[Tool] | None = None) -> FormatParserState:
        return HermesParserState(self.analysis, tools)

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        # Phase 2: implement GBNF grammar generation
        return None


class _HermesState(enum.Enum):
    CONTENT         = "content"          # emitting content_delta
    REASONING       = "reasoning"        # inside <think>...</think>
    MAYBE_TAG       = "maybe_tag"        # accumulating '<' prefix
    IN_TOOL_TAG     = "in_tool_tag"      # inside <tool_call>, accumulating JSON
    AFTER_TOOL_CALL = "after_tool_call"  # between </tool_call> and next <tool_call> or EOS


class HermesParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _HermesState.CONTENT
        self._buffer = ""              # accumulation buffer for partial matches
        self._json_buffer = ""         # accumulating JSON inside <tool_call>
        self._tool_index = 0           # current tool call index (0-based)
        self._committed_tools: list[dict] = []
        self._tag_start = analysis.tool_call_start or "<tool_call>"
        self._tag_end = analysis.tool_call_end or "</tool_call>"
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text

        while self._buffer:
            prev_buffer = self._buffer
            if self._state == _HermesState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _HermesState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _HermesState.MAYBE_TAG:
                events.extend(self._process_maybe_tag())
            elif self._state == _HermesState.IN_TOOL_TAG:
                events.extend(self._process_in_tool())
            elif self._state == _HermesState.AFTER_TOOL_CALL:
                events.extend(self._process_after_tool())
            else:
                break
            # Prevent infinite loop: if buffer didn't change and state didn't change
            if self._buffer == prev_buffer:
                break
        return events

    def _process_content(self) -> list[SemanticEvent]:
        """In CONTENT state: emit content deltas, watch for <tool_call> or <think>."""
        events: list[SemanticEvent] = []

        # Check for reasoning start
        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _HermesState.REASONING
            return events

        # Check if buffer could be a partial reasoning start
        if self._reason_start:
            for i in range(1, len(self._reason_start)):
                if self._buffer == self._reason_start[:i]:
                    # Buffer is an exact partial match -- hold back
                    return events

        # Check for tool_call start tag (complete match)
        if self._tag_start in self._buffer:
            idx = self._buffer.index(self._tag_start)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._tag_start):]
            self._json_buffer = ""
            tool_call_id = f"call_{self._tool_index}"
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
            ))
            self._state = _HermesState.IN_TOOL_TAG
            return events

        # Check if buffer might be start of a tag (partial match)
        for i in range(1, len(self._tag_start)):
            if self._buffer.endswith(self._tag_start[:i]):
                # Hold back the potential tag prefix
                safe = self._buffer[:-i]
                if safe:
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
                self._buffer = self._buffer[-i:]
                return events

        # No tag match possible -- emit entire buffer
        events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
        self._buffer = ""
        return events

    def _process_in_tool(self) -> list[SemanticEvent]:
        """In IN_TOOL_TAG state: accumulate JSON, emit args deltas, watch for </tool_call>."""
        events: list[SemanticEvent] = []
        if self._tag_end in self._buffer:
            idx = self._buffer.index(self._tag_end)
            json_chunk = self._buffer[:idx]
            self._json_buffer += json_chunk
            self._buffer = self._buffer[idx + len(self._tag_end):]

            # Parse the complete JSON
            try:
                parsed = json.loads(self._json_buffer)
                name = parsed.get(self._analysis.name_field, parsed.get("name", ""))
                args_raw = parsed.get(self._analysis.args_field, parsed.get("arguments", {}))
                args = json.dumps(args_raw) if not isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                name = ""
                args = self._json_buffer

            events.append(SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name))
            events.append(SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=args))
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
                tool_name=name,
                tool_arguments=args,
            ))
            self._committed_tools.append({"name": name, "arguments": args})
            self._tool_index += 1
            self._state = _HermesState.AFTER_TOOL_CALL
        else:
            # Check for partial end tag
            has_partial = False
            for i in range(1, len(self._tag_end)):
                if self._buffer.endswith(self._tag_end[:i]):
                    # Stream the safe part as args delta
                    safe = self._buffer[:-i]
                    if safe:
                        self._json_buffer += safe
                        events.append(SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=safe))
                    self._buffer = self._buffer[-i:]
                    has_partial = True
                    break
            if not has_partial:
                # Stream args deltas as they arrive
                self._json_buffer += self._buffer
                events.append(SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=self._buffer))
                self._buffer = ""
        return events

    def _process_reasoning(self) -> list[SemanticEvent]:
        """In REASONING state: emit reasoning deltas, watch for </think>."""
        events: list[SemanticEvent] = []
        if self._reason_end and self._reason_end in self._buffer:
            idx = self._buffer.index(self._reason_end)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._reason_end):]
            self._state = _HermesState.CONTENT
        else:
            # Check for partial reason_end at end of buffer
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

    def _process_maybe_tag(self) -> list[SemanticEvent]:
        """In MAYBE_TAG state: waiting for more data to confirm/deny a tag."""
        events: list[SemanticEvent] = []
        # This state is reached when we had a partial match
        # Try to match the full tag_start
        if self._tag_start.startswith(self._buffer):
            if len(self._buffer) >= len(self._tag_start):
                # Full match
                self._buffer = self._buffer[len(self._tag_start):]
                self._json_buffer = ""
                tool_call_id = f"call_{self._tool_index}"
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_CALL_START,
                    tool_index=self._tool_index,
                    tool_call_id=tool_call_id,
                ))
                self._state = _HermesState.IN_TOOL_TAG
            # else: still partial, wait for more data
        else:
            # Not a tag -- emit as content and return to CONTENT state
            events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
            self._buffer = ""
            self._state = _HermesState.CONTENT
        return events

    def _process_after_tool(self) -> list[SemanticEvent]:
        """In AFTER_TOOL_CALL state: look for another <tool_call> or transition back."""
        events: list[SemanticEvent] = []

        # Skip whitespace/newlines between tool calls
        stripped = self._buffer.lstrip()
        if not stripped:
            self._buffer = ""
            return events

        # Check for another tool_call start
        if self._tag_start in stripped:
            idx = stripped.index(self._tag_start)
            self._buffer = stripped[idx + len(self._tag_start):]
            self._json_buffer = ""
            tool_call_id = f"call_{self._tool_index}"
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
            ))
            self._state = _HermesState.IN_TOOL_TAG
            return events

        # Check partial match
        for i in range(1, len(self._tag_start)):
            if stripped.endswith(self._tag_start[:i]):
                self._buffer = stripped[-i:]
                return events

        # No more tool calls -- transition back to content
        self._buffer = stripped
        self._state = _HermesState.CONTENT
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def finish(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []

        # Commit any in-progress tool call (buffer may be empty if the
        # orchestrator's stop-string matcher already consumed </tool_call>).
        if self._state == _HermesState.IN_TOOL_TAG:
            self._json_buffer += self._buffer
            self._buffer = ""
            try:
                parsed = json.loads(self._json_buffer)
                name = parsed.get(self._analysis.name_field, parsed.get("name", ""))
                args_raw = parsed.get(self._analysis.args_field, parsed.get("arguments", {}))
                args = json.dumps(args_raw) if not isinstance(args_raw, str) else args_raw
                events.append(SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name))
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_CALL_COMMIT,
                    tool_index=self._tool_index,
                    tool_call_id=f"call_{self._tool_index}",
                    tool_name=name,
                    tool_arguments=args,
                ))
                self._tool_index += 1
            except json.JSONDecodeError:
                pass
        elif self._buffer:
            if self._state in (_HermesState.CONTENT, _HermesState.AFTER_TOOL_CALL):
                if self._buffer.strip():
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""
            elif self._state == _HermesState.REASONING:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _HermesState.IN_TOOL_TAG
