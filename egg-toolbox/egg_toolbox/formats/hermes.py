import enum
import json
import re
from typing import Any

from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


_NAME_FALLBACK = re.compile(r'"name"\s*:\s*"([^"]+)"')
_ARGS_FALLBACK = re.compile(r'"arguments"\s*:\s*(\{.*\}|\[.*\]|"[^"]*"|[^,}\s]+)', re.DOTALL)


_ESC_MAP = {
    '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
    "n": "\n", "r": "\r", "t": "\t",
}


class _StreamingBodyExtractor:
    """Projects the name and arguments values out of a Hermes <tool_call>
    body as chars arrive.

    The body is a JSON object with two interesting keys (``name`` and
    ``arguments``) plus possibly some other keys we ignore.  The
    extractor runs a small JSON state machine char-by-char.  Chars
    belonging to the name value accumulate into ``_name_buf``; chars
    belonging to the arguments value accumulate into ``_args_buf``.
    Callers drive via ``feed_chars`` and harvest fragments via
    ``drain_name_if_ready`` / ``drain_args``.

    On malformed input the extractor sets ``errored = True`` and stops
    emitting; callers should fall back to ``_parse_tool_call_body`` on
    the full body.
    """

    # Top-level parser states (only meaningful at depth 1)
    _EXPECT_OPEN  = 0
    _EXPECT_KEY   = 1
    _EXPECT_COLON = 2
    _EXPECT_VALUE = 3
    _EXPECT_COMMA = 4
    _AFTER_CLOSE  = 5

    def __init__(self, name_field: str = "name", args_field: str = "arguments"):
        self._name_field = name_field
        self._args_field = args_field

        self._state = self._EXPECT_OPEN
        self._depth = 0                 # JSON nesting depth
        self._in_string = False
        self._escape = False

        # Key accumulation (at depth 1, while reading a key string).
        self._current_key = ""
        self._last_key = ""

        # Projection bookkeeping: exactly one of these is True at a time
        # (or all False when we're outside an interesting value).
        self._in_name_string = False
        self._in_args_string = False
        self._in_args_nested = False
        self._in_args_literal = False
        self._args_nested_depth = 0     # local depth inside args {..}/[..]

        # "Ignoring" modes for values of uninteresting keys -- we still
        # need to track syntax to know when the value ends.
        self._ignore_value_kind = None  # None | "string" | "nested" | "literal"
        self._ignore_nested_depth = 0

        # Outputs
        self._name_buf: list[str] = []
        self._name_done = False
        self._name_emitted = False
        self._args_buf: list[str] = []
        self._full_body: list[str] = []

        # Error flag -- caller falls back on the complete body.
        self.errored = False

    # --------- public interface ---------------------------------

    def feed_chars(self, text: str) -> None:
        """Process a run of characters.  Never raises; sets
        ``errored`` on bad input."""
        for ch in text:
            self._full_body.append(ch)
            if self.errored:
                continue
            try:
                self._consume(ch)
            except Exception:
                self.errored = True

    def drain_args(self) -> str:
        """Return args chars accumulated since last drain."""
        if not self._args_buf:
            return ""
        out = "".join(self._args_buf)
        self._args_buf.clear()
        return out

    def drain_name_if_ready(self) -> str | None:
        """If the name value has fully arrived and hasn't been emitted,
        return it; else None.  Emits at most once."""
        if self._name_done and not self._name_emitted:
            self._name_emitted = True
            return "".join(self._name_buf)
        return None

    def full_body(self) -> str:
        return "".join(self._full_body)

    def name(self) -> str:
        """Best-effort name (may be partial if the body cut off early)."""
        return "".join(self._name_buf)

    # --------- state machine ------------------------------------

    def _consume(self, ch: str) -> None:
        # Enter the outer {
        if self._state == self._EXPECT_OPEN:
            if ch.isspace():
                return
            if ch == "{":
                self._depth = 1
                self._state = self._EXPECT_KEY
                return
            raise ValueError(f"expected {{, got {ch!r}")

        # Inside some string: character goes to whichever projection
        # is active (key, name value, args string, args nested string,
        # or ignored).
        if self._in_string:
            self._consume_in_string(ch)
            return

        # Inside an args-nested value: copy every char (except we still
        # track "..." as string for balanced brace counting).
        if self._in_args_nested:
            self._args_buf.append(ch)
            if ch == '"':
                self._in_string = True
                # we're inside a string within the nested args; marked
                # neither name nor args_string -- just "ignore quoting
                # context" while still routing chars to _args_buf.
                # We set a combined flag by making _in_args_nested true
                # AND _in_string true; _consume_in_string handles this.
            elif ch in "{[":
                self._args_nested_depth += 1
                self._depth += 1
            elif ch in "}]":
                self._args_nested_depth -= 1
                self._depth -= 1
                if self._args_nested_depth == 0:
                    self._in_args_nested = False
                    self._state = self._EXPECT_COMMA
            return

        # Inside an args literal (number/bool/null): copy until ender.
        if self._in_args_literal:
            if ch in ",}] \t\n\r":
                self._in_args_literal = False
                self._state = self._EXPECT_COMMA
                # fall through to re-process ch below
            else:
                self._args_buf.append(ch)
                return

        # Inside an ignored nested value: just track balance.
        if self._ignore_value_kind == "nested":
            if ch == '"':
                self._in_string = True
                return
            if ch in "{[":
                self._ignore_nested_depth += 1
                self._depth += 1
                return
            if ch in "}]":
                self._ignore_nested_depth -= 1
                self._depth -= 1
                if self._ignore_nested_depth == 0:
                    self._ignore_value_kind = None
                    self._state = self._EXPECT_COMMA
                return
            return

        # Inside an ignored literal: skip until ender.
        if self._ignore_value_kind == "literal":
            if ch in ",}] \t\n\r":
                self._ignore_value_kind = None
                self._state = self._EXPECT_COMMA
                # fall through
            else:
                return

        # Structural consumption at top level.
        if ch.isspace():
            return

        if self._state == self._EXPECT_KEY:
            if ch == '"':
                self._in_string = True
                self._current_key = ""
            elif ch == "}":
                self._depth -= 1
                self._state = self._AFTER_CLOSE
            else:
                raise ValueError(f"expected key or }}, got {ch!r}")
            return

        if self._state == self._EXPECT_COLON:
            if ch == ":":
                self._state = self._EXPECT_VALUE
                return
            raise ValueError(f"expected :, got {ch!r}")

        if self._state == self._EXPECT_VALUE:
            if self._last_key == self._name_field:
                if ch == '"':
                    self._in_string = True
                    self._in_name_string = True
                    return
                # name is not a string -- treat as ignored (degraded).
                self._enter_ignored_value(ch)
                return
            if self._last_key == self._args_field:
                if ch in "{[":
                    self._args_buf.append(ch)
                    self._in_args_nested = True
                    self._args_nested_depth = 1
                    self._depth += 1
                    return
                if ch == '"':
                    self._in_string = True
                    self._in_args_string = True
                    return
                # Literal value.
                self._args_buf.append(ch)
                self._in_args_literal = True
                return
            # Uninteresting key's value -- skip.
            self._enter_ignored_value(ch)
            return

        if self._state == self._EXPECT_COMMA:
            if ch == ",":
                self._state = self._EXPECT_KEY
                return
            if ch == "}":
                self._depth -= 1
                self._state = self._AFTER_CLOSE
                return
            raise ValueError(f"expected , or }}, got {ch!r}")

        if self._state == self._AFTER_CLOSE:
            # trailing junk -- ignore.
            return

    def _enter_ignored_value(self, ch: str) -> None:
        if ch == '"':
            self._in_string = True
            self._ignore_value_kind = "string"
        elif ch in "{[":
            self._depth += 1
            self._ignore_nested_depth = 1
            self._ignore_value_kind = "nested"
        else:
            self._ignore_value_kind = "literal"

    def _consume_in_string(self, ch: str) -> None:
        """Character dispatch while ``_in_string`` is True."""
        # Which bucket does this char flow into?
        # Priority:
        #   1. _in_name_string -> name_buf
        #   2. _in_args_string -> args_buf (JSON-unescaped content)
        #   3. _in_args_nested -> args_buf (verbatim including escapes)
        #   4. _state == EXPECT_KEY and _in_string -> current_key
        #   5. _ignore_value_kind == "string" -> drop
        if self._escape:
            self._escape = False
            decoded = _ESC_MAP.get(ch, ch)
            if self._in_name_string:
                self._name_buf.append(decoded)
            elif self._in_args_string:
                self._args_buf.append(decoded)
            elif self._in_args_nested:
                # Preserve the escape sequence verbatim in the nested blob.
                self._args_buf.append("\\")
                self._args_buf.append(ch)
            # key / ignored: drop
            return
        if ch == "\\":
            self._escape = True
            return
        if ch == '"':
            # closing quote
            self._in_string = False
            if self._in_name_string:
                self._in_name_string = False
                self._name_done = True
                self._state = self._EXPECT_COMMA
            elif self._in_args_string:
                self._in_args_string = False
                self._state = self._EXPECT_COMMA
            elif self._in_args_nested:
                # Close quote is part of the nested JSON value.
                self._args_buf.append('"')
            elif self._state == self._EXPECT_KEY:
                # End of key.
                self._last_key = self._current_key
                self._state = self._EXPECT_COLON
            elif self._ignore_value_kind == "string":
                self._ignore_value_kind = None
                self._state = self._EXPECT_COMMA
            return
        # Regular string char.
        if self._in_name_string:
            self._name_buf.append(ch)
        elif self._in_args_string:
            self._args_buf.append(ch)
        elif self._in_args_nested:
            self._args_buf.append(ch)
        elif self._state == self._EXPECT_KEY:
            self._current_key += ch
        # ignored string: drop


def _parse_tool_call_body(body: str, name_field: str, args_field: str) -> tuple[str, str]:
    """Extract (name, arguments_string) from a <tool_call> body.

    Attempts strict json.loads first, then tolerant repair
    (strip leading/trailing noise, collapse doubled braces, strip
    trailing commas), then regex fallback.  Returns empty strings
    if nothing usable can be extracted.
    """
    candidates = [body, body.strip()]
    stripped = body.strip()
    # Collapse a doubled leading brace ("{{" -> "{") which real-world
    # models occasionally emit as a single BPE token.
    if stripped.startswith("{{") and stripped.endswith("}}"):
        candidates.append("{" + stripped[2:-2] + "}")
    elif stripped.startswith("{{"):
        candidates.append("{" + stripped[2:])
    # Strip trailing commas before } or ]
    candidates.append(re.sub(r",(\s*[}\]])", r"\1", stripped))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get(name_field, parsed.get("name", "")) or ""
        raw = parsed.get(args_field, parsed.get("arguments", {}))
        args = json.dumps(raw) if not isinstance(raw, str) else raw
        return str(name), args

    # Regex fallback for unparseable bodies.
    name_match = _NAME_FALLBACK.search(body)
    args_match = _ARGS_FALLBACK.search(body)
    name = name_match.group(1) if name_match else ""
    args = args_match.group(1) if args_match else body.strip()
    return name, args


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
        self._tool_index = 0           # current tool call index (0-based)
        self._committed_tools: list[dict] = []
        self._tag_start = analysis.tool_call_start or "<tool_call>"
        self._tag_end = analysis.tool_call_end or "</tool_call>"
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end
        # Active streaming extractor inside a <tool_call> body.  Created
        # fresh on every open-tag, harvested/committed on close-tag.
        self._extractor: _StreamingBodyExtractor | None = None

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
            self._extractor = _StreamingBodyExtractor(
                self._analysis.name_field, self._analysis.args_field,
            )
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
        """In IN_TOOL_TAG state: stream the <tool_call> body through the
        extractor, emitting TOOL_CALL_NAME as soon as the name closes
        and one TOOL_ARGS_DELTA per token that contributes args chars.

        On ``</tool_call>`` we close the extractor and emit COMMIT.  If
        the extractor's JSON parse went wrong (doubled braces etc.) the
        commit step falls back to ``_parse_tool_call_body`` on the full
        body so robustness is preserved.
        """
        events: list[SemanticEvent] = []
        assert self._extractor is not None

        if self._tag_end in self._buffer:
            idx = self._buffer.index(self._tag_end)
            self._extractor.feed_chars(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(self._tag_end):]
            # Harvest final name + args fragment before committing.
            name_ready = self._extractor.drain_name_if_ready()
            if name_ready is not None:
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_CALL_NAME,
                    tool_index=self._tool_index,
                    tool_name=name_ready,
                ))
            final_args = self._extractor.drain_args()
            if final_args:
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_ARGS_DELTA,
                    tool_index=self._tool_index,
                    text=final_args,
                ))
            events.extend(self._commit_streaming_tool_call())
            self._state = _HermesState.AFTER_TOOL_CALL
            return events

        # Look for a partial end tag suffix so we don't eat into "</"
        for i in range(1, len(self._tag_end)):
            if self._buffer.endswith(self._tag_end[:i]):
                self._extractor.feed_chars(self._buffer[:-i])
                self._buffer = self._buffer[-i:]
                events.extend(self._drain_extractor_events())
                return events

        # No partial match -- absorb the whole buffer into extractor.
        self._extractor.feed_chars(self._buffer)
        self._buffer = ""
        events.extend(self._drain_extractor_events())
        return events

    def _drain_extractor_events(self) -> list[SemanticEvent]:
        """Harvest any pending name/args fragments from the extractor.

        Called at the end of each token-chunk feed so that each backend
        token that contributed to the name or args produces at most one
        NAME (first time only) and one ARGS_DELTA event.
        """
        assert self._extractor is not None
        events: list[SemanticEvent] = []
        name_ready = self._extractor.drain_name_if_ready()
        if name_ready is not None:
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_NAME,
                tool_index=self._tool_index,
                tool_name=name_ready,
            ))
        args_delta = self._extractor.drain_args()
        if args_delta:
            events.append(SemanticEvent(
                kind=EventKind.TOOL_ARGS_DELTA,
                tool_index=self._tool_index,
                text=args_delta,
            ))
        return events

    def _commit_streaming_tool_call(self) -> list[SemanticEvent]:
        """Emit TOOL_CALL_COMMIT carrying the full reassembled name +
        args.  Deltas have already gone out during streaming; COMMIT
        carries the canonical parsed values for non-streaming consumers
        (which accumulate from COMMIT, not from deltas).

        Always advances ``_tool_index`` -- the model clearly *intended*
        a tool call by opening <tool_call>.
        """
        assert self._extractor is not None
        extractor = self._extractor
        self._extractor = None

        # Parse the full body once through our tolerant parser.  If
        # strict JSON fails, regex fallbacks kick in; as a last resort
        # it returns the extractor's partial name and empty args.
        name, args = _parse_tool_call_body(
            extractor.full_body(),
            self._analysis.name_field,
            self._analysis.args_field,
        )
        if not name:
            name = extractor.name()

        tool_call_id = f"call_{self._tool_index}"
        events = [
            SemanticEvent(
                kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
                tool_name=name,
                tool_arguments=args,
            ),
        ]
        self._committed_tools.append({"name": name, "arguments": args})
        self._tool_index += 1
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
                self._extractor = _StreamingBodyExtractor(
                    self._analysis.name_field, self._analysis.args_field,
                )
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
            self._extractor = _StreamingBodyExtractor(
                self._analysis.name_field, self._analysis.args_field,
            )
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

        # Commit any in-progress tool call.  The orchestrator's
        # stop-string matcher typically consumes </tool_call> before it
        # ever reaches us, so reaching finish() in IN_TOOL_TAG is the
        # normal path for the real backend (not an error).
        if self._state == _HermesState.IN_TOOL_TAG:
            # Flush any remaining buffered chars through the extractor,
            # drain final name/args, then commit.  Matches the normal
            # </tool_call> path but without the closing tag in the
            # token stream.
            if self._extractor is None:
                self._extractor = _StreamingBodyExtractor(
                    self._analysis.name_field, self._analysis.args_field,
                )
            self._extractor.feed_chars(self._buffer)
            self._buffer = ""
            name_ready = self._extractor.drain_name_if_ready()
            if name_ready is not None:
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_CALL_NAME,
                    tool_index=self._tool_index,
                    tool_name=name_ready,
                ))
            final_args = self._extractor.drain_args()
            if final_args:
                events.append(SemanticEvent(
                    kind=EventKind.TOOL_ARGS_DELTA,
                    tool_index=self._tool_index,
                    text=final_args,
                ))
            events.extend(self._commit_streaming_tool_call())
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
