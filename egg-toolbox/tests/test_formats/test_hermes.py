"""Tests for the Hermes format handler.

Priority: verify that tool_call arguments are emitted as a stream of
TOOL_ARGS_DELTA events *as the body arrives*, not buffered until the
closing tag.  The prior implementation buffered the entire body and
emitted a single delta on close, which meant clients saw no partial
arguments and no name until the tool_call was fully generated --
awful UX for long argument payloads.
"""
from __future__ import annotations

import pytest

from egg_toolbox.formats.hermes import HermesParserState
from egg_toolbox.types import (
    EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode,
)


def _make_state() -> HermesParserState:
    analysis = FormatAnalysis(
        tool_mode=ToolFormatMode.HERMES,
        reasoning_mode=ReasoningMode.NONE,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        name_field="name",
        args_field="arguments",
    )
    return HermesParserState(analysis, tools=None)


def _feed(state: HermesParserState, text: str, chunk: int = 1) -> list:
    """Feed text in fixed-size chunks, returning the flat event list."""
    events = []
    i = 0
    while i < len(text):
        piece = text[i:i + chunk]
        events.extend(state.feed_token(-1, piece))
        i += chunk
    events.extend(state.finish())
    return events


def _kinds(events):
    return [e.kind for e in events]


def _collect_args_text(events) -> str:
    return "".join(
        e.text for e in events
        if e.kind == EventKind.TOOL_ARGS_DELTA and e.text
    )


def _collect_content_text(events) -> str:
    return "".join(
        e.text for e in events
        if e.kind == EventKind.CONTENT_DELTA and e.text
    )


# ---------------------------------------------------------------
# Baseline: the body fed in one big chunk still works end-to-end.
# ---------------------------------------------------------------

def test_single_chunk_body_emits_name_args_commit():
    state = _make_state()
    text = (
        'Here you go: <tool_call>{"name": "get_weather", '
        '"arguments": {"city": "Zagreb"}}</tool_call>'
    )
    events = _feed(state, text, chunk=len(text))
    kinds = _kinds(events)

    assert EventKind.TOOL_CALL_START in kinds
    assert EventKind.TOOL_CALL_NAME in kinds
    assert EventKind.TOOL_ARGS_DELTA in kinds
    assert EventKind.TOOL_CALL_COMMIT in kinds
    assert kinds[-1] == EventKind.DONE

    # Name event must carry the full name.
    name_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_NAME)
    assert name_ev.tool_name == "get_weather"

    # Combined args deltas must equal the JSON-serialized args.
    combined = _collect_args_text(events)
    assert combined == '{"city": "Zagreb"}' or combined.replace(" ", "") == '{"city":"Zagreb"}'

    # COMMIT carries the full args string for non-streaming consumers.
    commit_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_COMMIT)
    assert commit_ev.tool_name == "get_weather"
    assert commit_ev.tool_arguments.replace(" ", "") == '{"city":"Zagreb"}'

    # DONE reports tool_calls as the stop reason.
    done_ev = next(e for e in events if e.kind == EventKind.DONE)
    assert done_ev.stop_reason == StopReason.TOOL_CALLS


# ---------------------------------------------------------------
# THE priority test: char-by-char feed must yield multiple
# TOOL_ARGS_DELTA events, not one at the end.
# ---------------------------------------------------------------

def test_streaming_body_emits_multiple_args_deltas():
    """The contract we're introducing.  Feeding the body char-by-char
    must produce *multiple* TOOL_ARGS_DELTA events -- one or more as
    args arrive -- not a single delta dumped at the </tool_call>
    boundary."""
    state = _make_state()
    text = (
        '<tool_call>{"name": "run_query", '
        '"arguments": {"table": "users", "limit": 100}}</tool_call>'
    )
    events = _feed(state, text, chunk=1)

    kinds = _kinds(events)
    args_delta_count = sum(1 for k in kinds if k == EventKind.TOOL_ARGS_DELTA)
    assert args_delta_count >= 3, (
        f"expected streaming to produce multiple args deltas as the "
        f"body arrives; got {args_delta_count}. "
        f"Events: {kinds}"
    )

    # Everything still reassembles correctly.
    combined = _collect_args_text(events)
    assert "table" in combined
    assert "users" in combined
    assert "100" in combined
    assert combined.strip().startswith("{")
    assert combined.strip().endswith("}")

    # The name must be reported exactly once, before the first args delta.
    name_indices = [i for i, e in enumerate(events) if e.kind == EventKind.TOOL_CALL_NAME]
    first_args_idx = next(
        i for i, e in enumerate(events) if e.kind == EventKind.TOOL_ARGS_DELTA
    )
    assert len(name_indices) == 1
    assert name_indices[0] < first_args_idx, (
        "TOOL_CALL_NAME must be emitted before the first TOOL_ARGS_DELTA "
        "when the model emits name-then-arguments"
    )


def test_streaming_body_does_not_leak_name_into_args():
    """Regression: the buffered implementation existed because naive
    streaming pushed raw tokens (which include the "name" key and
    value) into TOOL_ARGS_DELTA.  Verify the streaming extractor
    projects ONLY the arguments value -- no 'name' text, no
    structural JSON noise from keys."""
    state = _make_state()
    text = (
        '<tool_call>{"name": "banana_tree_scanner", '
        '"arguments": {"depth": 7}}</tool_call>'
    )
    events = _feed(state, text, chunk=1)

    combined = _collect_args_text(events)
    assert "name" not in combined, (
        f"args output leaks the JSON key 'name': {combined!r}"
    )
    assert "banana_tree_scanner" not in combined, (
        f"args output leaks the tool NAME value: {combined!r}"
    )
    # Only the arguments value content should be there.
    assert '"depth"' in combined
    assert "7" in combined


def test_streaming_args_first_then_name():
    """Model emits arguments before name (rare but legal).  Each field
    must still project to its right bucket."""
    state = _make_state()
    text = (
        '<tool_call>{"arguments": {"x": 1}, '
        '"name": "f"}</tool_call>'
    )
    events = _feed(state, text, chunk=1)

    name_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_NAME)
    assert name_ev.tool_name == "f"

    combined = _collect_args_text(events)
    assert '"x"' in combined
    assert "1" in combined
    assert "name" not in combined
    assert '"f"' not in combined


def test_streaming_string_arguments():
    """Args value is a plain JSON string, not an object."""
    state = _make_state()
    text = '<tool_call>{"name": "echo", "arguments": "hello world"}</tool_call>'
    events = _feed(state, text, chunk=1)

    name_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_NAME)
    assert name_ev.tool_name == "echo"

    combined = _collect_args_text(events)
    # Convention elsewhere in the codebase: string args pass through as-is.
    assert "hello world" in combined


def test_malformed_body_falls_back_to_buffered_parse():
    """Regression for the reason buffering existed in the first place:
    tiny real-world oddities (doubled braces, trailing commas, etc.)
    must still yield a tool_call -- we don't want streaming to make
    the parser LESS robust than it was."""
    state = _make_state()
    # Doubled opening + closing braces: a known Hermes-quirk tokenization
    # where '{{' and '}}' come out as single BPE tokens.
    text = '<tool_call>{{"name": "weird", "arguments": {"a": 1}}}</tool_call>'
    events = _feed(state, text, chunk=1)

    kinds = _kinds(events)
    assert EventKind.TOOL_CALL_COMMIT in kinds, (
        f"malformed body must still commit a tool call (even if name/args "
        f"extraction is best-effort). Events: {kinds}"
    )
    done_ev = next(e for e in events if e.kind == EventKind.DONE)
    assert done_ev.stop_reason == StopReason.TOOL_CALLS, (
        "finish_reason must be tool_calls even when the body can't be "
        "strictly parsed -- the model clearly intended a tool call"
    )


def test_doubled_brace_still_streams_name_and_args():
    """P0 bug: the Hermes doubled-brace quirk ('{{...}}' as a single
    BPE token at body open, mirrored at close) was silently killing
    streaming.  The JSON state machine errored on the second '{',
    no TOOL_CALL_NAME or TOOL_ARGS_DELTA events reached the client,
    and finish_reason=tool_calls arrived with an empty tool_calls
    entry -- the client then had nothing to execute.

    Fix: handle the doubled-brace quirk inline so streaming still
    projects name and args.
    """
    state = _make_state()
    text = '<tool_call>{{"name": "weather", "arguments": {"city": "Zagreb"}}}</tool_call>'
    events = _feed(state, text, chunk=1)

    # Name must be emitted at least once with the correct value.
    name_events = [e for e in events if e.kind == EventKind.TOOL_CALL_NAME]
    assert len(name_events) >= 1, (
        f"doubled-brace body: no TOOL_CALL_NAME events emitted; "
        f"client would have no tool name to execute. Events: {_kinds(events)}"
    )
    assert name_events[0].tool_name == "weather", (
        f"expected name 'weather', got {name_events[0].tool_name!r}"
    )

    # Args must be emitted with the city value.
    combined = _collect_args_text(events)
    assert "Zagreb" in combined, (
        f"doubled-brace body: args streaming dropped the payload. "
        f"combined={combined!r}"
    )
    assert '"city"' in combined


def test_unstreamable_body_emits_catchup_from_commit(monkeypatch):
    """Safety net: even if the streaming extractor fails entirely and
    emits nothing during the body, the COMMIT path must emit at least
    one TOOL_CALL_NAME and one TOOL_ARGS_DELTA so the client sees a
    callable tool_call.  Without this, finish_reason=tool_calls
    arrives with empty name/args and the client drops the call.

    We simulate total streaming failure by swapping the extractor
    with one that errors immediately and emits nothing.
    """
    import egg_toolbox.formats.hermes as hermes_mod

    class _AlwaysErrorExtractor:
        def __init__(self, name_field="name", args_field="arguments"):
            self._body: list[str] = []
            self.errored = True   # pretend we errored immediately
        def feed_chars(self, text):
            self._body.append(text)
        def drain_args(self):
            return ""
        def drain_name_if_ready(self):
            return None
        def full_body(self):
            return "".join(self._body)
        def name(self):
            return ""

    monkeypatch.setattr(
        hermes_mod, "_StreamingBodyExtractor", _AlwaysErrorExtractor,
    )
    state = _make_state()
    text = (
        '<tool_call>{"name": "fetch", "arguments": {"url": "https://x"}}'
        '</tool_call>'
    )
    events = _feed(state, text, chunk=3)

    name_events = [e for e in events if e.kind == EventKind.TOOL_CALL_NAME]
    args_events = [e for e in events if e.kind == EventKind.TOOL_ARGS_DELTA]
    assert len(name_events) >= 1, (
        f"catch-up failed: no TOOL_CALL_NAME after streaming errored. "
        f"Events: {_kinds(events)}"
    )
    assert name_events[0].tool_name == "fetch"
    assert len(args_events) >= 1, (
        f"catch-up failed: no TOOL_ARGS_DELTA after streaming errored. "
        f"Events: {_kinds(events)}"
    )
    combined = "".join(e.text for e in args_events if e.text)
    assert "https://x" in combined


def test_content_before_tool_call_still_streams():
    """The prefix content before <tool_call> must still arrive as
    CONTENT_DELTA events; this regression was NOT a streaming-args
    bug but a sanity check we don't break the content path."""
    state = _make_state()
    text = 'Thinking... <tool_call>{"name": "f", "arguments": {}}</tool_call>'
    events = _feed(state, text, chunk=1)

    content = _collect_content_text(events)
    assert "Thinking..." in content


def test_nested_object_arguments_stream_verbatim():
    """Nested objects in arguments must stream with braces preserved."""
    state = _make_state()
    text = (
        '<tool_call>{"name": "f", "arguments": '
        '{"filter": {"age": {"gte": 18, "lte": 65}}, "limit": 10}}'
        '</tool_call>'
    )
    events = _feed(state, text, chunk=1)

    combined = _collect_args_text(events)
    # Balanced braces preserved.
    assert combined.count("{") == combined.count("}")
    assert '"age"' in combined
    assert '"gte"' in combined
    assert "18" in combined
    assert "65" in combined


def test_tool_call_ids_are_unique_across_parser_instances():
    """P0 bug: tool_call_id collision across conversation turns.

    Every request builds a fresh parser state where ``_tool_index``
    starts at 0, so the first tool_call in *every* turn used to get
    ``id='call_0'``.  In a 2-turn conversation where the first turn
    called a tool and the second turn calls another, both assistant
    responses carry ``id='call_0'`` -- clients that dedupe by id
    (egg-mono is one) treat the second call as already-approved and
    silently skip the approval dialog and the execution.

    Fix: tool_call_id must be globally unique per call, not just
    per request.  A UUID-prefixed id guarantees that.
    """
    seen_ids: set[str] = set()
    for turn in range(3):
        state = _make_state()
        events = _feed(
            state,
            '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
            chunk=len('<tool_call>{"name": "f", "arguments": {}}</tool_call>'),
        )
        start_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_START)
        commit_ev = next(e for e in events if e.kind == EventKind.TOOL_CALL_COMMIT)

        # START and COMMIT for the same call must share an id.
        assert start_ev.tool_call_id == commit_ev.tool_call_id, (
            f"turn {turn}: START id {start_ev.tool_call_id!r} != "
            f"COMMIT id {commit_ev.tool_call_id!r}"
        )

        # Id must not repeat across requests.
        assert start_ev.tool_call_id not in seen_ids, (
            f"turn {turn}: tool_call_id {start_ev.tool_call_id!r} was "
            f"already emitted by a previous parser instance -- clients "
            f"will dedupe by id and drop the second call"
        )
        seen_ids.add(start_ev.tool_call_id)


def test_two_tool_calls_in_one_response_have_distinct_ids():
    """Same-response multi-tool_call: each entry must have its own id."""
    state = _make_state()
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    events = _feed(state, text, chunk=len(text))
    commits = [e for e in events if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 2
    assert commits[0].tool_call_id != commits[1].tool_call_id, (
        f"two distinct tool calls got the same id: "
        f"{commits[0].tool_call_id!r}"
    )


def test_string_arguments_with_escapes_stream_correctly():
    """Escaped quotes inside a string-valued arguments field."""
    state = _make_state()
    text = (
        '<tool_call>{"name": "say", '
        r'"arguments": "she said \"hi\""'
        '}</tool_call>'
    )
    events = _feed(state, text, chunk=1)

    combined = _collect_args_text(events)
    # The streamed content should contain the literal text 'she said "hi"'
    # (backslash-escape sequences decoded).
    assert 'she said "hi"' in combined or r'she said \"hi\"' in combined
