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
