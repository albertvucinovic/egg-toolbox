"""Unit tests for the Generic tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.generic import GenericParserState, _parse_generic_body
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.GENERIC_JSON,
        reasoning_mode=ReasoningMode.NONE,
    )


def _feed_all(state: GenericParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def test_leading_json_object_treated_as_tool_call():
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state, '{"name": "get_weather", "arguments": {"city": "Paris"}}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}


def test_leading_json_array_multiple_calls():
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 2
    assert [c.tool_name for c in commits] == ["a", "b"]


def test_leading_prose_never_treated_as_tool_call():
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state, 'Here is a JSON blob: {"name": "fake"}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert commits == []
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == 'Here is a JSON blob: {"name": "fake"}'


def test_trailing_content_after_json_goes_to_content():
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '{"name":"f","arguments":{"x":1}} then some prose.'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content.strip() == "then some prose."


def test_openai_style_wrapped_function_object():
    """Generic parser accepts OpenAI-style {type:function, function:{...}}."""
    calls = _parse_generic_body(
        '{"type":"function","function":{"name":"f","arguments":{"x":1}}}',
        "arguments",
    )
    assert calls == [("f", '{"x": 1}')]


def test_openai_style_parameters_key():
    """Some tunes use 'parameters' instead of 'arguments'."""
    calls = _parse_generic_body(
        '{"name":"f","parameters":{"x":1}}',
        "arguments",
    )
    assert calls == [("f", '{"x": 1}')]


def test_plain_content_no_json():
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state, "Just prose.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Just prose."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP


def test_truncated_json_on_finish_regex_fallback():
    """Unclosed JSON at EOS: regex fallback still harvests the name."""
    state = GenericParserState(_analysis(), tools=None)
    out = _feed_all(state, '{"name":"f","arguments":{"x":1}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS
