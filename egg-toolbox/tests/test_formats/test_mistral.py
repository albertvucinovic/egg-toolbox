"""Unit tests for the Mistral tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.mistral import MistralParserState, _parse_tool_calls_array
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.MISTRAL,
        reasoning_mode=ReasoningMode.NONE,
        name_field="name",
        args_field="arguments",
    )


def _feed_all(state: MistralParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def test_single_tool_call_in_array():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_parallel_tool_calls():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '[TOOL_CALLS]['
        '{"name":"a","arguments":{"x":1}},'
        '{"name":"b","arguments":{"y":2}}'
        ']'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 2
    assert commits[0].tool_name == "a"
    assert commits[1].tool_name == "b"
    assert json.loads(commits[0].tool_arguments) == {"x": 1}
    assert json.loads(commits[1].tool_arguments) == {"y": 2}

    starts = [e for e in out if e.kind == EventKind.TOOL_CALL_START]
    assert [s.tool_index for s in starts] == [0, 1]


def test_content_before_trigger_emitted():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state,
        'thinking... [TOOL_CALLS][{"name":"f","arguments":{}}]'
    )
    content = "".join(
        e.text for e in out if e.kind == EventKind.CONTENT_DELTA
    )
    assert content == "thinking... "


def test_brackets_inside_string_values_do_not_confuse_scanner():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '[TOOL_CALLS][{"name":"search","arguments":{"q":"arr[0]=1]"}}]'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert json.loads(commits[0].tool_arguments) == {"q": "arr[0]=1]"}


def test_escaped_quote_in_string():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '[TOOL_CALLS][{"name":"f","arguments":{"q":"he said \\"hi\\""}}]'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert json.loads(commits[0].tool_arguments) == {"q": 'he said "hi"'}


def test_truncated_body_on_finish():
    """If EOS cuts off before the array closes, still emit tool_calls."""
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state, '[TOOL_CALLS][{"name":"f","arguments":{"x":1}}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_single_object_body_not_array():
    """Some Mistral fine-tunes emit a single object instead of an array."""
    name_args = _parse_tool_calls_array(
        '{"name":"f","arguments":{"x":1}}', "arguments"
    )
    assert name_args == [("f", '{"x": 1}')]


def test_plain_content_no_tool_call():
    state = MistralParserState(_analysis(), tools=None)
    out = _feed_all(state, "Just prose here.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Just prose here."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP
