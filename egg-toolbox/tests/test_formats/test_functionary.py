"""Unit tests for the Functionary tool-call parser (v3 and v3.1)."""
from __future__ import annotations

import json

from egg_toolbox.formats.functionary import FunctionaryParserState
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis(mode: ToolFormatMode = ToolFormatMode.FUNCTIONARY_V3) -> FormatAnalysis:
    return FormatAnalysis(tool_mode=mode, reasoning_mode=ReasoningMode.NONE)


def _feed_all(state: FunctionaryParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def test_v3_single_tool_call():
    state = FunctionaryParserState(_analysis(), tools=None)
    out = _feed_all(state, '<function=get_weather>{"city":"Paris"}</function>')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_v3_content_before_call():
    state = FunctionaryParserState(_analysis(), tools=None)
    out = _feed_all(state,
        'Checking.<function=f>{"x":1}</function>'
    )
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Checking."


def test_v3_parallel_calls():
    state = FunctionaryParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<function=a>{"x":1}</function><function=b>{"y":2}</function>'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert [c.tool_name for c in commits] == ["a", "b"]
    assert json.loads(commits[0].tool_arguments) == {"x": 1}
    assert json.loads(commits[1].tool_arguments) == {"y": 2}


def test_v31_single_call():
    state = FunctionaryParserState(_analysis(ToolFormatMode.FUNCTIONARY_V3_1), tools=None)
    out = _feed_all(state, '>>>get_weather\n{"city":"Paris"}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}


def test_v31_parallel_calls():
    state = FunctionaryParserState(_analysis(ToolFormatMode.FUNCTIONARY_V3_1), tools=None)
    out = _feed_all(state,
        '>>>a\n{"x":1}>>>b\n{"y":2}'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert [c.tool_name for c in commits] == ["a", "b"]


def test_plain_content_no_call():
    state = FunctionaryParserState(_analysis(), tools=None)
    out = _feed_all(state, "Just prose.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Just prose."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP
