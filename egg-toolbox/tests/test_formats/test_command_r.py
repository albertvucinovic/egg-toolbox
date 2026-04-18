"""Unit tests for the Command-R tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.command_r import CommandRParserState
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.COMMAND_R,
        reasoning_mode=ReasoningMode.NONE,
    )


def _feed_all(state: CommandRParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def test_single_tool_call():
    state = CommandRParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<|START_ACTION|>[{"tool_name":"get_weather","parameters":{"location":"Zagreb"}}]<|END_ACTION|>'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"location": "Zagreb"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_parallel_tool_calls():
    state = CommandRParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<|START_ACTION|>['
        '{"tool_name":"a","parameters":{"x":1}},'
        '{"tool_name":"b","parameters":{"y":2}}'
        ']<|END_ACTION|>'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert [c.tool_name for c in commits] == ["a", "b"]


def test_content_before_action():
    state = CommandRParserState(_analysis(), tools=None)
    out = _feed_all(state,
        'I will check.<|START_ACTION|>[{"tool_name":"f","parameters":{}}]<|END_ACTION|>'
    )
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "I will check."


def test_truncated_body_on_finish():
    state = CommandRParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<|START_ACTION|>[{"tool_name":"f","parameters":{"x":1}}'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"


def test_plain_content_no_tool_call():
    state = CommandRParserState(_analysis(), tools=None)
    out = _feed_all(state, "Plain prose.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Plain prose."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP
