"""Unit tests for the Llama 3 tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.llama3 import Llama3ParserState, _parse_llama3_body
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.LLAMA3,
        reasoning_mode=ReasoningMode.NONE,
        name_field="name",
        args_field="parameters",
    )


def _feed_all(state: Llama3ParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def test_json_tool_call():
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state,
        'Hello<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_python_call_syntax():
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state, '<|python_tag|>get_weather(city="Paris", units="c")<|eom_id|>')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris", "units": "c"}


def test_content_before_trigger_emitted():
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state, 'I will check.<|python_tag|>{"name":"f","parameters":{}}<|eom_id|>')
    content = "".join(
        e.text for e in out if e.kind == EventKind.CONTENT_DELTA
    )
    assert content == "I will check."


def test_eot_id_also_closes_tool_call():
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state, '<|python_tag|>{"name":"f","parameters":{"x":1}}<|eot_id|>')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"
    assert json.loads(commits[0].tool_arguments) == {"x": 1}


def test_plain_content_no_tool_call():
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state, "Hello there.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Hello there."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP


def test_stop_string_eaten_by_orchestrator():
    """Real backends strip <|eom_id|>/<|eot_id|> via StopStringMatcher
    before the parser sees it.  finish() must still commit the tool
    call from the buffered body."""
    state = Llama3ParserState(_analysis(), tools=None)
    out = _feed_all(state, '<|python_tag|>{"name":"f","parameters":{"x":1}}')
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"


def test_parse_body_json_fallback_via_regex():
    name, args = _parse_llama3_body(
        '{"name": "f", "parameters": {"x": 1},}',  # trailing comma
        "parameters",
    )
    assert name == "f"
    assert json.loads(args) == {"x": 1}


def test_parse_body_python_call_positional():
    name, args = _parse_llama3_body('f(1, 2, x="y")', "parameters")
    assert name == "f"
    parsed = json.loads(args)
    assert parsed["x"] == "y"
    assert parsed["_args"] == [1, 2]
