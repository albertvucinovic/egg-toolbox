"""Unit tests for the DeepSeek tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.deepseek import (
    DeepSeekParserState,
    _parse_deepseek_inner,
    _INNER_BEGIN,
    _INNER_END,
    _OUTER_BEGIN,
    _OUTER_END,
    _SEP,
)
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.DEEPSEEK,
        reasoning_mode=ReasoningMode.NONE,
    )


def _feed_all(state: DeepSeekParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


def _wrap(inner: str) -> str:
    return f"{_OUTER_BEGIN}{_INNER_BEGIN}{inner}{_INNER_END}{_OUTER_END}"


def test_single_tool_call_with_json_fence():
    body = f'function{_SEP}get_weather\n```json\n{{"city": "Paris"}}\n```'
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, _wrap(body))

    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_parallel_tool_calls():
    body1 = f'function{_SEP}a\n```json\n{{"x":1}}\n```'
    body2 = f'function{_SEP}b\n```json\n{{"y":2}}\n```'
    wrapped = (
        f"{_OUTER_BEGIN}"
        f"{_INNER_BEGIN}{body1}{_INNER_END}"
        f"{_INNER_BEGIN}{body2}{_INNER_END}"
        f"{_OUTER_END}"
    )
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, wrapped)

    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 2
    assert commits[0].tool_name == "a"
    assert commits[1].tool_name == "b"
    assert json.loads(commits[0].tool_arguments) == {"x": 1}
    assert json.loads(commits[1].tool_arguments) == {"y": 2}


def test_content_before_outer_marker_emitted():
    body = f'function{_SEP}f\n```json\n{{"x":1}}\n```'
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, f"Checking... {_wrap(body)}")

    content = "".join(
        e.text for e in out if e.kind == EventKind.CONTENT_DELTA
    )
    assert content == "Checking... "


def test_bare_json_without_fence():
    """Some fine-tunes omit the ```json fence; fall back to raw."""
    body = f'function{_SEP}f\n{{"x":1}}'
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, _wrap(body))
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"
    assert json.loads(commits[0].tool_arguments) == {"x": 1}


def test_missing_function_keyword():
    """Some fine-tunes skip the literal 'function' prefix."""
    body = f'f\n```json\n{{"x":1}}\n```'
    name, args = _parse_deepseek_inner(body)
    assert name == "f"
    assert json.loads(args) == {"x": 1}


def test_truncated_body_on_finish():
    """EOS before <｜tool▁call▁end｜>; still emit the call."""
    partial = f'{_OUTER_BEGIN}{_INNER_BEGIN}function{_SEP}f\n```json\n{{"x":1}}\n```'
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, partial)
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert len(commits) == 1
    assert commits[0].tool_name == "f"


def test_plain_content_no_tool_call():
    state = DeepSeekParserState(_analysis(), tools=None)
    out = _feed_all(state, "No tools here.")
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "No tools here."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP
