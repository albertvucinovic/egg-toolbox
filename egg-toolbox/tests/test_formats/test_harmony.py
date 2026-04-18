"""Unit tests for the Harmony multi-channel tool-call parser."""
from __future__ import annotations

import json

from egg_toolbox.formats.harmony import HarmonyParserState
from egg_toolbox.types import EventKind, FormatAnalysis, ReasoningMode, StopReason, ToolFormatMode


def _analysis() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.HARMONY,
        reasoning_mode=ReasoningMode.NONE,
    )


def _feed_all(state: HarmonyParserState, text: str):
    events = []
    for ch in text:
        events.extend(state.feed_token(-1, ch))
    events.extend(state.finish())
    return events


# ---------- Dialect 1: simplified channel markers ----------

def test_simple_all_three_channels():
    state = HarmonyParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<|analysis|>thinking<|commentary|>functions.get_weather({"city":"Paris"})<|final|>Here you go.'
    )
    reasoning = "".join(e.text for e in out if e.kind == EventKind.REASONING_DELTA)
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]

    assert reasoning == "thinking"
    assert content == "Here you go."
    assert len(commits) == 1
    assert commits[0].tool_name == "get_weather"
    assert json.loads(commits[0].tool_arguments) == {"city": "Paris"}

    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_simple_final_only():
    state = HarmonyParserState(_analysis(), tools=None)
    out = _feed_all(state, '<|final|>Hello.')
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "Hello."
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.STOP


def test_simple_parallel_tool_calls_in_commentary():
    state = HarmonyParserState(_analysis(), tools=None)
    out = _feed_all(state,
        '<|commentary|>a({"x":1})\nb({"y":2})<|final|>done.'
    )
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    assert [c.tool_name for c in commits] == ["a", "b"]


# ---------- Dialect 2: full gpt-oss protocol ----------

def test_gptoss_tool_call():
    state = HarmonyParserState(_analysis(), tools=None)
    text = (
        '<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>'
        '{"city":"Paris"}<|call|>'
    )
    out = _feed_all(state, text)
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]
    names = [e.tool_name for e in out if e.kind == EventKind.TOOL_CALL_NAME]
    assert len(commits) == 1
    assert names == ["get_weather"]
    args_deltas = [e.text for e in out if e.kind == EventKind.TOOL_ARGS_DELTA]
    assert args_deltas and json.loads(args_deltas[0]) == {"city": "Paris"}
    done = [e for e in out if e.kind == EventKind.DONE]
    assert done[0].stop_reason == StopReason.TOOL_CALLS


def test_gptoss_final_channel():
    state = HarmonyParserState(_analysis(), tools=None)
    text = '<|start|>assistant<|channel|>final<|message|>The answer.<|return|>'
    out = _feed_all(state, text)
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    assert content == "The answer."


def test_gptoss_analysis_then_tool_call_then_final():
    state = HarmonyParserState(_analysis(), tools=None)
    text = (
        '<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>'
        '<|start|>assistant<|channel|>commentary to=functions.f<|message|>{"x":1}<|call|>'
        '<|start|>assistant<|channel|>final<|message|>ok.<|return|>'
    )
    out = _feed_all(state, text)
    reasoning = "".join(e.text for e in out if e.kind == EventKind.REASONING_DELTA)
    content = "".join(e.text for e in out if e.kind == EventKind.CONTENT_DELTA)
    commits = [e for e in out if e.kind == EventKind.TOOL_CALL_COMMIT]

    assert reasoning == "thinking"
    assert content == "ok."
    assert len(commits) == 1
    names = [e.tool_name for e in out if e.kind == EventKind.TOOL_CALL_NAME]
    assert names == ["f"]


def test_plain_content_without_markers_is_swallowed():
    """Outside any channel, Harmony text is ignored (IDLE state).
    This matches gpt-oss behaviour -- user-visible text MUST be in
    a channel."""
    state = HarmonyParserState(_analysis(), tools=None)
    out = _feed_all(state, "stray text")
    # No REASONING, no CONTENT, no TOOL_CALLs -- just DONE.
    assert [e.kind for e in out] == [EventKind.DONE]
    assert out[0].stop_reason == StopReason.STOP
