"""Format auto-detection via template analysis.

Phase 1: Stub implementation using keyword matching.
Phase 2 will implement full differential template analysis.
"""
from __future__ import annotations

from .types import FormatAnalysis, ToolFormatMode, ReasoningMode
from .template import ChatTemplate


def detect_format(template: ChatTemplate) -> FormatAnalysis:
    """Detect the tool calling format from a chat template.

    Phase 1 stub: keyword match on template source.
    Phase 2: full differential analysis (port of llama.cpp autoparser).
    """
    source = template.source

    # Detect reasoning mode
    reasoning_mode = ReasoningMode.NONE
    reasoning_start = ""
    reasoning_end = ""
    if "enable_thinking" in source or "reasoning_content" in source:
        reasoning_mode = ReasoningMode.TAG_BASED
        reasoning_start = "<think>"
        reasoning_end = "</think>"

    # Detect tool format via keyword matching
    if "<tool_call>" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.HERMES,
            reasoning_mode=reasoning_mode,
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
            name_field="name",
            args_field="arguments",
        )

    if "<|python_tag|>" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.LLAMA3,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if "[TOOL_CALLS]" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.MISTRAL,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if "\uff5c" in source or "fullwidth" in source.lower():
        return FormatAnalysis(
            tool_mode=ToolFormatMode.DEEPSEEK,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if "<function=" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.FUNCTIONARY_V3,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if ">>>" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.FUNCTIONARY_V3_1,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if "<|START_ACTION|>" in source:
        return FormatAnalysis(
            tool_mode=ToolFormatMode.COMMAND_R,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    if "<|channel|>" in source or ("<|analysis|>" in source and "<|commentary|>" in source):
        return FormatAnalysis(
            tool_mode=ToolFormatMode.HARMONY,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    # Fallback: if template references tools at all, assume generic JSON
    if template.supports_tools():
        return FormatAnalysis(
            tool_mode=ToolFormatMode.GENERIC_JSON,
            reasoning_mode=reasoning_mode,
            reasoning_start=reasoning_start,
            reasoning_end=reasoning_end,
        )

    return FormatAnalysis(
        tool_mode=ToolFormatMode.NONE,
        reasoning_mode=reasoning_mode,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
    )
