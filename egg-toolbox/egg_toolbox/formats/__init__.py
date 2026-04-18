from __future__ import annotations

from ..types import FormatAnalysis, ToolFormatMode
from .base import FormatHandler
from .hermes import HermesHandler
from .llama3 import Llama3Handler


def get_handler_for_format(analysis: FormatAnalysis) -> FormatHandler:
    """Return the appropriate FormatHandler for the detected format."""
    if analysis.tool_mode == ToolFormatMode.HERMES:
        return HermesHandler(analysis)
    if analysis.tool_mode == ToolFormatMode.LLAMA3:
        return Llama3Handler(analysis)
    # Phase 2 remaining: mistral, deepseek, functionary, command_r, harmony, generic
    # Until those land, fall back to Hermes for any format that has tools.
    if analysis.tool_mode != ToolFormatMode.NONE:
        return HermesHandler(analysis)
    # No tool support detected -- Hermes acts as a passthrough.
    return HermesHandler(analysis)
