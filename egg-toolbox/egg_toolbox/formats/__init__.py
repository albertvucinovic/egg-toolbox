from __future__ import annotations

from ..types import FormatAnalysis, ToolFormatMode
from .base import FormatHandler
from .hermes import HermesHandler


def get_handler_for_format(analysis: FormatAnalysis) -> FormatHandler:
    """Return the appropriate FormatHandler for the detected format."""
    if analysis.tool_mode == ToolFormatMode.HERMES:
        return HermesHandler(analysis)
    # Phase 2 will add: llama3, mistral, deepseek, functionary, command_r, harmony, generic
    # For now, fall back to Hermes for any format that has tools
    if analysis.tool_mode != ToolFormatMode.NONE:
        return HermesHandler(analysis)
    # No tool support detected -- still return Hermes as a passthrough
    return HermesHandler(analysis)
