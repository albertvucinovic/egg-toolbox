from __future__ import annotations

from ..types import FormatAnalysis, ToolFormatMode
from .base import FormatHandler
from .deepseek import DeepSeekHandler
from .hermes import HermesHandler
from .llama3 import Llama3Handler
from .mistral import MistralHandler


def get_handler_for_format(analysis: FormatAnalysis) -> FormatHandler:
    """Return the appropriate FormatHandler for the detected format."""
    if analysis.tool_mode == ToolFormatMode.HERMES:
        return HermesHandler(analysis)
    if analysis.tool_mode == ToolFormatMode.LLAMA3:
        return Llama3Handler(analysis)
    if analysis.tool_mode == ToolFormatMode.MISTRAL:
        return MistralHandler(analysis)
    if analysis.tool_mode == ToolFormatMode.DEEPSEEK:
        return DeepSeekHandler(analysis)
    # Phase 2 remaining: functionary, command_r, harmony, generic
    # Until those land, fall back to Hermes for any format that has tools.
    if analysis.tool_mode != ToolFormatMode.NONE:
        return HermesHandler(analysis)
    # No tool support detected -- Hermes acts as a passthrough.
    return HermesHandler(analysis)
