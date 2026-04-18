from __future__ import annotations

from ..types import FormatAnalysis, ToolFormatMode
from .base import FormatHandler
from .command_r import CommandRHandler
from .deepseek import DeepSeekHandler
from .functionary import FunctionaryHandler
from .generic import GenericHandler
from .harmony import HarmonyHandler
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
    if analysis.tool_mode == ToolFormatMode.COMMAND_R:
        return CommandRHandler(analysis)
    if analysis.tool_mode in (ToolFormatMode.FUNCTIONARY_V3, ToolFormatMode.FUNCTIONARY_V3_1):
        return FunctionaryHandler(analysis)
    if analysis.tool_mode == ToolFormatMode.HARMONY:
        return HarmonyHandler(analysis)
    if analysis.tool_mode == ToolFormatMode.GENERIC_JSON:
        return GenericHandler(analysis)
    # All Phase 2 format handlers implemented.
    # Fallback to Hermes for unknown tool-supporting modes.
    if analysis.tool_mode != ToolFormatMode.NONE:
        return HermesHandler(analysis)
    # No tool support detected -- Hermes acts as a passthrough.
    return HermesHandler(analysis)
