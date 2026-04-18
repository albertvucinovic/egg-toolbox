from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# --- Tool Definitions (input) ---

@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str                                   # JSON Schema type
    description: str = ""
    required: bool = False
    enum: tuple[str, ...] | None = None
    properties: dict[str, ToolParameter] | None = None  # nested objects
    items: ToolParameter | None = None          # array items


@dataclass(frozen=True)
class ToolFunction:
    name: str
    description: str
    parameters: dict[str, ToolParameter] = field(default_factory=dict)
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    type: str                                   # always "function" for now
    function: ToolFunction

    def to_json_schema(self) -> dict[str, Any]:
        """Produce the JSON Schema object for this tool's parameters."""
        properties: dict[str, Any] = {}
        for name, param in self.function.parameters.items():
            properties[name] = _param_to_schema(param)
        return {
            "type": "object",
            "properties": properties,
            "required": list(self.function.required),
        }


def _param_to_schema(param: ToolParameter) -> dict[str, Any]:
    """Convert a ToolParameter to a JSON Schema dict."""
    schema: dict[str, Any] = {"type": param.type}
    if param.description:
        schema["description"] = param.description
    if param.enum is not None:
        schema["enum"] = list(param.enum)
    if param.properties is not None:
        schema["properties"] = {
            n: _param_to_schema(p) for n, p in param.properties.items()
        }
    if param.items is not None:
        schema["items"] = _param_to_schema(param.items)
    return schema


# --- Sampling Parameters ---

@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1                             # -1 = disabled
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None


# --- Chat Messages (input) ---

@dataclass(frozen=True)
class ContentPart:
    type: str                                   # "text" | "image_url"
    text: str | None = None
    image_url: dict[str, str] | None = None


@dataclass(frozen=True)
class ToolCallFunction:
    name: str
    arguments: str                              # raw JSON string


@dataclass(frozen=True)
class ToolCall:
    id: str
    type: str                                   # "function"
    function: ToolCallFunction


@dataclass(frozen=True)
class ChatMessage:
    role: str                                   # "system"|"user"|"assistant"|"tool"
    content: str | list[ContentPart] | None = None
    name: str | None = None                     # tool name (for role=tool)
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None             # for role=tool


# --- Semantic Events (internal, backend-independent) ---

class EventKind(enum.Enum):
    CONTENT_DELTA      = "content_delta"
    REASONING_DELTA    = "reasoning_delta"
    TOOL_CALL_START    = "tool_call_start"
    TOOL_CALL_NAME     = "tool_call_name"       # name finalized
    TOOL_ARGS_DELTA    = "tool_args_delta"
    TOOL_CALL_COMMIT   = "tool_call_commit"     # tool call fully parsed
    DONE               = "done"
    ERROR              = "error"


class StopReason(enum.Enum):
    STOP       = "stop"
    LENGTH     = "length"
    TOOL_CALLS = "tool_calls"
    ERROR      = "error"


@dataclass(frozen=True)
class SemanticEvent:
    kind: EventKind
    text: str | None = None                     # content_delta, reasoning_delta, tool_args_delta
    tool_index: int | None = None               # tool_call_start, tool_call_name, tool_args_delta, tool_call_commit
    tool_call_id: str | None = None             # tool_call_start
    tool_name: str | None = None                # tool_call_name, tool_call_commit
    tool_arguments: str | None = None           # tool_call_commit (complete JSON)
    stop_reason: StopReason | None = None       # done
    error_message: str | None = None            # error
    raw_token: str | None = None                # the literal decoded token (for debugging)
    token_id: int | None = None                 # the token ID (for debugging)
    prompt_tokens: int | None = None            # token count (DONE events only)
    completion_tokens: int | None = None        # token count (DONE events only)


# --- Compiled Request (middleware output to backend) ---

@dataclass(frozen=True)
class CompiledRequest:
    """Everything the backend needs to execute a generation request."""
    prompt_tokens: list[int]                    # tokenized, template-rendered prompt
    sampling: SamplingParams
    stop_strings: tuple[str, ...]               # extra stop strings from format handler
    stop_token_ids: tuple[int, ...]             # extra stop token IDs from format handler
    grammar: str | None = None                  # GBNF grammar string (optional)
    json_schema: dict[str, Any] | None = None   # for structured_outputs backends
    format_handler_name: str = ""               # which FormatHandler is active


# --- Format Analysis (output of detector) ---

class ToolFormatMode(enum.Enum):
    NONE            = "none"
    HERMES          = "hermes"           # <tool_call>JSON</tool_call>
    LLAMA3          = "llama3"           # <|python_tag|>func(...)
    MISTRAL         = "mistral"          # [TOOL_CALLS]
    DEEPSEEK        = "deepseek"         # fullwidth unicode markers
    FUNCTIONARY_V3  = "functionary_v3"   # <function=name>
    FUNCTIONARY_V3_1 = "functionary_v3_1" # >>> routing
    COMMAND_R       = "command_r"        # <|START_ACTION|>
    HARMONY         = "harmony"          # multi-channel <|channel|> format
    GENERIC_JSON    = "generic_json"     # fallback: bare JSON array/object


class ReasoningMode(enum.Enum):
    NONE       = "none"
    TAG_BASED  = "tag_based"             # <think>...</think>
    TOOLS_ONLY = "tools_only"


@dataclass(frozen=True)
class FormatAnalysis:
    """Result of analyzing a model's chat template for tool calling format."""
    tool_mode: ToolFormatMode
    reasoning_mode: ReasoningMode

    # Tool markers (extracted from template)
    tool_call_start: str = ""                   # e.g. "<tool_call>"
    tool_call_end: str = ""                     # e.g. "</tool_call>"
    section_start: str = ""                     # e.g. "<tool_call>" (section-level)
    section_end: str = ""

    # Reasoning markers
    reasoning_start: str = ""                   # e.g. "<think>"
    reasoning_end: str = ""                     # e.g. "</think>"

    # Content markers (if content is wrapped)
    content_start: str = ""
    content_end: str = ""

    # JSON structure fields (for JSON-based formats)
    name_field: str = "name"
    args_field: str = "arguments"

    # Stop conditions
    extra_stop_strings: tuple[str, ...] = ()
    extra_stop_token_ids: tuple[int, ...] = ()
