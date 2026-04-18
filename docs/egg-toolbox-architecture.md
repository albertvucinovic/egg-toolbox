# Architecture: `egg-toolbox` -- Universal Tool Calling Middleware for Local LLMs

## 1. Project Identity

**Name**: `egg-toolbox`
**Repository**: Standalone (not part of egg-mono)
**Language**: Pure Python (no Rust, no C extensions)
**License**: MIT
**Python**: 3.11+

---

## 2. Module Architecture Diagram

```
                          API Consumers
                    (OpenAI / Anthropic clients)
                              |
                    +---------+---------+
                    |    API Layer      |
                    | (Starlette/ASGI)  |
                    |  openai_api.py    |
                    |  anthropic_api.py |
                    +---------+---------+
                              |
                     SemanticEvent stream
                              |
                    +---------+---------+
                    |   Orchestrator    |
                    | orchestrator.py   |
                    +---------+---------+
                         /          \
                        /            \
            +----------+--+    +-----+---------+
            | Template    |    | Stream Parser |
            | Engine      |    | (state machine|
            | template.py |    |  + format     |
            +------+------+    |  handlers)    |
                   |           | parser.py     |
            +------+------+   +-------+-------+
            | Format      |           |
            | Detector    |    +------+------+
            | detector.py |    | Grammar     |
            +---------+---+    | Generator   |
                      |        | grammar.py  |
                      |        +------+------+
                      |               |
            +---------+---------+-----+
            |  Format Handlers        |
            |  formats/               |
            |    hermes.py            |
            |    llama3.py            |
            |    mistral.py           |
            |    deepseek.py          |
            |    functionary.py       |
            |    command_r.py         |
            |    harmony.py           |
            |    generic.py           |
            +----------+--------------+
                       |
            +----------+-----------+
            |  Backend Abstraction |
            |  backends/           |
            |    base.py (ABC)     |
            |    tinygrad.py       |
            |    vllm.py           |
            |    sglang.py         |
            |    llamacpp.py       |
            +----------------------+
```

---

## 3. Directory Layout

```
egg_toolbox/
  pyproject.toml
  README.md
  egg_toolbox/
    __init__.py
    __main__.py              # CLI entry point (argparse)
    types.py                 # Core data types (frozen dataclasses)
    orchestrator.py          # Request lifecycle: template -> generate -> parse -> emit
    template.py              # Jinja2 chat template engine
    detector.py              # Format auto-detection (differential analysis)
    parser.py                # Streaming state-machine parser
    grammar.py               # JSON schema -> GBNF/Xgrammar constraint generation
    formats/
      __init__.py
      base.py                # FormatHandler ABC
      hermes.py              # Hermes/Qwen <tool_call> format
      llama3.py              # Llama 3.x <|python_tag|> format
      mistral.py             # Mistral [TOOL_CALLS] format
      deepseek.py            # DeepSeek fullwidth-unicode format
      functionary.py         # Functionary >>> / <function=> formats
      command_r.py           # Command-R <|START_ACTION|> format
      harmony.py             # Harmony multi-channel format
      generic.py             # Fallback: raw JSON object/array extraction
    backends/
      __init__.py
      base.py                # Backend ABC (StepBackend / ConstraintBackend)
      tinygrad.py            # tinygrad Transformer.generate() integration
      vllm.py                # vLLM AsyncLLM / structured_outputs integration
      sglang.py              # SGLang Engine integration
      llamacpp.py            # llama-cpp-python Llama class integration
    api/
      __init__.py
      openai.py              # /v1/chat/completions endpoint
      anthropic.py           # /v1/messages endpoint
      sse.py                 # SSE streaming utilities
      middleware.py          # ASGI middleware (CORS, auth, error handling)
  tests/
    __init__.py
    test_types.py
    test_template.py
    test_detector.py
    test_parser.py
    test_grammar.py
    test_formats/
      test_hermes.py
      test_llama3.py
      test_mistral.py
      ...
    test_backends/
      test_tinygrad.py
      ...
    test_api/
      test_openai.py
      test_anthropic.py
    fixtures/
      templates/             # Collected Jinja2 chat templates from real models
      outputs/               # Sample model outputs for parser tests
```

---

## 4. Core Types (`egg_toolbox/types.py`)

All types are frozen dataclasses or enums. No mutability in the data layer.

```python
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# --- Tool Definitions (input) ---

@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str                                   # JSON Schema type
    description: str = ""
    required: bool = False
    enum: tuple[str, ...] | None = None
    properties: dict[str, "ToolParameter"] | None = None  # nested objects
    items: "ToolParameter | None" = None        # array items

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
        ...


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
    # Fields used by different event kinds (only relevant ones are set):
    text: str | None = None                     # content_delta, reasoning_delta, tool_args_delta
    tool_index: int | None = None               # tool_call_start, tool_call_name, tool_args_delta, tool_call_commit
    tool_call_id: str | None = None             # tool_call_start
    tool_name: str | None = None                # tool_call_name, tool_call_commit
    tool_arguments: str | None = None           # tool_call_commit (complete JSON)
    stop_reason: StopReason | None = None       # done
    error_message: str | None = None            # error
    raw_token: str | None = None                # the literal decoded token (for debugging)
    token_id: int | None = None                 # the token ID (for debugging)


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
```

---

## 5. Chat Template Engine (`egg_toolbox/template.py`)

The template engine loads Jinja2 chat templates from three sources:
1. GGUF file metadata (`tokenizer.chat_template`)
2. HuggingFace `tokenizer_config.json`
3. Override file path

```python
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Callable

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .types import ChatMessage, Tool


class ChatTemplate:
    """Loads and renders a model's Jinja2 chat template."""

    def __init__(self, template_str: str, bos_token: str = "", eos_token: str = ""):
        self.source = template_str
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._env = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        # Register custom filters/functions that Jinja2 templates expect:
        self._env.globals.update({
            "raise_exception": self._raise_exception,
            "strftime_now": self._strftime_now,
        })
        self._template = self._env.from_string(template_str)

    @staticmethod
    def from_gguf(path: str | Path) -> "ChatTemplate":
        """Extract chat_template, BOS, and EOS from GGUF metadata."""
        ...

    @staticmethod
    def from_hf_config(path: str | Path) -> "ChatTemplate":
        """Load from a HuggingFace tokenizer_config.json."""
        ...

    def render(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        """Render messages + tools into the full prompt string.

        Converts our ChatMessage/Tool types into the dict format that
        Jinja2 templates expect (OpenAI-style message dicts).
        """
        msg_dicts = [self._msg_to_dict(m) for m in messages]
        tool_dicts = [self._tool_to_dict(t) for t in tools] if tools else None

        return self._template.render(
            messages=msg_dicts,
            tools=tool_dicts,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def render_variants(
        self,
        modifier: Callable[[dict[str, Any]], dict[str, Any]],
        **base_kwargs: Any,
    ) -> tuple[str, str]:
        """Render the template twice: once with base params, once with
        modifier(base_params). Used by the format detector for differential analysis."""
        base_output = self._template.render(**base_kwargs)
        modified_kwargs = modifier(dict(base_kwargs))
        modified_output = self._template.render(**modified_kwargs)
        return base_output, modified_output

    # -- Capability probing --

    def supports_tools(self) -> bool:
        """Heuristic: does the template source reference 'tools'?"""
        return "tools" in self.source

    def supports_reasoning(self) -> bool:
        """Does the template reference 'reasoning_content' or 'enable_thinking'?"""
        return "reasoning_content" in self.source or "enable_thinking" in self.source

    def supports_parallel_tools(self) -> bool:
        """Does the template iterate over tool_calls (implying parallel support)?"""
        return "for " in self.source and "tool_call" in self.source

    # -- Internal helpers --

    @staticmethod
    def _msg_to_dict(msg: ChatMessage) -> dict[str, Any]: ...
    @staticmethod
    def _tool_to_dict(tool: Tool) -> dict[str, Any]: ...
    @staticmethod
    def _raise_exception(msg: str) -> None: ...
    @staticmethod
    def _strftime_now(fmt: str) -> str: ...
```

### GGUF metadata extraction

The `from_gguf` method reads the GGUF header to extract three keys:

| GGUF key | Maps to |
|---|---|
| `tokenizer.chat_template` | `template_str` |
| `tokenizer.ggml.bos_token_id` | look up in token list for `bos_token` |
| `tokenizer.ggml.eos_token_id` | look up in token list for `eos_token` |

Implementation approach: Parse the GGUF key-value metadata section only (no need to load tensors). GGUF format is well-documented and the KV section is a flat list of typed entries. We read until we find the three keys we need.

---

## 6. Format Detector (`egg_toolbox/detector.py`)

This is the Python port of llama.cpp's autoparser. It uses differential template analysis.

### Algorithm

```
Given: a ChatTemplate object

Phase 1 -- Detect reasoning markers:
  R1: Render with reasoning_content="REASON_A" vs without
      -> Extract reasoning_start, reasoning_end markers from diff

Phase 2 -- Detect tool call format:
  T1: Render with 1 tool call (name="FFF_FIRST", args={"arg":"AA_FIRST"})
      vs render with content-only (no tools)
      -> Classify as HERMES/LLAMA3/MISTRAL/etc based on:
         - Presence of known markers (<tool_call>, <|python_tag|>, [TOOL_CALLS])
         - JSON structure analysis (is function name inside a JSON key?)

  T2: Render with 1 tool call vs 2 tool calls
      -> Extract per_call_start, per_call_end markers from diff
      -> Disambiguate section vs per-call markers

  T3: Render with name="FFF_FIRST" vs name="SSS_SECOND"
      -> Extract name_prefix, name_suffix markers from diff

  T4: Render with args={"arg1":"val1"} vs args={"arg1":"val1","arg2":"val2"}
      -> Extract argument separators (for non-JSON formats only)

  T5: Render with tool_call_id="call001" vs "call002"
      -> Extract ID prefix, suffix, and position

Phase 3 -- Classify:
  Map extracted markers to a ToolFormatMode enum value.
  If markers match a known format, return that.
  If JSON-like but unrecognized framing, return GENERIC_JSON.
  If no tools section detected, return NONE.
```

### Core diff function

```python
@dataclass(frozen=True)
class DiffSplit:
    """Result of comparing two template renders."""
    prefix: str            # common leading text
    suffix: str            # common trailing text
    left: str              # text unique to variant A
    right: str             # text unique to variant B

def calculate_diff_split(text_a: str, text_b: str) -> DiffSplit:
    """Find longest common prefix and suffix, preserving tag boundaries.

    Tag boundary preservation: if a split point falls inside a <...> or [...] token,
    adjust it outward to the nearest tag boundary. This prevents splitting
    XML tags or control tokens in half.
    """
    # 1. Find raw common prefix length
    prefix_len = 0
    for i in range(min(len(text_a), len(text_b))):
        if text_a[i] != text_b[i]:
            break
        prefix_len = i + 1

    # 2. Find raw common suffix length (from remaining text after prefix)
    suffix_len = 0
    a_remaining = text_a[prefix_len:]
    b_remaining = text_b[prefix_len:]
    for i in range(1, min(len(a_remaining), len(b_remaining)) + 1):
        if a_remaining[-i] != b_remaining[-i]:
            break
        suffix_len = i

    # 3. Adjust for tag boundaries
    prefix_len = _adjust_prefix_for_tags(text_a, prefix_len)
    suffix_len = _adjust_suffix_for_tags(text_a, len(text_a) - suffix_len if suffix_len else len(text_a))

    prefix = text_a[:prefix_len]
    suffix = text_a[len(text_a) - suffix_len:] if suffix_len else ""
    left = text_a[prefix_len:len(text_a) - suffix_len if suffix_len else len(text_a)]
    right = text_b[prefix_len:len(text_b) - suffix_len if suffix_len else len(text_b)]

    return DiffSplit(prefix=prefix, suffix=suffix, left=left, right=right)
```

### Format classification heuristics

```python
def classify_tool_format(diff: DiffSplit, template: ChatTemplate) -> ToolFormatMode:
    """Classify the tool calling format based on extracted markers."""
    left = diff.left   # the tool-call content in the rendered template

    # Check for known marker patterns
    if "<tool_call>" in left:
        return ToolFormatMode.HERMES
    if "<|python_tag|>" in left:
        return ToolFormatMode.LLAMA3
    if "[TOOL_CALLS]" in left:
        return ToolFormatMode.MISTRAL
    if "\uff5c" in left or "\uff1c" in left:  # fullwidth chars
        return ToolFormatMode.DEEPSEEK
    if "<function=" in left:
        return ToolFormatMode.FUNCTIONARY_V3
    if ">>>" in left and "all\n" in left:
        return ToolFormatMode.FUNCTIONARY_V3_1
    if "<|START_ACTION|>" in left:
        return ToolFormatMode.COMMAND_R
    if "<|channel|>" in left or "analysis" in left and "commentary" in left:
        return ToolFormatMode.HARMONY

    # Fallback: if template supports tools and output contains JSON
    if template.supports_tools() and ("{" in left or "[" in left):
        return ToolFormatMode.GENERIC_JSON

    return ToolFormatMode.NONE
```

### Public API

```python
def detect_format(template: ChatTemplate) -> FormatAnalysis:
    """Run full differential analysis on a chat template.

    Returns a FormatAnalysis with all extracted markers,
    reasoning mode, tool format mode, and stop conditions.
    """
    ...
```

---

## 7. Format Handlers (`egg_toolbox/formats/base.py`)

Each format handler knows how to:
1. Provide stop strings and stop token IDs for its format
2. Parse a token stream to extract tool calls
3. Generate grammar constraints (optional)

```python
from __future__ import annotations

import abc
from typing import Any

from ..types import (
    FormatAnalysis, SemanticEvent, Tool,
)


class FormatHandler(abc.ABC):
    """Abstract base for tool-call format handlers.

    A FormatHandler is constructed once per model load from the
    FormatAnalysis produced by the detector. It is then used for
    every request to that model.
    """

    def __init__(self, analysis: FormatAnalysis):
        self.analysis = analysis

    @abc.abstractmethod
    def stop_strings(self) -> tuple[str, ...]:
        """Additional stop strings the backend should watch for."""
        ...

    @abc.abstractmethod
    def stop_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        """Additional stop token IDs (resolved from strings via tokenizer)."""
        ...

    @abc.abstractmethod
    def create_parser_state(self, tools: list[Tool] | None = None) -> "FormatParserState":
        """Create a fresh, per-request parser state machine."""
        ...

    def generate_grammar(self, tools: list[Tool]) -> str | None:
        """Generate a GBNF grammar constraining output to valid tool calls.

        Returns None if this format does not support grammar constraints.
        Default implementation returns None.
        """
        return None

    def generate_json_schema(self, tools: list[Tool]) -> dict[str, Any] | None:
        """Generate a JSON schema for structured_outputs backends.

        Returns None if this format does not support structured outputs.
        Default implementation returns None.
        """
        return None


class FormatParserState(abc.ABC):
    """Per-request mutable state for streaming tool call extraction.

    The parser state machine processes tokens one at a time (for step
    backends) or chunks of text (for constraint backends). It emits
    SemanticEvents describing what it found.

    The state machine lifecycle:
    1. Created fresh per request via FormatHandler.create_parser_state()
    2. fed tokens/text via feed_token() or feed_text()
    3. finalized via finish()
    4. Discarded after the request completes
    """

    @abc.abstractmethod
    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        """Process a single decoded token. Returns zero or more events."""
        ...

    @abc.abstractmethod
    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        """Process a chunk of text (for constraint backends that deliver
        text in larger chunks). Default implementation splits into
        characters and calls feed_token with token_id=-1.

        Subclasses SHOULD override for more efficient regex-based parsing
        when processing large text chunks from constraint backends.
        """
        events = []
        for ch in new_text:
            events.extend(self.feed_token(-1, ch))
        return events

    @abc.abstractmethod
    def finish(self) -> list[SemanticEvent]:
        """Signal end of generation. Flush any pending tool calls
        and emit TOOL_CALL_COMMIT / DONE events."""
        ...

    @abc.abstractmethod
    def has_pending_tool_call(self) -> bool:
        """Is a tool call currently being assembled?"""
        ...
```

### Concrete Format Handler: Hermes (`egg_toolbox/formats/hermes.py`)

The Hermes format is the most common (used by Qwen, Hermes, many fine-tunes).

Pattern: `<tool_call>{"name":"func","arguments":{"k":"v"}}</tool_call>`

```python
import enum
import json
import re
from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


class HermesHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        return (self.analysis.tool_call_end,)   # "</tool_call>"

    def stop_token_ids(self, tokenizer) -> tuple[int, ...]:
        # Resolve </tool_call> to token ID if it's a single token
        ...

    def create_parser_state(self, tools=None):
        return HermesParserState(self.analysis, tools)

    def generate_grammar(self, tools):
        return _hermes_gbnf(tools, self.analysis)


class _HermesState(enum.Enum):
    CONTENT         = "content"          # emitting content_delta
    REASONING       = "reasoning"        # inside <think>...</think>
    MAYBE_TAG       = "maybe_tag"        # accumulating '<' prefix
    IN_TOOL_TAG     = "in_tool_tag"      # inside <tool_call>, accumulating JSON
    AFTER_TOOL_CALL = "after_tool_call"  # between </tool_call> and next <tool_call> or EOS


class HermesParserState(FormatParserState):
    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _HermesState.CONTENT
        self._buffer = ""              # accumulation buffer for partial matches
        self._json_buffer = ""         # accumulating JSON inside <tool_call>
        self._tool_index = 0           # current tool call index (0-based)
        self._committed_tools: list[dict] = []
        self._tag_start = analysis.tool_call_start or "<tool_call>"
        self._tag_end = analysis.tool_call_end or "</tool_call>"
        self._reason_start = analysis.reasoning_start
        self._reason_end = analysis.reasoning_end

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text

        while self._buffer:
            if self._state == _HermesState.CONTENT:
                events.extend(self._process_content())
            elif self._state == _HermesState.REASONING:
                events.extend(self._process_reasoning())
            elif self._state == _HermesState.MAYBE_TAG:
                events.extend(self._process_maybe_tag())
            elif self._state == _HermesState.IN_TOOL_TAG:
                events.extend(self._process_in_tool())
            elif self._state == _HermesState.AFTER_TOOL_CALL:
                events.extend(self._process_after_tool())
            else:
                break
        return events

    def _process_content(self) -> list[SemanticEvent]:
        """In CONTENT state: emit content deltas, watch for <tool_call> or <think>."""
        events = []
        # Check for reasoning start
        if self._reason_start and self._buffer.startswith(self._reason_start):
            self._buffer = self._buffer[len(self._reason_start):]
            self._state = _HermesState.REASONING
            return events

        # Check for tool_call start tag
        if self._tag_start and self._tag_start in self._buffer:
            # Emit everything before the tag as content
            idx = self._buffer.index(self._tag_start)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._tag_start):]
            self._json_buffer = ""
            tool_call_id = f"call_{self._tool_index}"
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index,
                tool_call_id=tool_call_id,
            ))
            self._state = _HermesState.IN_TOOL_TAG
            return events

        # Check if buffer might be start of a tag (partial match)
        for i in range(1, len(self._tag_start)):
            if self._buffer.endswith(self._tag_start[:i]):
                # Hold back the potential tag prefix
                safe = self._buffer[:-i]
                if safe:
                    events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=safe))
                self._buffer = self._buffer[-i:]
                return events

        # No tag match possible -- emit entire buffer
        events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
        self._buffer = ""
        return events

    def _process_in_tool(self) -> list[SemanticEvent]:
        """In IN_TOOL_TAG state: accumulate JSON, emit args deltas, watch for </tool_call>."""
        events = []
        if self._tag_end in self._buffer:
            idx = self._buffer.index(self._tag_end)
            json_chunk = self._buffer[:idx]
            self._json_buffer += json_chunk
            self._buffer = self._buffer[idx + len(self._tag_end):]

            # Parse the complete JSON
            try:
                parsed = json.loads(self._json_buffer)
                name = parsed.get("name", "")
                args = json.dumps(parsed.get("arguments", {}))
            except json.JSONDecodeError:
                name = ""
                args = self._json_buffer

            events.append(SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name))
            events.append(SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=args))
            events.append(SemanticEvent(
                kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index,
                tool_call_id=f"call_{self._tool_index}",
                tool_name=name,
                tool_arguments=args,
            ))
            self._tool_index += 1
            self._state = _HermesState.AFTER_TOOL_CALL
        else:
            # Stream args deltas as they arrive
            self._json_buffer += self._buffer
            events.append(SemanticEvent(kind=EventKind.TOOL_ARGS_DELTA, tool_index=self._tool_index, text=self._buffer))
            self._buffer = ""
        return events

    def _process_reasoning(self) -> list[SemanticEvent]:
        """In REASONING state: emit reasoning deltas, watch for </think>."""
        events = []
        if self._reason_end and self._reason_end in self._buffer:
            idx = self._buffer.index(self._reason_end)
            if idx > 0:
                events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer[:idx]))
            self._buffer = self._buffer[idx + len(self._reason_end):]
            self._state = _HermesState.CONTENT
        else:
            events.append(SemanticEvent(kind=EventKind.REASONING_DELTA, text=self._buffer))
            self._buffer = ""
        return events

    def _process_maybe_tag(self) -> list[SemanticEvent]: ...
    def _process_after_tool(self) -> list[SemanticEvent]: ...

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def finish(self) -> list[SemanticEvent]:
        events = []
        # Flush any remaining buffer
        if self._buffer:
            if self._state == _HermesState.IN_TOOL_TAG:
                self._json_buffer += self._buffer
                self._buffer = ""
                # Attempt to commit incomplete tool call
                try:
                    parsed = json.loads(self._json_buffer)
                    name = parsed.get("name", "")
                    args = json.dumps(parsed.get("arguments", {}))
                    events.append(SemanticEvent(kind=EventKind.TOOL_CALL_NAME, tool_index=self._tool_index, tool_name=name))
                    events.append(SemanticEvent(kind=EventKind.TOOL_CALL_COMMIT,
                        tool_index=self._tool_index, tool_call_id=f"call_{self._tool_index}",
                        tool_name=name, tool_arguments=args))
                except json.JSONDecodeError:
                    pass
            elif self._state == _HermesState.CONTENT:
                events.append(SemanticEvent(kind=EventKind.CONTENT_DELTA, text=self._buffer))
                self._buffer = ""

        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _HermesState.IN_TOOL_TAG
```

### Other format handlers follow the same pattern

Each implements:
- Its own state machine enum
- Its own `FormatParserState` subclass with `feed_token`, `finish`
- Its own `FormatHandler` subclass providing stop strings and grammar

| Handler | Trigger detection | JSON extraction method | Special concerns |
|---|---|---|---|
| `HermesHandler` | `<tool_call>` | JSON.parse between tags | Most common; works for Qwen, Hermes |
| `Llama3Handler` | `<\|python_tag\|>` | Python function-call syntax parse | Must parse `func_name(arg=val)` syntax |
| `MistralHandler` | `[TOOL_CALLS]` | JSON array after marker | Control tokens may be single tokens |
| `DeepSeekHandler` | Fullwidth `\uff5c` chars | JSON between fullwidth markers | Unicode-aware scanning |
| `FunctionaryHandler` | `<function=` or `>>>` | JSON after `<function=name>` tag | Two sub-variants (v3, v3.1) |
| `CommandRHandler` | `<\|START_ACTION\|>` | JSON between action tags | Agent-oriented with plan/action |
| `HarmonyHandler` | `<\|channel\|>` markers | Multi-channel state machine | 3 channels: analysis→reasoning, commentary→tool calls, final→content. TypeScript namespace tool defs |
| `GenericHandler` | Raw `{` or `[` | Incremental JSON bracket matching | Fallback; no structural markers |

### Concrete Format Handler: Harmony (`egg_toolbox/formats/harmony.py`)

Harmony uses a multi-channel architecture where the model writes to 3 named channels:
- **analysis**: Internal reasoning (maps to `REASONING_DELTA`)
- **commentary**: Tool calls as TypeScript function invocations (maps to `TOOL_CALL_*` events)
- **final**: User-facing content (maps to `CONTENT_DELTA`)

Channels are delimited by markers like `<|analysis|>`, `<|commentary|>`, `<|final|>`.

```python
import enum
import json
import re
from ..types import SemanticEvent, EventKind, StopReason, Tool, FormatAnalysis
from .base import FormatHandler, FormatParserState


class HarmonyHandler(FormatHandler):
    def stop_strings(self) -> tuple[str, ...]:
        return ()  # Harmony uses EOS, not stop strings

    def stop_token_ids(self, tokenizer) -> tuple[int, ...]:
        return ()

    def create_parser_state(self, tools=None):
        return HarmonyParserState(self.analysis, tools)

    def generate_grammar(self, tools):
        return None  # Harmony uses its own constrained format


class _HarmonyState(enum.Enum):
    IDLE            = "idle"            # between channels or at start
    ANALYSIS        = "analysis"        # inside <|analysis|> channel → REASONING_DELTA
    COMMENTARY      = "commentary"      # inside <|commentary|> channel → TOOL_CALL_*
    FINAL           = "final"           # inside <|final|> channel → CONTENT_DELTA


class HarmonyParserState(FormatParserState):
    """Multi-channel state machine for Harmony format.

    Channel → SemanticEvent mapping:
      analysis   → REASONING_DELTA
      commentary → TOOL_CALL_START / TOOL_CALL_NAME / TOOL_ARGS_DELTA / TOOL_CALL_COMMIT
      final      → CONTENT_DELTA
    """

    # Channel markers
    _CHANNEL_RE = re.compile(r'<\|(analysis|commentary|final)\|>')

    def __init__(self, analysis: FormatAnalysis, tools: list[Tool] | None):
        self._analysis = analysis
        self._tools = tools
        self._state = _HarmonyState.IDLE
        self._buffer = ""
        self._tool_index = 0
        self._commentary_buffer = ""  # accumulates commentary for tool call extraction

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        self._buffer += token_text

        while self._buffer:
            match = self._CHANNEL_RE.search(self._buffer)

            if match:
                # Emit everything before the channel marker
                before = self._buffer[:match.start()]
                if before:
                    events.extend(self._emit_in_current_state(before))

                # Transition to new channel
                channel = match.group(1)
                events.extend(self._transition_to(channel))
                self._buffer = self._buffer[match.end():]
            else:
                # Check for partial channel marker at end of buffer
                partial = False
                for i in range(1, min(len(self._buffer), 15)):
                    if "<|".startswith(self._buffer[-i:]) or \
                       any(f"<|{ch}|>".startswith(self._buffer[-i:])
                           for ch in ("analysis", "commentary", "final")):
                        # Hold back potential partial marker
                        safe = self._buffer[:-i]
                        if safe:
                            events.extend(self._emit_in_current_state(safe))
                        self._buffer = self._buffer[-i:]
                        partial = True
                        break

                if not partial:
                    events.extend(self._emit_in_current_state(self._buffer))
                    self._buffer = ""

        return events

    def _emit_in_current_state(self, text: str) -> list[SemanticEvent]:
        """Emit events based on which channel we're currently in."""
        if self._state == _HarmonyState.ANALYSIS:
            return [SemanticEvent(kind=EventKind.REASONING_DELTA, text=text)]
        elif self._state == _HarmonyState.FINAL:
            return [SemanticEvent(kind=EventKind.CONTENT_DELTA, text=text)]
        elif self._state == _HarmonyState.COMMENTARY:
            self._commentary_buffer += text
            return self._try_extract_tool_calls()
        else:  # IDLE
            return []  # discard text outside channels

    def _transition_to(self, channel: str) -> list[SemanticEvent]:
        """Handle channel transition, flushing any pending state."""
        events = []
        # Flush commentary buffer on channel exit
        if self._state == _HarmonyState.COMMENTARY and self._commentary_buffer:
            events.extend(self._flush_commentary())

        self._state = {
            "analysis": _HarmonyState.ANALYSIS,
            "commentary": _HarmonyState.COMMENTARY,
            "final": _HarmonyState.FINAL,
        }[channel]
        return events

    def _try_extract_tool_calls(self) -> list[SemanticEvent]:
        """Extract tool calls from commentary buffer.

        Harmony tool calls in commentary look like TypeScript function invocations:
          namespace.functionName({ arg1: "value1", arg2: 42 })
        """
        # Stream args deltas for now; full extraction happens at flush
        return [SemanticEvent(
            kind=EventKind.TOOL_ARGS_DELTA,
            tool_index=self._tool_index,
            text=self._commentary_buffer[-1:] if self._commentary_buffer else "",
        )]

    def _flush_commentary(self) -> list[SemanticEvent]:
        """Parse accumulated commentary into tool call events."""
        events = []
        # Parse TypeScript-style function calls: funcName({ ... })
        call_re = re.compile(r'(\w+(?:\.\w+)*)\s*\(\s*(\{.*?\})\s*\)', re.DOTALL)
        for match in call_re.finditer(self._commentary_buffer):
            name = match.group(1).split(".")[-1]  # strip namespace prefix
            args_str = match.group(2)
            call_id = f"call_{self._tool_index}"

            events.append(SemanticEvent(kind=EventKind.TOOL_CALL_START,
                tool_index=self._tool_index, tool_call_id=call_id))
            events.append(SemanticEvent(kind=EventKind.TOOL_CALL_NAME,
                tool_index=self._tool_index, tool_name=name))
            events.append(SemanticEvent(kind=EventKind.TOOL_CALL_COMMIT,
                tool_index=self._tool_index, tool_call_id=call_id,
                tool_name=name, tool_arguments=args_str))
            self._tool_index += 1

        self._commentary_buffer = ""
        return events

    def feed_text(self, new_text: str) -> list[SemanticEvent]:
        return self.feed_token(-1, new_text)

    def finish(self) -> list[SemanticEvent]:
        events = []
        if self._buffer:
            events.extend(self._emit_in_current_state(self._buffer))
            self._buffer = ""
        if self._state == _HarmonyState.COMMENTARY and self._commentary_buffer:
            events.extend(self._flush_commentary())
        stop = StopReason.TOOL_CALLS if self._tool_index > 0 else StopReason.STOP
        events.append(SemanticEvent(kind=EventKind.DONE, stop_reason=stop))
        return events

    def has_pending_tool_call(self) -> bool:
        return self._state == _HarmonyState.COMMENTARY
```

#### Harmony Tool Definition Conversion

Harmony models expect tool definitions as TypeScript namespace declarations rather than JSON Schema. The handler includes a converter:

```python
def tools_to_typescript_namespace(tools: list[Tool], namespace: str = "functions") -> str:
    """Convert Tool definitions to TypeScript namespace format for Harmony templates.

    Input (JSON Schema style):
      Tool(function=ToolFunction(name="get_weather",
           parameters={"location": ToolParameter(type="string", required=True)}))

    Output (TypeScript namespace):
      namespace functions {
        // Get the current weather
        function get_weather(params: { location: string }): any;
      }
    """
    lines = [f"namespace {namespace} {{"]
    for tool in tools:
        fn = tool.function
        lines.append(f"  // {fn.description}")
        params = ", ".join(
            f"{name}: {_ts_type(param.type)}"
            for name, param in fn.parameters.items()
        )
        lines.append(f"  function {fn.name}(params: {{ {params} }}): any;")
    lines.append("}")
    return "\n".join(lines)

def _ts_type(json_type: str) -> str:
    """Map JSON Schema types to TypeScript types."""
    return {"string": "string", "integer": "number", "number": "number",
            "boolean": "boolean", "array": "any[]", "object": "Record<string, any>"
           }.get(json_type, "any")
```

---

## 8. Streaming Parser (`egg_toolbox/parser.py`)

The parser wraps a `FormatParserState` and adds:
1. Reasoning prefix handling (generation prompt stripping)
2. The "lazy grammar" pattern: grammar constraints only activate after a trigger word is detected
3. Streaming diff computation for API consumers

```python
from __future__ import annotations

from .types import SemanticEvent, EventKind, Tool, FormatAnalysis
from .formats.base import FormatHandler, FormatParserState


class StreamingParser:
    """Top-level parser that wraps a FormatHandler's parser state.

    Handles:
    1. Generation prompt stripping (the template's generation_prompt
       prefix is not part of the model's output)
    2. Lazy grammar activation (for step backends)
    3. Collecting all events into a final message structure
    """

    def __init__(
        self,
        handler: FormatHandler,
        tools: list[Tool] | None = None,
        generation_prompt_suffix: str = "",
    ):
        self._handler = handler
        self._state: FormatParserState = handler.create_parser_state(tools)
        self._gen_prompt_suffix = generation_prompt_suffix
        self._gen_prompt_consumed = False
        self._pending_text = ""

        # Accumulated results (for non-streaming consumers)
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls: list[dict] = []  # list of {id, name, arguments}

    def feed_token(self, token_id: int, token_text: str) -> list[SemanticEvent]:
        """Feed a single token into the parser. Returns events to emit."""
        # Strip generation prompt prefix from first tokens
        if not self._gen_prompt_consumed:
            self._pending_text += token_text
            if len(self._pending_text) >= len(self._gen_prompt_suffix):
                remainder = self._pending_text[len(self._gen_prompt_suffix):]
                self._gen_prompt_consumed = True
                if remainder:
                    return self._state.feed_token(token_id, remainder)
                return []
            return []

        events = self._state.feed_token(token_id, token_text)
        self._accumulate(events)
        return events

    def feed_text(self, text: str) -> list[SemanticEvent]:
        """Feed a chunk of text (for constraint backends)."""
        events = self._state.feed_text(text)
        self._accumulate(events)
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._state.finish()
        self._accumulate(events)
        return events

    def _accumulate(self, events: list[SemanticEvent]) -> None:
        for ev in events:
            if ev.kind == EventKind.CONTENT_DELTA and ev.text:
                self._content_parts.append(ev.text)
            elif ev.kind == EventKind.REASONING_DELTA and ev.text:
                self._reasoning_parts.append(ev.text)
            elif ev.kind == EventKind.TOOL_CALL_COMMIT:
                self._tool_calls.append({
                    "id": ev.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": ev.tool_name,
                        "arguments": ev.tool_arguments,
                    },
                })

    # -- Accessors for final message construction --

    @property
    def content(self) -> str:
        return "".join(self._content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def tool_calls(self) -> list[dict]:
        return list(self._tool_calls)

    def trigger_detected(self) -> bool:
        """Has the parser detected a tool-call trigger word?
        Used by step backends to decide whether to activate grammar constraints."""
        return self._state.has_pending_tool_call()
```

---

## 9. Backend Abstraction (`egg_toolbox/backends/base.py`)

Two abstract base classes reflecting the two backend modes.

```python
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Iterator

from ..types import CompiledRequest, SamplingParams


class Tokenizer(abc.ABC):
    """Minimal tokenizer interface that backends must expose."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str: ...

    @abc.abstractmethod
    def decode_single(self, token_id: int) -> str: ...

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> int: ...

    @property
    @abc.abstractmethod
    def bos_token_id(self) -> int | None: ...

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int: ...


class StepBackend(abc.ABC):
    """Backend where we control the generation loop token-by-token.

    Suitable for: tinygrad, llama-cpp-python.
    """

    @abc.abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer: ...

    @abc.abstractmethod
    def chat_template_source(self) -> str:
        """Return the raw Jinja2 chat template string from model metadata."""
        ...

    @abc.abstractmethod
    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        """Yield token IDs one at a time.

        The caller (orchestrator) is responsible for:
        - Decoding each token
        - Feeding it to the parser
        - Checking stop conditions
        - Breaking the loop when done

        The backend handles:
        - KV cache management
        - Sampling with the given parameters
        """
        ...

    @abc.abstractmethod
    def model_name(self) -> str: ...


class ConstraintBackend(abc.ABC):
    """Backend that owns the generation loop but accepts constraints.

    Suitable for: vLLM, SGLang.
    """

    @abc.abstractmethod
    async def load_model(self, model_path: str, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer: ...

    @abc.abstractmethod
    def chat_template_source(self) -> str: ...

    @abc.abstractmethod
    async def generate_stream(self, request: CompiledRequest) -> AsyncIterator[str]:
        """Yield text chunks (not individual tokens).

        The backend applies constraints (grammar, structured_outputs, stop
        strings) internally. The caller receives decoded text chunks and
        feeds them to the parser.
        """
        ...

    @abc.abstractmethod
    def model_name(self) -> str: ...


# Union type for any backend
Backend = StepBackend | ConstraintBackend
```

### tinygrad backend (`egg_toolbox/backends/tinygrad.py`)

```python
from __future__ import annotations

from typing import Any, Iterator

from .base import StepBackend, Tokenizer
from ..types import CompiledRequest


class TinygradTokenizer(Tokenizer):
    """Wraps tinygrad's SimpleTokenizer to our interface."""

    def __init__(self, inner):
        # inner: tinygrad.apps.llm.SimpleTokenizer
        self._inner = inner
        self._eos_id: int = ...        # from GGUF metadata
        self._bos_id: int | None = ... # from GGUF metadata

    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._inner.decode(token_ids)

    def decode_single(self, token_id: int) -> str:
        return self._inner.decode([token_id])

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    @property
    def vocab_size(self) -> int:
        return len(self._inner._normal_tokens) + len(self._inner._special_tokens)


class TinygradBackend(StepBackend):
    """Integration with tinygrad's Transformer.generate().

    This wraps tinygrad's existing token generation loop. The key integration
    point is Transformer.generate() which yields token IDs one at a time.
    """

    def __init__(self):
        self._model = None
        self._tokenizer: TinygradTokenizer | None = None
        self._model_name: str = ""
        self._chat_template: str = ""

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        # Import tinygrad at load time to keep it optional
        from tinygrad.apps.llm import Transformer, SimpleTokenizer
        # Load model and extract metadata from GGUF
        self._model = Transformer.build(model_path, **kwargs)
        # Extract tokenizer, chat_template from GGUF metadata
        ...

    def tokenizer(self) -> Tokenizer:
        assert self._tokenizer is not None
        return self._tokenizer

    def chat_template_source(self) -> str:
        return self._chat_template

    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        """Wrap Transformer.generate() which already yields token IDs."""
        assert self._model is not None
        for token_id in self._model.generate(
            list(request.prompt_tokens),
            temperature=request.sampling.temperature,
        ):
            yield token_id

    def model_name(self) -> str:
        return self._model_name
```

### vLLM backend (`egg_toolbox/backends/vllm.py`)

```python
class VLLMBackend(ConstraintBackend):
    """Integration with vLLM's AsyncLLM.

    Uses constraint injection:
    - structured_outputs for JSON schema constraints
    - stop strings passed in SamplingParams
    - Custom logits processors for grammar constraints (future)
    """

    async def generate_stream(self, request: CompiledRequest) -> AsyncIterator[str]:
        from vllm import SamplingParams as VLLMSamplingParams
        from vllm import AsyncLLM

        vllm_params = VLLMSamplingParams(
            temperature=request.sampling.temperature,
            top_p=request.sampling.top_p,
            max_tokens=request.sampling.max_tokens,
            stop=list(request.stop_strings),
            stop_token_ids=list(request.stop_token_ids),
        )

        # Inject structured_outputs if we have a JSON schema
        if request.json_schema:
            vllm_params.structured_outputs = {
                "type": "json_schema",
                "value": request.json_schema,
            }

        async for output in self._engine.generate(
            prompt_token_ids=request.prompt_tokens,
            sampling_params=vllm_params,
        ):
            # vLLM yields RequestOutput with .outputs[0].text
            # We compute delta from previous text
            new_text = output.outputs[0].text
            delta = new_text[len(self._prev_text):]
            self._prev_text = new_text
            if delta:
                yield delta
```

---

## 10. Orchestrator (`egg_toolbox/orchestrator.py`)

The orchestrator is the central coordinator. It owns the request lifecycle:
`receive request -> compile prompt -> generate -> parse -> emit events`

```python
from __future__ import annotations

import uuid
from typing import AsyncIterator

from .types import (
    ChatMessage, CompiledRequest, SamplingParams, SemanticEvent,
    EventKind, StopReason, Tool, FormatAnalysis,
)
from .template import ChatTemplate
from .detector import detect_format
from .parser import StreamingParser
from .formats import get_handler_for_format
from .backends.base import Backend, StepBackend, ConstraintBackend


class Orchestrator:
    """Central coordinator for the tool-calling middleware.

    One Orchestrator instance per loaded model. Constructed at model load time.
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self._tokenizer = backend.tokenizer()

        # Load and analyze the chat template
        template_source = backend.chat_template_source()
        self._template = ChatTemplate(
            template_str=template_source,
            bos_token=self._tokenizer.decode_single(self._tokenizer.bos_token_id)
                if self._tokenizer.bos_token_id is not None else "",
            eos_token=self._tokenizer.decode_single(self._tokenizer.eos_token_id),
        )

        # Detect tool calling format
        self._format_analysis: FormatAnalysis = detect_format(self._template)
        self._handler = get_handler_for_format(self._format_analysis)

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        sampling: SamplingParams = SamplingParams(),
        stream: bool = True,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute a chat completion request and yield semantic events.

        This is the primary entry point. The API layer calls this and
        projects SemanticEvents into OpenAI or Anthropic SSE format.
        """

        # 1. Render the prompt using the model's chat template
        prompt_str = self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )
        prompt_tokens = self._tokenizer.encode(prompt_str)

        # 2. Compute stop conditions from format handler
        stop_strings = self._handler.stop_strings() + sampling.stop
        stop_token_ids = self._handler.stop_token_ids(self._tokenizer) + (self._tokenizer.eos_token_id,)

        # 3. Optionally generate grammar constraints
        grammar = None
        json_schema = None
        if tools:
            grammar = self._handler.generate_grammar(tools)
            json_schema = self._handler.generate_json_schema(tools)

        # 4. Build compiled request
        request = CompiledRequest(
            prompt_tokens=prompt_tokens,
            sampling=sampling,
            stop_strings=stop_strings,
            stop_token_ids=stop_token_ids,
            grammar=grammar,
            json_schema=json_schema,
            format_handler_name=type(self._handler).__name__,
        )

        # 5. Create parser
        parser = StreamingParser(
            handler=self._handler,
            tools=tools,
        )

        # 6. Generate and parse based on backend type
        if isinstance(self._backend, StepBackend):
            async for event in self._run_step_backend(request, parser):
                yield event
        elif isinstance(self._backend, ConstraintBackend):
            async for event in self._run_constraint_backend(request, parser):
                yield event

    async def _run_step_backend(
        self,
        request: CompiledRequest,
        parser: StreamingParser,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute generation on a step backend (tinygrad, llama-cpp-python).

        We control the token loop. For each token:
        1. Decode it
        2. Check stop conditions
        3. Feed it to the parser
        4. Yield any events the parser produces
        5. Optionally activate grammar constraints when trigger is detected
        """
        import asyncio

        assert isinstance(self._backend, StepBackend)

        token_count = 0
        max_tokens = request.sampling.max_tokens

        for token_id in self._backend.generate_tokens(request):
            # Check stop token
            if token_id in request.stop_token_ids:
                break

            token_text = self._tokenizer.decode_single(token_id)

            # Check stop strings
            # (simplified; real impl needs partial-match buffering)
            should_stop = False
            for stop_str in request.stop_strings:
                if stop_str in token_text:
                    should_stop = True
                    break

            if should_stop:
                break

            # Feed to parser
            events = parser.feed_token(token_id, token_text)
            for event in events:
                yield event

            token_count += 1
            if max_tokens and token_count >= max_tokens:
                break

        # Finalize
        for event in parser.finish():
            yield event

    async def _run_constraint_backend(
        self,
        request: CompiledRequest,
        parser: StreamingParser,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute generation on a constraint backend (vLLM, SGLang).

        The backend controls the loop. We inject constraints and parse output.
        """
        assert isinstance(self._backend, ConstraintBackend)

        async for text_chunk in self._backend.generate_stream(request):
            events = parser.feed_text(text_chunk)
            for event in events:
                yield event

        for event in parser.finish():
            yield event
```

---

## 11. API Layer (`egg_toolbox/api/`)

### OpenAI-Compatible API (`egg_toolbox/api/openai.py`)

Projects `SemanticEvent` stream into OpenAI `/v1/chat/completions` SSE format.

```python
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from ..types import (
    ChatMessage, ContentPart, SamplingParams, SemanticEvent,
    EventKind, StopReason, Tool, ToolCall, ToolCallFunction,
)
from ..orchestrator import Orchestrator


async def chat_completions(request: Request, orchestrator: Orchestrator):
    """Handle POST /v1/chat/completions"""
    body = await request.json()

    # Parse request
    messages = _parse_messages(body["messages"])
    tools = _parse_tools(body.get("tools")) if body.get("tools") else None
    sampling = _parse_sampling(body)
    stream = body.get("stream", False)
    model_name = body.get("model", orchestrator._backend.model_name())

    if stream:
        return StreamingResponse(
            _stream_response(orchestrator, messages, tools, sampling, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        return await _non_stream_response(orchestrator, messages, tools, sampling, model_name)


async def _stream_response(
    orchestrator: Orchestrator,
    messages: list[ChatMessage],
    tools: list[Tool] | None,
    sampling: SamplingParams,
    model_name: str,
):
    """Generate SSE events in OpenAI streaming format."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Initial role chunk
    chunk = _make_chunk(request_id, created, model_name, {
        "role": "assistant",
        "content": "",
    })
    yield f"data: {json.dumps(chunk)}\n\n"

    # Stream semantic events
    async for event in orchestrator.chat_completion(messages, tools, sampling, stream=True):
        chunks = _event_to_openai_chunks(event, request_id, created, model_name)
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


def _event_to_openai_chunks(
    event: SemanticEvent,
    request_id: str,
    created: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Convert a SemanticEvent to zero or more OpenAI SSE chunks."""
    chunks = []

    if event.kind == EventKind.CONTENT_DELTA:
        chunks.append(_make_chunk(request_id, created, model_name, {
            "content": event.text,
        }))

    elif event.kind == EventKind.REASONING_DELTA:
        chunks.append(_make_chunk(request_id, created, model_name, {
            "reasoning_content": event.text,
        }))

    elif event.kind == EventKind.TOOL_CALL_START:
        chunks.append(_make_chunk(request_id, created, model_name, {
            "tool_calls": [{
                "index": event.tool_index,
                "id": event.tool_call_id,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }],
        }))

    elif event.kind == EventKind.TOOL_CALL_NAME:
        chunks.append(_make_chunk(request_id, created, model_name, {
            "tool_calls": [{
                "index": event.tool_index,
                "function": {"name": event.tool_name},
            }],
        }))

    elif event.kind == EventKind.TOOL_ARGS_DELTA:
        chunks.append(_make_chunk(request_id, created, model_name, {
            "tool_calls": [{
                "index": event.tool_index,
                "function": {"arguments": event.text},
            }],
        }))

    elif event.kind == EventKind.DONE:
        finish_reason = {
            StopReason.STOP: "stop",
            StopReason.LENGTH: "length",
            StopReason.TOOL_CALLS: "tool_calls",
            StopReason.ERROR: "stop",
        }.get(event.stop_reason, "stop")

        chunks.append(_make_chunk(request_id, created, model_name,
            {}, finish_reason=finish_reason))

    return chunks


def _make_chunk(
    request_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


# -- Request parsing helpers --

def _parse_messages(raw: list[dict]) -> list[ChatMessage]: ...
def _parse_tools(raw: list[dict]) -> list[Tool]: ...
def _parse_sampling(body: dict) -> SamplingParams: ...
```

### Anthropic-Compatible API (`egg_toolbox/api/anthropic.py`)

Projects the same `SemanticEvent` stream into Anthropic `/v1/messages` SSE format.

```python
async def messages(request: Request, orchestrator: Orchestrator):
    """Handle POST /v1/messages"""
    body = await request.json()

    messages = _parse_anthropic_messages(body["messages"])
    tools = _parse_anthropic_tools(body.get("tools"))
    sampling = _parse_anthropic_sampling(body)
    stream = body.get("stream", False)
    model_name = body.get("model", orchestrator._backend.model_name())

    if stream:
        return StreamingResponse(
            _stream_anthropic(orchestrator, messages, tools, sampling, model_name),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_anthropic(orchestrator, messages, tools, sampling, model_name)


def _event_to_anthropic_events(
    event: SemanticEvent,
    block_index: int,
) -> list[dict[str, Any]]:
    """Convert SemanticEvent to Anthropic SSE events.

    Anthropic format uses content_block_start / content_block_delta /
    content_block_stop events with typed blocks.
    """
    sse_events = []

    if event.kind == EventKind.CONTENT_DELTA:
        sse_events.append({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": event.text},
        })

    elif event.kind == EventKind.REASONING_DELTA:
        sse_events.append({
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "thinking_delta", "thinking": event.text},
        })

    elif event.kind == EventKind.TOOL_CALL_START:
        sse_events.append({
            "type": "content_block_start",
            "index": event.tool_index,
            "content_block": {
                "type": "tool_use",
                "id": event.tool_call_id,
                "name": "",
                "input": {},
            },
        })

    elif event.kind == EventKind.TOOL_ARGS_DELTA:
        sse_events.append({
            "type": "content_block_delta",
            "index": event.tool_index,
            "delta": {"type": "input_json_delta", "partial_json": event.text},
        })

    elif event.kind == EventKind.TOOL_CALL_COMMIT:
        sse_events.append({
            "type": "content_block_stop",
            "index": event.tool_index,
        })

    elif event.kind == EventKind.DONE:
        stop_reason = {
            StopReason.STOP: "end_turn",
            StopReason.LENGTH: "max_tokens",
            StopReason.TOOL_CALLS: "tool_use",
        }.get(event.stop_reason, "end_turn")

        sse_events.append({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
        })
        sse_events.append({"type": "message_stop"})

    return sse_events
```

### ASGI Application (`egg_toolbox/api/middleware.py`)

```python
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .openai import chat_completions
from .anthropic import messages


def create_app(orchestrator) -> Starlette:
    routes = [
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/messages", messages, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ]
    return Starlette(
        routes=routes,
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"]),
        ],
    )
```

---

## 12. Grammar Generation (`egg_toolbox/grammar.py`)

Generates GBNF grammars from tool definitions. Used by step backends (tinygrad, llama-cpp-python) to constrain output to valid tool calls.

```python
from __future__ import annotations

from .types import Tool, FormatAnalysis, ToolFormatMode


def generate_gbnf(
    tools: list[Tool],
    analysis: FormatAnalysis,
) -> str:
    """Generate a GBNF grammar that constrains output to valid tool calls.

    The grammar structure depends on the tool format:
    - HERMES: <tool_call>{"name":"...", "arguments":{...}}</tool_call>
    - LLAMA3: func_name(arg1=val1, arg2=val2)
    - GENERIC_JSON: {"name":"...", "arguments":{...}}

    The grammar allows the model to freely generate content OR tool calls
    (lazy grammar pattern). Grammar activation happens at the trigger word.
    """
    if analysis.tool_mode == ToolFormatMode.HERMES:
        return _hermes_gbnf(tools, analysis)
    elif analysis.tool_mode == ToolFormatMode.LLAMA3:
        return _llama3_gbnf(tools, analysis)
    elif analysis.tool_mode == ToolFormatMode.GENERIC_JSON:
        return _generic_json_gbnf(tools, analysis)
    else:
        return _generic_json_gbnf(tools, analysis)


def _hermes_gbnf(tools: list[Tool], analysis: FormatAnalysis) -> str:
    """GBNF for Hermes-style tool calls.

    root ::= (content | tool-call)+
    tool-call ::= "<tool_call>" tool-json "</tool_call>"
    tool-json ::= "{" ws "\"name\"" ws ":" ws tool-name "," ws "\"arguments\"" ws ":" ws tool-args ws "}"
    tool-name ::= "\"func_a\"" | "\"func_b\"" | ...
    tool-args ::= <JSON schema for each tool's parameters>
    """
    lines = []
    lines.append('root ::= content* (tool-call content*)* ')
    lines.append(f'tool-call ::= "{analysis.tool_call_start}" tool-json "{analysis.tool_call_end}"')

    # Build tool-name alternatives
    name_alts = " | ".join(f'"\\"{t.function.name}\\""' for t in tools)
    lines.append(f'tool-name ::= {name_alts}')

    # Build tool-json
    lines.append('tool-json ::= "{\\"name\\":" ws tool-name ",\\"arguments\\":" ws tool-args "}"')

    # Build tool-args as union of per-tool schemas
    args_alts = []
    for i, tool in enumerate(tools):
        rule_name = f'args-{i}'
        schema_rules = _json_schema_to_gbnf(tool.to_json_schema(), rule_name)
        lines.extend(schema_rules)
        args_alts.append(rule_name)
    lines.append(f'tool-args ::= {" | ".join(args_alts)}')

    # Standard JSON primitives
    lines.extend(_json_primitives_gbnf())

    return "\n".join(lines)


def _json_schema_to_gbnf(schema: dict, rule_prefix: str) -> list[str]:
    """Convert a JSON Schema to GBNF rules.

    Handles: object, array, string, number, integer, boolean, null,
    enum, oneOf, anyOf, $ref (flattened).
    """
    ...


def _json_primitives_gbnf() -> list[str]:
    """Standard GBNF rules for JSON primitives."""
    return [
        'ws ::= [ \\t\\n]*',
        'content ::= [^<]*',
        'json-string ::= "\\"" [^"\\\\]* "\\"" ',
        'json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
        'json-bool ::= "true" | "false"',
        'json-null ::= "null"',
    ]
```

---

## 13. CLI Entry Point (`egg_toolbox/__main__.py`)

```python
import argparse
import asyncio
import uvicorn

def main():
    parser = argparse.ArgumentParser(
        description="egg-toolbox: Universal tool calling middleware for local LLMs"
    )
    parser.add_argument("model", help="Path to model (GGUF file or HF model ID)")
    parser.add_argument("--backend", choices=["tinygrad", "vllm", "sglang", "llamacpp"],
                        default="tinygrad", help="Backend to use")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--chat-template", help="Override chat template (path to .jinja file)")
    parser.add_argument("--tool-format", help="Override auto-detected tool format",
                        choices=["hermes", "llama3", "mistral", "deepseek",
                                 "functionary", "command_r", "generic"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (llamacpp)")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="TP size (vllm/sglang)")

    args = parser.parse_args()

    # Load backend
    backend = _create_backend(args)
    backend.load_model(args.model)

    # Create orchestrator
    from .orchestrator import Orchestrator
    from .api.middleware import create_app
    orchestrator = Orchestrator(backend)

    # Create and run ASGI app
    app = create_app(orchestrator)
    uvicorn.run(app, host=args.host, port=args.port)


def _create_backend(args):
    if args.backend == "tinygrad":
        from .backends.tinygrad import TinygradBackend
        return TinygradBackend()
    elif args.backend == "vllm":
        from .backends.vllm import VLLMBackend
        return VLLMBackend(tensor_parallel_size=args.tensor_parallel)
    elif args.backend == "sglang":
        from .backends.sglang import SGLangBackend
        return SGLangBackend(tensor_parallel_size=args.tensor_parallel)
    elif args.backend == "llamacpp":
        from .backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend(n_gpu_layers=args.gpu_layers)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
```

---

## 14. Key Algorithms

### 14.1 Streaming Parser State Machine (General Pattern)

Every format handler implements a state machine with these characteristics:

```
States:
  CONTENT         -- emitting content deltas (default state)
  REASONING       -- inside a reasoning/thinking block
  MAYBE_TRIGGER   -- buffer matches a prefix of a trigger word
  IN_TOOL_CALL    -- accumulating tool call data
  BETWEEN_CALLS   -- after one tool call, before next or EOS

Transitions:
  CONTENT -> REASONING     : when reasoning_start marker detected
  REASONING -> CONTENT     : when reasoning_end marker detected
  CONTENT -> MAYBE_TRIGGER : when partial trigger prefix seen
  MAYBE_TRIGGER -> CONTENT : when buffer fails to match trigger
  MAYBE_TRIGGER -> IN_TOOL : when full trigger matched
  IN_TOOL -> BETWEEN_CALLS : when tool call end marker detected
  BETWEEN_CALLS -> IN_TOOL : when another trigger detected
  BETWEEN_CALLS -> CONTENT : on EOS or non-tool content
```

The critical correctness property: **partial match buffering**. When the token stream contains a partial prefix of a trigger word (e.g., `<tool` but not yet `<tool_call>`), the parser must buffer without emitting, then either:
- Flush the buffer as content if the match fails
- Transition to IN_TOOL if the match succeeds

This is implemented in every format handler's `_process_content` method.

### 14.2 Stop String Partial Match (for Step Backends)

```python
class StopStringMatcher:
    """Efficient partial-match detector for multiple stop strings.

    Used by the orchestrator's step-backend loop to detect stop
    conditions without false negatives from partial token matches.
    """

    def __init__(self, stop_strings: tuple[str, ...]):
        self._stops = stop_strings
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str | None]:
        """Feed new text. Returns (safe_text, matched_stop_or_none).

        safe_text: text that can be emitted (definitely not part of a stop string)
        matched_stop: the stop string that was matched, or None
        """
        self._buffer += text

        # Check for complete matches
        for stop in self._stops:
            idx = self._buffer.find(stop)
            if idx >= 0:
                safe = self._buffer[:idx]
                self._buffer = ""
                return safe, stop

        # Check for partial matches (suffix of buffer = prefix of any stop string)
        max_partial = 0
        for stop in self._stops:
            for i in range(1, min(len(self._buffer), len(stop)) + 1):
                if self._buffer[-i:] == stop[:i]:
                    max_partial = max(max_partial, i)

        if max_partial > 0:
            safe = self._buffer[:-max_partial]
            self._buffer = self._buffer[-max_partial:]
            return safe, None
        else:
            safe = self._buffer
            self._buffer = ""
            return safe, None

    def flush(self) -> str:
        """Flush remaining buffer (at end of generation)."""
        remaining = self._buffer
        self._buffer = ""
        return remaining
```

### 14.3 Lazy Grammar Activation (for Step Backends)

The lazy grammar pattern allows the model to freely generate content, then constrains to grammar when a tool call trigger is detected.

```
Algorithm:
1. Start generation WITHOUT grammar constraints
2. For each token:
   a. Feed token to parser
   b. If parser.trigger_detected() transitions to True:
      - Activate grammar constraint for remaining generation
      - This is only possible on step backends where we control sampling
3. After trigger, grammar ensures all subsequent tokens form valid JSON

Implementation note: For tinygrad, "activating grammar" means switching
the sampling function to constrain logits. This requires the backend to
expose a logits-masking hook, which tinygrad does not currently have.

Practical approach for Phase 1: Do NOT implement lazy grammar for tinygrad.
Instead, rely purely on parsing. Grammar is a quality-of-life optimization
that can be added later when tinygrad exposes a logits processor hook.

For llama-cpp-python: Use their existing GBNF grammar sampler support.
Activate it at trigger detection by re-creating the sampler chain.
```

### 14.4 Format Auto-Detection via Differential Analysis

```
Algorithm (Python port of llama.cpp autoparser):

Input: ChatTemplate object

Step 1: Probe capabilities
  - supports_tools = "tools" in template.source
  - supports_reasoning = "reasoning_content" in template.source
  - supports_parallel = "for " in source and "tool_call" in source

Step 2: If supports_tools, run tool-call differential analysis

  SENTINEL values (chosen to be unique and unlikely to appear in templates):
    FUN_FIRST  = "FFF_FIRST_FUN_F"
    FUN_SECOND = "SSS_SECOND_FUN_S"
    ARG_FIRST  = "AA_ARG_FST_AA"
    ARG_SECOND = "BB_ARG_SND_BB"

  T1 -- Format classification:
    Render A: messages=[user, assistant_with_1_tool_call(name=FUN_FIRST, args={ARG_FIRST: ARG_FIRST})]
    Render B: messages=[user, assistant_with_content_only("content text")]
    diff = calculate_diff_split(A, B)
    tool_mode = classify_tool_format(diff, template)

  T2 -- Per-call markers (if supports_parallel):
    Render A: 1 tool call
    Render B: 2 tool calls
    diff = calculate_diff_split(A, B)
    -> Extract per_call_start, per_call_end from B's extra text

  T3 -- Name markers:
    Render A: tool call with name=FUN_FIRST
    Render B: tool call with name=FUN_SECOND
    diff = calculate_diff_split(A, B)
    -> name_prefix = diff.prefix (between per_call_start and the name)
    -> name_suffix = diff.suffix (between the name and args)

  T4 -- Argument structure (TAG_WITH_TAGGED only):
    Render A: 1 arg
    Render B: 2 args
    diff = calculate_diff_split(A, B)
    -> Extract argument separators, name/value markers

  T5 -- Call ID markers:
    Render A: tool_call_id="call001"
    Render B: tool_call_id="call002"
    diff = calculate_diff_split(A, B)
    -> Extract ID prefix, suffix, position

Step 3: Reasoning analysis (if supports_reasoning)
  R1: Render with reasoning_content="REASON_A" vs without
  diff -> Extract reasoning_start, reasoning_end

Step 4: Assemble FormatAnalysis from all extracted markers
```

### 14.5 Speculative Decoding with Grammar Constraints

Speculative decoding uses a small "draft" model to propose multiple tokens at once, then verifies them against the target model in a single forward pass. This interacts with egg-toolbox at two levels:

#### Level 1: Backend-transparent (Phase 3)

Speculative decoding that happens entirely within the backend is invisible to egg_toolbox. The backend's `generate_tokens()` still yields verified tokens one at a time via `Iterator[int]`. This works automatically with:

| Backend | How to enable | egg-toolbox changes |
|---|---|---|
| llama-cpp-python | `n_draft=N` parameter + draft model | None -- verified tokens come through same interface |
| vLLM | `--speculative-model` flag | None -- text chunks arrive via same `AsyncIterator[str]` |
| SGLang | `--speculative-model` flag | None |
| tinygrad | Not yet supported upstream | N/A |

#### Level 2: Grammar-guided speculation (Phase 4)

When grammar constraints are active (lazy grammar activated after trigger detection), draft tokens can be rejected early if they violate the grammar -- before the expensive target model verification. This requires a `GrammarState` abstraction:

```python
class GrammarState(abc.ABC):
    """Tracks grammar acceptance state for constrained decoding.

    Used during speculative decoding to reject draft tokens
    that violate grammar constraints before target model verification.
    """

    @abc.abstractmethod
    def is_valid(self, token_id: int) -> bool:
        """Check if a token is valid in the current grammar state."""
        ...

    @abc.abstractmethod
    def valid_token_ids(self) -> set[int]:
        """Return the set of all valid next token IDs.

        Used for logit masking during draft generation.
        """
        ...

    @abc.abstractmethod
    def advance(self, token_id: int) -> None:
        """Advance the grammar state by accepting a token.

        Raises ValueError if the token is invalid.
        """
        ...

    @abc.abstractmethod
    def clone(self) -> "GrammarState":
        """Create a copy of the current state.

        Needed because speculative decoding may need to roll back
        if draft tokens are rejected by the target model.
        """
        ...
```

The `StepBackend` ABC gains an optional extension:

```python
class StepBackend(abc.ABC):
    # ... existing methods ...

    @property
    def supports_speculative(self) -> bool:
        """Whether this backend supports grammar-guided speculative decoding."""
        return False

    def generate_tokens_speculative(
        self,
        request: CompiledRequest,
        grammar_state: GrammarState | None = None,
    ) -> Iterator[int]:
        """Generate tokens with optional grammar-guided speculation.

        If grammar_state is provided:
        1. Draft model proposes N tokens
        2. Each draft token is checked against grammar_state.is_valid()
        3. Invalid draft tokens are rejected immediately (no target verification needed)
        4. Valid draft tokens are verified against target model
        5. Accepted tokens advance the grammar state

        Default implementation falls back to generate_tokens().
        """
        yield from self.generate_tokens(request)
```

#### Orchestrator integration

The orchestrator's step-backend loop gains grammar-aware speculation:

```
Algorithm (grammar-guided speculative decoding):

1. Start generation without grammar (lazy grammar pattern)
2. For each token from generate_tokens():
   a. Feed token to parser
   b. If parser.trigger_detected() transitions to True AND backend.supports_speculative:
      - Build GrammarState from GBNF grammar
      - Switch to generate_tokens_speculative(request, grammar_state)
      - For each subsequent verified token:
        * grammar_state.advance(token)
        * Feed to parser
        * Yield events
3. On rollback (draft rejection):
   - grammar_state was cloned before draft phase
   - Restore cloned state, continue from last accepted position
```

This is a Phase 4 optimization. The Phase 1-3 path works without it -- grammar-guided speculation only improves throughput for constrained generation segments.

---

## 15. Implementation Phases

### Phase 1: Foundation

**Goal**: Hermes tool calling on tinygrad, OpenAI streaming API

| Priority | Component | Details |
|---|---|---|
| P0 | `types.py` | All frozen dataclasses, enums |
| P0 | `template.py` | Jinja2 engine with GGUF extraction |
| P0 | `formats/hermes.py` | Hermes parser state machine |
| P0 | `parser.py` | StreamingParser wrapping FormatParserState |
| P0 | `backends/tinygrad.py` | Wrap Transformer.generate() |
| P0 | `orchestrator.py` | Step-backend flow only |
| P0 | `api/openai.py` | /v1/chat/completions with streaming |
| P0 | `__main__.py` | CLI entry point |
| P1 | `detector.py` | Stub: return HERMES if template contains `<tool_call>` |
| P1 | Tests | Template rendering, Hermes parsing, OpenAI SSE format |

**Validation**: Load a Qwen 2.5 GGUF in tinygrad, send a tool-calling request
via OpenAI client, receive streaming tool calls back.

### Phase 2: Format Coverage

**Goal**: All major formats, auto-detection, Anthropic API

| Priority | Component | Details |
|---|---|---|
| P0 | `detector.py` | Full differential analysis algorithm |
| P0 | `formats/llama3.py` | Llama 3.x format handler |
| P0 | `formats/mistral.py` | Mistral format handler |
| P0 | `formats/deepseek.py` | DeepSeek format handler |
| P1 | `formats/functionary.py` | Functionary v3/v3.1 handlers |
| P1 | `formats/command_r.py` | Command-R handler |
| P1 | `formats/harmony.py` | Harmony multi-channel handler (analysis/commentary/final) |
| P1 | `formats/generic.py` | Generic JSON fallback |
| P0 | `api/anthropic.py` | /v1/messages with streaming |
| P1 | `grammar.py` | GBNF generation for Hermes format |
| P1 | Tests | Per-format parser tests with real model outputs |

**Validation**: Auto-detect format for 20+ popular models' templates.
Parse recorded outputs from each format correctly.

### Phase 3: Backend Expansion

**Goal**: llama-cpp-python and vLLM backends

| Priority | Component | Details |
|---|---|---|
| P0 | `backends/llamacpp.py` | Llama class integration with grammar support |
| P0 | `backends/vllm.py` | AsyncLLM integration with structured_outputs |
| P1 | `backends/sglang.py` | Engine integration with Xgrammar |
| P0 | `grammar.py` | Full JSON-schema-to-GBNF for all formats |
| P1 | Lazy grammar | For llama-cpp-python backend |
| P1 | Speculative decoding | Backend-transparent: enable via backend config (`n_draft`, `--speculative-model`). No egg-toolbox changes needed -- verified tokens flow through same interfaces |
| P1 | Tests | Integration tests per backend |

**Validation**: Same test suite passes on all 4 backends.
Grammar constraints produce valid tool calls on llama-cpp-python.
Speculative decoding (where backend supports it) produces identical tool call results.

### Phase 4: Production Hardening

**Goal**: Robustness, performance, edge cases

| Priority | Component | Details |
|---|---|---|
| P0 | Error handling | Malformed JSON recovery, partial tool calls |
| P0 | Timeout/cancellation | Request cancellation, generation timeouts |
| P1 | Parallel tool calls | Multiple tool calls in single response |
| P1 | Token usage tracking | prompt_tokens, completion_tokens in responses |
| P1 | Non-streaming mode | Full non-streaming response assembly |
| P1 | Grammar-guided speculation | `GrammarState` class for draft-phase token rejection. Extends `StepBackend` with `generate_tokens_speculative()`. See Section 14.5 |
| P2 | Performance | Token throughput measurement, bottleneck profiling |
| P2 | Documentation | API docs, format handler guide, backend guide |

---

## 16. Design Decisions

### Pattern Choice: State Machine Parser (not Regex)

**Chosen**: Per-format state machines for streaming parsing.

**Why**: Regex-based parsing requires the complete text to be available. State machines process tokens incrementally, which is essential for streaming. Each token triggers at most one state transition and emits zero or more events.

**Trade-off**: More code per format handler than a regex approach. But streaming correctness is non-negotiable.

### Pattern Choice: Frozen Dataclasses (not Pydantic)

**Chosen**: Standard library `dataclasses` with `frozen=True`.

**Why**: Zero dependencies for the type layer. Pydantic adds import time, complexity, and a large dependency. Frozen dataclasses enforce immutability naturally. JSON serialization is handled explicitly in the API layer where it belongs.

**Trade-off**: No automatic validation or schema generation from Pydantic. But the API layer handles validation explicitly anyway.

### Pattern Choice: ASGI/Starlette (not FastAPI)

**Chosen**: Starlette for the HTTP layer.

**Why**: FastAPI adds Pydantic model generation overhead and auto-docs that we don't need. Starlette is lighter, still ASGI-native, and supports streaming responses natively. FastAPI is built on Starlette anyway.

**Trade-off**: No auto-generated OpenAPI docs. But the API surface is small and well-defined.

### Decision: No Rust

**Chosen**: Pure Python.

**Why**: The original design document suggested a compiled Rust core. After analysis, the hot path in this middleware is NOT in Python: it's in the backends (tinygrad's GPU kernels, vLLM's C++/CUDA, llama.cpp's GGML). The middleware processes one token at a time through a state machine, which is trivially fast in Python. The bottleneck is always the model, never the parsing.

**Trade-off**: If the middleware ever needs to process thousands of concurrent requests with complex grammar matching, Python could become a bottleneck. But that scenario requires a serving engine (vLLM/SGLang) that already has its own tool parsing.

### Decision: SemanticEvent as the Universal IR

**Chosen**: A single `SemanticEvent` enum-based type as the internal representation.

**Why**: Both OpenAI and Anthropic streaming formats can be projected from the same event stream without loss. This keeps format-specific logic at the edges (format handlers produce events; API layers consume events) and the core clean.

**Alternative considered**: Separate event types per API format. Rejected because it would require format handlers to know about API formats, violating separation of concerns.

### Decision: Two Backend ABCs (not one)

**Chosen**: Separate `StepBackend` and `ConstraintBackend` ABCs.

**Why**: Step backends (tinygrad, llama-cpp-python) and constraint backends (vLLM, SGLang) have fundamentally different control flow. A single ABC would either be too broad (unused methods) or require runtime type checking. Two ABCs make the contract explicit.

**Alternative considered**: A single `Backend` ABC with optional methods. Rejected because it pushes the step-vs-constraint distinction into runtime, making bugs harder to find.

### Decision: Format Detection at Model Load (not per-request)

**Chosen**: Run differential analysis once at model load, cache the `FormatAnalysis`.

**Why**: The chat template doesn't change between requests. Running the detector per-request would waste time. The detector creates multiple template renders (7+ comparisons), each involving Jinja2 rendering, so caching is important.

**Trade-off**: Cannot handle models that change their tool format based on request parameters. No known models do this.

---

## 17. Dependency Graph

```
egg-toolbox (installable extras)
  Required:
    jinja2          # chat template rendering
    starlette       # ASGI HTTP framework
    uvicorn         # ASGI server
    httptools       # fast HTTP parsing for uvicorn

  Backend extras:
    egg-toolbox[tinygrad]:    tinygrad
    egg-toolbox[vllm]:        vllm
    egg-toolbox[sglang]:      sglang
    egg-toolbox[llamacpp]:    llama-cpp-python

  Dev extras:
    egg-toolbox[dev]:         pytest, pytest-asyncio, httpx
```

### `pyproject.toml` skeleton

```toml
[project]
name = "egg-toolbox"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jinja2>=3.1",
    "starlette>=0.36",
    "uvicorn[standard]>=0.27",
]

[project.optional-dependencies]
tinygrad = ["tinygrad"]
vllm = ["vllm>=0.6"]
sglang = ["sglang"]
llamacpp = ["llama-cpp-python>=0.3"]
dev = ["pytest", "pytest-asyncio", "httpx"]

[project.scripts]
egg-toolbox = "egg_toolbox.__main__:main"
```

---

## 18. Integration with egg-mono

egg-toolbox is a standalone library. egg-mono can consume it in two ways:

### Option A: As a remote server

egg-mono's `eggllm` routes requests to a running `egg-toolbox` server just like any other OpenAI-compatible endpoint. No code changes needed in eggllm.

### Option B: As a library dependency

```python
# In eggllm/providers/egg_toolbox_local.py
from egg_toolbox.orchestrator import Orchestrator
from egg_toolbox.backends.tinygrad import TinygradBackend

class OmnitoolLocalProvider(ProviderAdapter):
    def __init__(self, model_path: str):
        backend = TinygradBackend()
        backend.load_model(model_path)
        self.orchestrator = Orchestrator(backend)

    async def stream_async(self, url, headers, payload, **kwargs):
        messages = self._parse_eggllm_payload(payload)
        async for event in self.orchestrator.chat_completion(messages, ...):
            yield self._semantic_to_eggllm(event)
```

---

## 19. Test Strategy

### Unit tests (no model required)

| Test area | What's tested | Fixtures |
|---|---|---|
| `test_types.py` | Dataclass construction, immutability | None |
| `test_template.py` | Jinja2 rendering with tools, GGUF parsing | Collected templates from HF |
| `test_detector.py` | Format detection on 20+ real templates | Template strings |
| `test_parser.py` | StreamingParser integration | None |
| `test_formats/*.py` | Each format handler's state machine | Recorded token sequences |
| `test_grammar.py` | GBNF generation, schema conversion | Tool definitions |
| `test_api/*.py` | OpenAI/Anthropic SSE format correctness | SemanticEvent sequences |

### Integration tests (model required, CI-optional)

| Test area | What's tested |
|---|---|
| `test_backends/test_tinygrad.py` | End-to-end with a small GGUF model |
| `test_backends/test_llamacpp.py` | End-to-end with grammar constraints |
| `test_e2e.py` | Full HTTP request/response cycle |

### Test fixtures: Collected templates

The `tests/fixtures/templates/` directory should contain real chat templates from popular models:

```
qwen2.5-7b-instruct.jinja
llama-3.1-8b-instruct.jinja
mistral-7b-instruct-v0.3.jinja
deepseek-v3.jinja
hermes-2-pro.jinja
functionary-v3.1.jinja
command-r-plus.jinja
gemma-4-27b-it.jinja
```

These templates are extracted from HuggingFace `tokenizer_config.json` or GGUF files and committed as test fixtures. The detector tests verify that `detect_format()` produces the correct `ToolFormatMode` for each.

### Test fixtures: Recorded outputs

The `tests/fixtures/outputs/` directory should contain recorded token-by-token outputs from real models making tool calls. Format:

```json
{
  "model": "qwen2.5-7b-instruct",
  "format": "hermes",
  "tokens": [
    {"id": 151644, "text": "<tool_call>"},
    {"id": 5765, "text": "{\""},
    {"id": 609, "text": "name"},
    ...
  ],
  "expected_tool_calls": [
    {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
  ]
}
```

---

## 20. Validation Checklist

- [x] All requirements addressed (4 backends, 2 API formats, streaming, tool calling)
- [x] Module boundaries align with domain concepts (template, detection, parsing, backends, API)
- [x] Interfaces are minimal and complete (SemanticEvent is the universal IR)
- [x] Dependencies form a DAG (types <- template <- detector; types <- formats <- parser; types <- backends; all <- orchestrator <- api)
- [x] Each module has a single purpose
- [x] Design supports expected scale (one model per process; parser cost is negligible vs generation)
- [x] Security: Jinja2 sandboxed rendering; no eval/exec
- [x] Implementation path is incremental (Phase 1 delivers a working product)

---

## 21. Future Considerations

1. **Batching for step backends**: tinygrad and llama-cpp-python are single-request. If concurrent requests arrive, they queue. A future optimization could implement request batching in the orchestrator.

2. **Vision/multimodal**: The current design handles text content parts. Image content parts would need backend-specific handling (tinygrad doesn't support vision; vLLM does).

3. **Tool execution**: egg-toolbox is pure middleware (prompt in, tool calls out). It does not execute tools. Tool execution is the consumer's responsibility (e.g., egg-mono's `eggthreads`).

4. **Model hot-swapping**: Currently one model per process. Supporting multiple models or hot-swapping would require an orchestrator registry.

5. **MCP (Model Context Protocol)**: Could be added as an additional API projection from SemanticEvents, similar to the OpenAI and Anthropic projections.

6. **Speculative decoding**: Backend-transparent speculative decoding is planned for Phase 3 (works automatically via existing interfaces). Grammar-guided speculative decoding (draft-phase rejection via `GrammarState`) is planned for Phase 4. See Section 14.5 for full design.
