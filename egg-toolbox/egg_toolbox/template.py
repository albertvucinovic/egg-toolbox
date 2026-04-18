from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .types import ChatMessage, ContentPart, Tool, ToolCall, ToolCallFunction


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
        self._env.globals.update({
            "raise_exception": self._raise_exception,
            "strftime_now": self._strftime_now,
        })
        self._env.filters["tojson"] = self._tojson
        self._template = self._env.from_string(template_str)

    @staticmethod
    def from_gguf(path: str | Path) -> ChatTemplate:
        """Extract chat_template, BOS, and EOS from GGUF metadata."""
        path = Path(path)
        chat_template = ""
        bos_token = ""
        eos_token = ""
        token_list: list[str] = []
        bos_id: int | None = None
        eos_id: int | None = None

        with open(path, "rb") as f:
            # GGUF magic: 0x46475547 ("GGUF" in little-endian)
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file: {path}")

            version = struct.unpack("<I", f.read(4))[0]
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key = _read_gguf_string(f, version)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = _read_gguf_value(f, value_type, version)

                if key == "tokenizer.chat_template":
                    chat_template = value
                elif key == "tokenizer.ggml.bos_token_id":
                    bos_id = value
                elif key == "tokenizer.ggml.eos_token_id":
                    eos_id = value
                elif key == "tokenizer.ggml.tokens":
                    token_list = value

        if token_list:
            if bos_id is not None and 0 <= bos_id < len(token_list):
                bos_token = token_list[bos_id]
            if eos_id is not None and 0 <= eos_id < len(token_list):
                eos_token = token_list[eos_id]

        if not chat_template:
            raise ValueError(f"No chat_template found in GGUF metadata: {path}")

        return ChatTemplate(chat_template, bos_token=bos_token, eos_token=eos_token)

    @staticmethod
    def from_hf_config(path: str | Path) -> ChatTemplate:
        """Load from a HuggingFace tokenizer_config.json."""
        path = Path(path)
        with open(path) as f:
            config = json.load(f)

        template_str = config.get("chat_template", "")
        if not template_str:
            raise ValueError(f"No chat_template in {path}")

        bos_token = ""
        eos_token = ""
        if "bos_token" in config:
            bt = config["bos_token"]
            bos_token = bt if isinstance(bt, str) else bt.get("content", "")
        if "eos_token" in config:
            et = config["eos_token"]
            eos_token = et if isinstance(et, str) else et.get("content", "")

        return ChatTemplate(template_str, bos_token=bos_token, eos_token=eos_token)

    def render(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        """Render messages + tools into the full prompt string."""
        msg_dicts = [self._msg_to_dict(m) for m in messages]
        tool_dicts = [self._tool_to_dict(t) for t in tools] if tools else None

        kwargs: dict[str, Any] = {
            "messages": msg_dicts,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "add_generation_prompt": add_generation_prompt,
        }
        if tool_dicts is not None:
            kwargs["tools"] = tool_dicts
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking

        return self._template.render(**kwargs)

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
        return "tools" in self.source

    def supports_reasoning(self) -> bool:
        return "reasoning_content" in self.source or "enable_thinking" in self.source

    def supports_parallel_tools(self) -> bool:
        return "for " in self.source and "tool_call" in self.source

    # -- Internal helpers --

    @staticmethod
    def _msg_to_dict(msg: ChatMessage) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role}
        # Always include a 'content' key -- some chat templates
        # (notably Qwen3's) do bare `message.content` attribute
        # access, which Jinja translates to dict-key lookup and
        # which raises UndefinedError if the key is absent.  We
        # substitute an empty string for a missing/None content
        # so ``{% if message.content %}`` still evaluates falsy.
        if msg.content is None:
            d["content"] = ""
        elif isinstance(msg.content, str):
            d["content"] = msg.content
        else:
            d["content"] = [
                {"type": p.type, "text": p.text} if p.text is not None
                else {"type": p.type, "image_url": p.image_url}
                for p in msg.content
            ]
        if msg.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        # Parse the OpenAI-style JSON-string arguments into
                        # a dict so templates (which invariably expect an
                        # object here) serialise it as real JSON rather
                        # than a JSON-encoded string.  If parsing fails
                        # fall back to the original string so no info is
                        # lost.
                        "arguments": _parse_tool_arguments(tc.function.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            d["name"] = msg.name
        return d

    @staticmethod
    def _tool_to_dict(tool: Tool) -> dict[str, Any]:
        fn = tool.function
        properties: dict[str, Any] = {}
        required_list: list[str] = []
        for pname, param in fn.parameters.items():
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.enum is not None:
                prop["enum"] = list(param.enum)
            properties[pname] = prop
            if param.required:
                required_list.append(pname)
        # Also include required from the function definition
        for r in fn.required:
            if r not in required_list:
                required_list.append(r)

        return {
            "type": tool.type,
            "function": {
                "name": fn.name,
                "description": fn.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_list,
                },
            },
        }

    @staticmethod
    def _raise_exception(msg: str) -> None:
        raise jinja2.exceptions.TemplateRuntimeError(msg)

    @staticmethod
    def _strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    @staticmethod
    def _tojson(value: Any, indent: int | None = None) -> str:
        return json.dumps(value, indent=indent, ensure_ascii=False)


def _parse_tool_arguments(arguments: Any) -> Any:
    """Convert OpenAI-style JSON-string tool arguments into the dict
    shape that all chat templates expect.  Non-string inputs pass
    through untouched so this is safe to call on any source shape.
    """
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return arguments
    return parsed


# --- GGUF parsing helpers ---

# GGUF value type constants
_GGUF_TYPE_UINT8    = 0
_GGUF_TYPE_INT8     = 1
_GGUF_TYPE_UINT16   = 2
_GGUF_TYPE_INT16    = 3
_GGUF_TYPE_UINT32   = 4
_GGUF_TYPE_INT32    = 5
_GGUF_TYPE_FLOAT32  = 6
_GGUF_TYPE_BOOL     = 7
_GGUF_TYPE_STRING   = 8
_GGUF_TYPE_ARRAY    = 9
_GGUF_TYPE_UINT64   = 10
_GGUF_TYPE_INT64    = 11
_GGUF_TYPE_FLOAT64  = 12


def _read_gguf_string(f, version: int) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_gguf_value(f, value_type: int, version: int) -> Any:
    if value_type == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == _GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    elif value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(f, version)
    elif value_type == _GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, elem_type, version) for _ in range(count)]
    elif value_type == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")
