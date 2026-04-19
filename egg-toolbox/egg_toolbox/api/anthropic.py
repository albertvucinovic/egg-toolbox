"""Anthropic Messages API projection.

Exposes ``POST /v1/messages`` over the same ``SemanticEvent`` stream
used by the OpenAI projection.  The two API shapes are largely a
relabelling exercise on top of the universal IR:

- OpenAI ``role=system`` message  <->  Anthropic top-level ``system``
- OpenAI ``tool_calls`` on assistant  <->  Anthropic ``tool_use`` blocks
- OpenAI ``role=tool`` message  <->  Anthropic ``tool_result`` user block
- OpenAI ``function.parameters`` (JSON Schema)  <->  Anthropic ``input_schema``
- OpenAI ``stop`` (string or list)  <->  Anthropic ``stop_sequences``

Streaming: Anthropic uses named SSE events rather than bare ``data:``
lines.  The event sequence per tool call is
``content_block_start`` -> N x ``content_block_delta`` ->
``content_block_stop`` -> final ``message_delta`` -> ``message_stop``.

Because Anthropic's ``content_block_start`` for ``tool_use`` must
carry the tool ``name``, and our ``SemanticEvent.TOOL_CALL_START``
fires *before* ``TOOL_CALL_NAME`` arrives, we defer emitting the
``content_block_start`` for a tool until the NAME event lands.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any


# Synthesised ``signature`` we attach to ``thinking`` content blocks so
# Anthropic-SDK clients don't reject the output.  Real Anthropic
# signatures are cryptographic authenticity tags over the thinking
# payload; since this is a local runtime we produce a deterministic
# hash instead.  Clients that simply round-trip the field (the common
# case) are unaffected; clients that actively verify would reject, and
# that's acceptable -- they wouldn't trust an unknown issuer anyway.
_SIGNATURE_PREFIX = "egg-toolbox:"


def _synth_signature(thinking_text: str) -> str:
    digest = hashlib.sha256(thinking_text.encode("utf-8")).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest[:32]}"

from starlette.responses import JSONResponse, StreamingResponse

from ..types import (
    ChatMessage, ContentPart, SamplingParams, SemanticEvent,
    EventKind, StopReason, Tool, ToolFunction, ToolParameter,
    ToolCall, ToolCallFunction,
)
from ..orchestrator import Orchestrator


_STOP_REASON_MAP = {
    StopReason.STOP: "end_turn",
    StopReason.LENGTH: "max_tokens",
    StopReason.TOOL_CALLS: "tool_use",
    StopReason.ERROR: "end_turn",
}


async def messages(body: dict, orchestrator: Orchestrator):
    """Handle POST /v1/messages."""
    msgs = _parse_messages(body["messages"], body.get("system"))
    tools = _parse_tools(body.get("tools")) if body.get("tools") else None
    sampling = _parse_sampling(body)
    stream = body.get("stream", False)
    model_name = body.get("model", orchestrator._backend.model_name())

    tool_choice = body.get("tool_choice") or {}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        tools = None

    if stream:
        return StreamingResponse(
            _stream_response(orchestrator, msgs, tools, sampling, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _non_stream_response(orchestrator, msgs, tools, sampling, model_name)


# --------------------------------------------------------------------- #
# Streaming projection                                                  #
# --------------------------------------------------------------------- #

async def _stream_response(
    orchestrator: Orchestrator,
    msgs: list[ChatMessage],
    tools: list[Tool] | None,
    sampling: SamplingParams,
    model_name: str,
):
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Emit message_start
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    projector = _BlockProjector()
    done_event: SemanticEvent | None = None

    async for event in orchestrator.chat_completion(msgs, tools, sampling, stream=True):
        if event.kind == EventKind.DONE:
            done_event = event
        for sse_event, payload in projector.feed(event):
            yield _sse(sse_event, payload)

    # Close any still-open block.
    for sse_event, payload in projector.close():
        yield _sse(sse_event, payload)

    stop_reason = _STOP_REASON_MAP.get(
        done_event.stop_reason if done_event else StopReason.STOP,
        "end_turn",
    )
    input_tokens = (done_event.prompt_tokens if done_event else 0) or 0
    output_tokens = (done_event.completion_tokens if done_event else 0) or 0

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {
        "type": "message_stop",
    })


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


class _BlockProjector:
    """Tracks Anthropic block state over a stream of SemanticEvents.

    Anthropic content blocks are sequentially indexed and each has
    a ``content_block_start`` / ``*_delta`` / ``content_block_stop``
    lifecycle.  This class hides the bookkeeping from the outer
    streaming loop, returning ``(sse_event_name, payload)`` tuples.
    """

    def __init__(self) -> None:
        self._block_index = -1
        self._current: str | None = None      # "text" | "thinking" | "tool_use"
        # Map from TOOL_CALL_START tool_index to its assigned block_index.
        # Entries added at START, populated with name at NAME, emitted
        # as content_block_start at NAME time (so name is known).
        self._tool_block_index: dict[int, int] = {}
        self._tool_started: set[int] = set()   # have we emitted content_block_start?
        # Accumulated thinking text for the currently-open thinking
        # block.  On close we synthesise a signature over this text and
        # emit a ``signature_delta`` before ``content_block_stop`` --
        # matches Anthropic's native streaming shape, so SDK clients
        # that round-trip the signature do so cleanly.
        self._thinking_buffer: list[str] = []

    def feed(self, event: SemanticEvent) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []

        if event.kind == EventKind.CONTENT_DELTA:
            if event.text is None:
                return out
            if self._current != "text":
                out.extend(self._close_current())
                self._block_index += 1
                self._current = "text"
                out.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            out.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "text_delta", "text": event.text},
            }))

        elif event.kind == EventKind.REASONING_DELTA:
            if event.text is None:
                return out
            if self._current != "thinking":
                out.extend(self._close_current())
                self._block_index += 1
                self._current = "thinking"
                self._thinking_buffer = []
                out.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }))
            self._thinking_buffer.append(event.text)
            out.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "thinking_delta", "thinking": event.text},
            }))

        elif event.kind == EventKind.TOOL_CALL_START:
            # Close any open text/thinking block first; defer emitting
            # the tool's content_block_start until NAME arrives.
            out.extend(self._close_current())
            self._block_index += 1
            self._tool_block_index[event.tool_index] = self._block_index
            self._current = "tool_use"

        elif event.kind == EventKind.TOOL_CALL_NAME:
            block_idx = self._tool_block_index.get(event.tool_index)
            if block_idx is None:
                # TOOL_CALL_NAME without a prior START -- synthesise.
                self._block_index += 1
                block_idx = self._block_index
                self._tool_block_index[event.tool_index] = block_idx
                self._current = "tool_use"
            if event.tool_index not in self._tool_started:
                out.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{event.tool_index:02d}",
                        "name": event.tool_name or "",
                        "input": {},
                    },
                }))
                self._tool_started.add(event.tool_index)

        elif event.kind == EventKind.TOOL_ARGS_DELTA:
            if event.text is None:
                return out
            block_idx = self._tool_block_index.get(event.tool_index)
            if block_idx is None or event.tool_index not in self._tool_started:
                # Args before a proper start -- cannot project cleanly.
                return out
            out.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "input_json_delta", "partial_json": event.text},
            }))

        elif event.kind == EventKind.TOOL_CALL_COMMIT:
            block_idx = self._tool_block_index.get(event.tool_index)
            if block_idx is not None and event.tool_index in self._tool_started:
                out.append(("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_idx,
                }))
            self._current = None

        return out

    def close(self) -> list[tuple[str, dict]]:
        return self._close_current()

    def _close_current(self) -> list[tuple[str, dict]]:
        if self._current == "thinking":
            # Emit signature_delta right before stop, mirroring
            # Anthropic's native extended-thinking stream shape.  The
            # signature is synthesised from the accumulated text so
            # clients that validate it see a stable tag over the same
            # thinking body they received.
            thinking_text = "".join(self._thinking_buffer)
            self._thinking_buffer = []
            sig_event = ("content_block_delta", {
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {
                    "type": "signature_delta",
                    "signature": _synth_signature(thinking_text),
                },
            })
            stop_event = ("content_block_stop", {
                "type": "content_block_stop",
                "index": self._block_index,
            })
            self._current = None
            return [sig_event, stop_event]
        if self._current == "text":
            event = ("content_block_stop", {
                "type": "content_block_stop",
                "index": self._block_index,
            })
            self._current = None
            return [event]
        # tool_use blocks close on TOOL_CALL_COMMIT explicitly.
        return []


# --------------------------------------------------------------------- #
# Non-streaming projection                                              #
# --------------------------------------------------------------------- #

async def _non_stream_response(
    orchestrator: Orchestrator,
    msgs: list[ChatMessage],
    tools: list[Tool] | None,
    sampling: SamplingParams,
    model_name: str,
) -> JSONResponse:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    async for event in orchestrator.chat_completion(msgs, tools, sampling, stream=False):
        if event.kind == EventKind.CONTENT_DELTA and event.text:
            text_parts.append(event.text)
        elif event.kind == EventKind.REASONING_DELTA and event.text:
            reasoning_parts.append(event.text)
        elif event.kind == EventKind.TOOL_CALL_COMMIT:
            try:
                parsed_input = json.loads(event.tool_arguments or "{}")
            except (ValueError, TypeError):
                parsed_input = {}
            tool_calls.append({
                "type": "tool_use",
                "id": f"toolu_{event.tool_index:02d}",
                "name": event.tool_name or "",
                "input": parsed_input,
            })
        elif event.kind == EventKind.DONE:
            stop_reason = _STOP_REASON_MAP.get(event.stop_reason, "end_turn")
            input_tokens = event.prompt_tokens or 0
            output_tokens = event.completion_tokens or 0

    content: list[dict] = []
    if reasoning_parts:
        thinking_text = "".join(reasoning_parts)
        content.append({
            "type": "thinking",
            "thinking": thinking_text,
            "signature": _synth_signature(thinking_text),
        })
    if text_parts:
        content.append({"type": "text", "text": "".join(text_parts)})
    content.extend(tool_calls)
    if not content:
        content.append({"type": "text", "text": ""})

    return JSONResponse({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_name,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    })


# --------------------------------------------------------------------- #
# Request parsing                                                        #
# --------------------------------------------------------------------- #

def _parse_messages(raw: list[dict], system: Any) -> list[ChatMessage]:
    """Convert Anthropic-format messages into ChatMessage.

    Anthropic top-level ``system`` becomes a leading ``role=system``
    ChatMessage.  Anthropic ``content`` arrays with ``tool_use`` or
    ``tool_result`` blocks expand into our internal message shape.
    """
    out: list[ChatMessage] = []

    # System prompt (optional top-level).
    if system is not None:
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            sys_text = "".join(
                b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            sys_text = ""
        if sys_text:
            out.append(ChatMessage(role="system", content=sys_text))

    for m in raw:
        role = m.get("role", "user")
        content = m.get("content")

        # String content: pass through.
        if isinstance(content, str):
            out.append(ChatMessage(role=role, content=content))
            continue

        if not isinstance(content, list):
            out.append(ChatMessage(role=role, content=""))
            continue

        # Array content: split into text / thinking / tool_use / tool_result parts.
        text_pieces: list[str] = []
        thinking_pieces: list[str] = []
        tool_calls_out: list[ToolCall] = []
        tool_results: list[tuple[str, str]] = []  # (tool_use_id, result_text)
        image_parts: list[ContentPart] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_pieces.append(block.get("text", ""))
            elif btype == "thinking":
                # Assistant's prior chain-of-thought, echoed back for
                # replay.  ``signature`` is Anthropic's authenticity
                # tag; we accept it for wire compatibility but don't
                # verify since this is a local runtime.  Concatenate
                # if multiple thinking blocks appear (Anthropic allows
                # redacted_thinking blocks interspersed; we treat them
                # the same -- raw text round-trip).
                thinking_pieces.append(block.get("thinking", ""))
            elif btype == "redacted_thinking":
                # Anthropic's opaque "encrypted thinking" -- we don't
                # have keys to decrypt.  Preserve the placeholder so
                # the client's round-trip remains faithful; the
                # template ignores it.  (If a future local runtime
                # wanted to replay a hosted-Anthropic conversation,
                # it would need to skip these blocks; today we just
                # preserve their text payload if any.)
                thinking_pieces.append(block.get("data", ""))
            elif btype == "image":
                src = block.get("source") or {}
                image_parts.append(ContentPart(
                    type="image_url",
                    image_url=src.get("data") or src.get("url"),
                ))
            elif btype == "tool_use":
                # Assistant's prior tool call, echoed back for context.
                input_obj = block.get("input", {})
                args = json.dumps(input_obj) if not isinstance(input_obj, str) else input_obj
                tool_calls_out.append(ToolCall(
                    id=block.get("id", ""),
                    type="function",
                    function=ToolCallFunction(
                        name=block.get("name", ""),
                        arguments=args,
                    ),
                ))
            elif btype == "tool_result":
                use_id = block.get("tool_use_id", "")
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner_text = "".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    inner_text = str(inner)
                tool_results.append((use_id, inner_text))

        if tool_results:
            # Anthropic groups tool_results inside a user message; we
            # emit one ChatMessage(role="tool") per result.
            for use_id, result_text in tool_results:
                out.append(ChatMessage(
                    role="tool",
                    content=result_text,
                    tool_call_id=use_id,
                ))
            # If the same user-role block also contained plain text,
            # surface it as a separate user message after the tools.
            if text_pieces:
                out.append(ChatMessage(role="user", content="".join(text_pieces)))
            continue

        parts: list[ContentPart] = []
        if text_pieces:
            parts.append(ContentPart(type="text", text="".join(text_pieces)))
        parts.extend(image_parts)

        final_content: Any
        if image_parts:
            final_content = parts
        elif text_pieces:
            final_content = "".join(text_pieces)
        else:
            final_content = ""

        # Thinking content only makes sense on assistant history
        # messages; for other roles we just drop it (Anthropic's API
        # wouldn't accept it there either).
        reasoning_str = (
            "".join(thinking_pieces)
            if thinking_pieces and role == "assistant"
            else None
        )

        out.append(ChatMessage(
            role=role,
            content=final_content,
            tool_calls=tuple(tool_calls_out) if tool_calls_out else None,
            reasoning=reasoning_str,
        ))

    return out


def _parse_tools(raw: list[dict]) -> list[Tool]:
    """Anthropic tool definitions use ``input_schema`` rather than
    OpenAI's ``parameters``; otherwise equivalent."""
    tools: list[Tool] = []
    for t in raw:
        schema = t.get("input_schema") or t.get("parameters") or {}
        parameters: dict[str, ToolParameter] = {}
        required_fields: list[str] = schema.get("required", []) or []

        for pname, pdef in (schema.get("properties") or {}).items():
            parameters[pname] = ToolParameter(
                name=pname,
                type=pdef.get("type", "string"),
                description=pdef.get("description", ""),
                required=pname in required_fields,
                enum=tuple(pdef["enum"]) if "enum" in pdef else None,
            )

        tools.append(Tool(
            type="function",
            function=ToolFunction(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=parameters,
                required=tuple(required_fields),
            ),
        ))
    return tools


def _parse_sampling(body: dict) -> SamplingParams:
    kwargs: dict[str, Any] = {}
    if "max_tokens" in body:
        kwargs["max_tokens"] = int(body["max_tokens"])
    if "temperature" in body:
        kwargs["temperature"] = float(body["temperature"])
    if "top_p" in body:
        kwargs["top_p"] = float(body["top_p"])
    if "top_k" in body:
        kwargs["top_k"] = int(body["top_k"])
    if "stop_sequences" in body and body["stop_sequences"]:
        kwargs["stop"] = tuple(body["stop_sequences"])
    return SamplingParams(**kwargs)
