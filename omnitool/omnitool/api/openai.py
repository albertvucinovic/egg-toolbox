from __future__ import annotations

import json
import time
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from ..types import (
    ChatMessage, ContentPart, SamplingParams, SemanticEvent,
    EventKind, StopReason, Tool, ToolFunction, ToolParameter,
    ToolCall, ToolCallFunction,
)
from ..orchestrator import Orchestrator


async def chat_completions(request: Request, orchestrator: Orchestrator) -> StreamingResponse | JSONResponse:
    """Handle POST /v1/chat/completions"""
    body = await request.json()

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
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n"

    yield "data: [DONE]\n\n"


async def _non_stream_response(
    orchestrator: Orchestrator,
    messages: list[ChatMessage],
    tools: list[Tool] | None,
    sampling: SamplingParams,
    model_name: str,
) -> JSONResponse:
    """Collect all events and return a single JSON response."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_list: list[dict] = []
    finish_reason = "stop"

    async for event in orchestrator.chat_completion(messages, tools, sampling, stream=False):
        if event.kind == EventKind.CONTENT_DELTA and event.text:
            content_parts.append(event.text)
        elif event.kind == EventKind.REASONING_DELTA and event.text:
            reasoning_parts.append(event.text)
        elif event.kind == EventKind.TOOL_CALL_COMMIT:
            tool_calls_list.append({
                "id": event.tool_call_id,
                "type": "function",
                "function": {
                    "name": event.tool_name,
                    "arguments": event.tool_arguments,
                },
            })
        elif event.kind == EventKind.DONE:
            finish_reason = {
                StopReason.STOP: "stop",
                StopReason.LENGTH: "length",
                StopReason.TOOL_CALLS: "tool_calls",
                StopReason.ERROR: "stop",
            }.get(event.stop_reason, "stop")

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls_list:
        message["tool_calls"] = tool_calls_list

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


def _event_to_openai_chunks(
    event: SemanticEvent,
    request_id: str,
    created: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Convert a SemanticEvent to zero or more OpenAI SSE chunks."""
    chunks: list[dict[str, Any]] = []

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

def _parse_messages(raw: list[dict]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for m in raw:
        content = m.get("content")
        if isinstance(content, list):
            content = [
                ContentPart(
                    type=p.get("type", "text"),
                    text=p.get("text"),
                    image_url=p.get("image_url"),
                )
                for p in content
            ]

        tool_calls = None
        if "tool_calls" in m and m["tool_calls"]:
            tool_calls = tuple(
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"].get("arguments", "{}"),
                    ),
                )
                for tc in m["tool_calls"]
            )

        messages.append(ChatMessage(
            role=m["role"],
            content=content,
            name=m.get("name"),
            tool_calls=tool_calls,
            tool_call_id=m.get("tool_call_id"),
        ))
    return messages


def _parse_tools(raw: list[dict]) -> list[Tool]:
    tools: list[Tool] = []
    for t in raw:
        fn = t.get("function", {})
        params_raw = fn.get("parameters", {})
        parameters: dict[str, ToolParameter] = {}
        required_fields: list[str] = params_raw.get("required", [])

        for pname, pdef in params_raw.get("properties", {}).items():
            parameters[pname] = ToolParameter(
                name=pname,
                type=pdef.get("type", "string"),
                description=pdef.get("description", ""),
                required=pname in required_fields,
                enum=tuple(pdef["enum"]) if "enum" in pdef else None,
            )

        tools.append(Tool(
            type=t.get("type", "function"),
            function=ToolFunction(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=parameters,
                required=tuple(required_fields),
            ),
        ))
    return tools


def _parse_sampling(body: dict) -> SamplingParams:
    kwargs: dict[str, Any] = {}
    if "temperature" in body:
        kwargs["temperature"] = float(body["temperature"])
    if "top_p" in body:
        kwargs["top_p"] = float(body["top_p"])
    if "max_tokens" in body:
        kwargs["max_tokens"] = int(body["max_tokens"])
    if "max_completion_tokens" in body:
        kwargs["max_tokens"] = int(body["max_completion_tokens"])
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            kwargs["stop"] = (stop,)
        elif isinstance(stop, list):
            kwargs["stop"] = tuple(stop)
    if "frequency_penalty" in body:
        kwargs["frequency_penalty"] = float(body["frequency_penalty"])
    if "presence_penalty" in body:
        kwargs["presence_penalty"] = float(body["presence_penalty"])
    if "seed" in body and body["seed"] is not None:
        kwargs["seed"] = int(body["seed"])
    return SamplingParams(**kwargs)
