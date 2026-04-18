"""End-to-end tests for the Anthropic /v1/messages projection.

Uses the ScriptedBackend from conftest so we can pin the model's
output tokens and verify SemanticEvent -> Anthropic-SSE projection.
"""
from __future__ import annotations

import json

import pytest


TOOL_OUTPUT = (
    '<tool_call>{"name": "get_weather", '
    '"arguments": {"location": "Paris"}}</tool_call>'
)

WEATHER_TOOL = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}]


def _anthropic_body(content: str, *, stream: bool = False, tools=None,
                   system=None, max_tokens: int = 100, **extra) -> dict:
    body: dict = {
        "model": "test",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        **extra,
    }
    if tools:
        body["tools"] = tools
    if system is not None:
        body["system"] = system
    return body


def _parse_sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse Anthropic-style SSE into (event_name, payload) tuples."""
    out = []
    current_event = None
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            current_event = None
            continue
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: ") and current_event is not None:
            payload = json.loads(line[len("data: "):])
            out.append((current_event, payload))
    return out


# =================================================================== #
# Request dispatch / validation                                       #
# =================================================================== #

class TestRequestValidation:
    def test_missing_messages(self, make_client):
        client = make_client("ignored")
        resp = client.post("/v1/messages", json={"model": "x", "max_tokens": 10})
        assert resp.status_code == 400
        assert "messages" in resp.json()["error"]["message"]

    def test_missing_max_tokens(self, make_client):
        client = make_client("ignored")
        resp = client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
        assert "max_tokens" in resp.json()["error"]["message"]

    def test_invalid_json(self, make_client):
        client = make_client("ignored")
        resp = client.post(
            "/v1/messages",
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400


# =================================================================== #
# Non-streaming                                                       #
# =================================================================== #

class TestNonStreamingContent:
    def test_plain_text_response(self, make_client):
        client = make_client("Hello, world!")
        resp = client.post("/v1/messages", json=_anthropic_body("Hi"))
        assert resp.status_code == 200
        body = resp.json()

        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["stop_reason"] == "end_turn"
        assert len(body["content"]) == 1
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "Hello, world!"

    def test_usage_populated(self, make_client):
        client = make_client("Hi")
        resp = client.post("/v1/messages", json=_anthropic_body("Hi"))
        body = resp.json()
        assert "usage" in body
        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0

    def test_empty_content_still_returns_one_block(self, make_client):
        client = make_client("")
        resp = client.post("/v1/messages", json=_anthropic_body("Hi"))
        body = resp.json()
        assert len(body["content"]) == 1
        assert body["content"][0]["type"] == "text"


class TestNonStreamingToolCall:
    def test_tool_use_block(self, make_client):
        client = make_client(TOOL_OUTPUT)
        resp = client.post("/v1/messages", json=_anthropic_body(
            "Weather in Paris?", tools=WEATHER_TOOL,
        ))
        assert resp.status_code == 200
        body = resp.json()

        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        tu = tool_blocks[0]
        assert tu["name"] == "get_weather"
        # Anthropic sends input as a JSON object (not a stringified JSON).
        assert tu["input"] == {"location": "Paris"}
        assert tu["id"].startswith("toolu_")
        assert body["stop_reason"] == "tool_use"


# =================================================================== #
# Streaming                                                           #
# =================================================================== #

class TestStreamingContent:
    def test_event_sequence(self, make_client):
        client = make_client("Hi.")
        resp = client.post(
            "/v1/messages",
            json=_anthropic_body("Hello", stream=True),
        )
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        event_names = [e for e, _ in events]

        assert event_names[0] == "message_start"
        assert event_names[-1] == "message_stop"
        assert "message_delta" in event_names
        assert "content_block_start" in event_names
        assert "content_block_delta" in event_names
        assert "content_block_stop" in event_names

    def test_text_deltas_reconstruct_output(self, make_client):
        client = make_client("Hello!")
        resp = client.post(
            "/v1/messages",
            json=_anthropic_body("Hi", stream=True),
        )
        events = _parse_sse_events(resp.text)
        pieces = [
            p["delta"]["text"]
            for name, p in events
            if name == "content_block_delta" and p["delta"].get("type") == "text_delta"
        ]
        assert "".join(pieces) == "Hello!"

    def test_message_delta_has_stop_reason(self, make_client):
        client = make_client("Hi.")
        resp = client.post(
            "/v1/messages",
            json=_anthropic_body("Hi", stream=True),
        )
        events = _parse_sse_events(resp.text)
        message_deltas = [p for name, p in events if name == "message_delta"]
        assert message_deltas
        assert message_deltas[0]["delta"]["stop_reason"] == "end_turn"


class TestStreamingToolCall:
    def test_tool_use_block_start_delta_stop(self, make_client):
        client = make_client(TOOL_OUTPUT)
        resp = client.post(
            "/v1/messages",
            json=_anthropic_body("Weather?", stream=True, tools=WEATHER_TOOL),
        )
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        # content_block_start for tool_use must carry name + empty input
        tool_starts = [
            p for name, p in events
            if name == "content_block_start"
            and p["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "get_weather"
        assert tool_starts[0]["content_block"]["input"] == {}

        # Args streamed via input_json_delta
        arg_deltas = [
            p["delta"]["partial_json"]
            for name, p in events
            if name == "content_block_delta"
            and p["delta"].get("type") == "input_json_delta"
        ]
        assert arg_deltas
        joined = "".join(arg_deltas)
        # Should be parseable JSON for the arguments.
        assert json.loads(joined) == {"location": "Paris"}

        # Block stop fires for the tool_use block
        stop_events = [
            p for name, p in events
            if name == "content_block_stop"
        ]
        assert any(p["index"] == tool_starts[0]["index"] for p in stop_events)

        # message_delta should report tool_use as stop reason.
        message_deltas = [p for name, p in events if name == "message_delta"]
        assert message_deltas[0]["delta"]["stop_reason"] == "tool_use"


# =================================================================== #
# Input parsing: Anthropic-specific shapes                            #
# =================================================================== #

class TestInputShapes:
    def test_system_prompt_accepted(self, make_client):
        client = make_client("ok")
        resp = client.post("/v1/messages", json=_anthropic_body(
            "Hi", system="You are concise.",
        ))
        assert resp.status_code == 200

    def test_system_prompt_as_blocks(self, make_client):
        client = make_client("ok")
        resp = client.post("/v1/messages", json=_anthropic_body(
            "Hi",
            system=[{"type": "text", "text": "You are concise."}],
        ))
        assert resp.status_code == 200

    def test_tool_result_in_user_message(self, make_client):
        client = make_client("Thanks for the weather.")
        resp = client.post("/v1/messages", json={
            "model": "test",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01",
                     "name": "get_weather",
                     "input": {"location": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01",
                     "content": "sunny 22C"},
                ]},
            ],
        })
        assert resp.status_code == 200
