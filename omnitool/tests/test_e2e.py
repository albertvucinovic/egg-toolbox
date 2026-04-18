"""End-to-end tests for the omnitool pipeline.

Scripted tests validate API -> Orchestrator -> Parser -> SSE using a
deterministic fake backend (no model needed, fast).

Real-model tests download a small GGUF and run through tinygrad
(skipped when tinygrad is unavailable).
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}]


def _chat_body(content: str, *, stream: bool = True, tools=None, **extra) -> dict:
    body: dict = {
        "model": "test",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        **extra,
    }
    if tools:
        body["tools"] = tools
    return body


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE response text into a list of JSON chunk dicts."""
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


# ===================================================================
# Scripted-backend tests (fast, deterministic, no external deps)
# ===================================================================

class TestContentOnly:
    """Content-only requests (no tool calls)."""

    def test_streaming_content(self, make_client):
        client = make_client("Hello, how can I help you?")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=True),
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)
        assert len(chunks) > 1

        # First chunk carries the role
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Reassemble content
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert "Hello" in content

        # Final chunk has finish_reason=stop
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

        # Stream ends with [DONE]
        assert resp.text.rstrip().endswith("data: [DONE]")

    def test_non_streaming_content(self, make_client):
        client = make_client("Hello, how can I help you?")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=False),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "Hello" in msg["content"]
        assert body["choices"][0]["finish_reason"] == "stop"


class TestToolCalling:
    """Tool-calling requests through the full pipeline."""

    TOOL_OUTPUT = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"location": "Paris"}}</tool_call>'
    )

    def test_streaming_tool_call(self, make_client):
        client = make_client(self.TOOL_OUTPUT)
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Weather in Paris?", stream=True, tools=WEATHER_TOOL),
        )
        assert resp.status_code == 200
        chunks = _parse_sse(resp.text)

        # Must contain tool_call chunks
        tc_chunks = [
            c for c in chunks
            if "tool_calls" in c["choices"][0].get("delta", {})
        ]
        assert len(tc_chunks) > 0

        # Tool name must appear
        names = [
            c["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            for c in tc_chunks
            if c["choices"][0]["delta"]["tool_calls"][0].get("function", {}).get("name")
        ]
        assert "get_weather" in names

        # finish_reason = tool_calls
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_non_streaming_tool_call(self, make_client):
        client = make_client(self.TOOL_OUTPUT)
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Weather in Paris?", stream=False, tools=WEATHER_TOOL),
        )
        assert resp.status_code == 200
        body = resp.json()

        msg = body["choices"][0]["message"]
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args["location"] == "Paris"
        assert body["choices"][0]["finish_reason"] == "tool_calls"


class TestSSEFormat:
    """Validate Server-Sent Events format correctness."""

    def test_every_line_is_data(self, make_client):
        client = make_client("Test output.")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Test", stream=True),
        )
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line:
                assert line.startswith("data: "), f"Bad SSE line: {line!r}"

    def test_chunk_json_structure(self, make_client):
        client = make_client("OK")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Test", stream=True),
        )
        for chunk in _parse_sse(resp.text):
            assert chunk["id"].startswith("chatcmpl-")
            assert chunk["object"] == "chat.completion.chunk"
            assert isinstance(chunk["created"], int)
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]

    def test_done_sentinel(self, make_client):
        client = make_client("Done.")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Test", stream=True),
        )
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"


class TestAuxEndpoints:
    def test_health(self, make_client):
        client = make_client("")
        assert client.get("/health").json() == {"status": "ok"}

    def test_list_models(self, make_client):
        client = make_client("")
        body = client.get("/v1/models").json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "scripted-test"


# ===================================================================
# Real model tests (tinygrad + GGUF download, skipped if unavailable)
# ===================================================================

try:
    from tinygrad.apps.llm import Transformer, SimpleTokenizer  # noqa: F401
    HAS_TINYGRAD_LLM = True
except (ImportError, ModuleNotFoundError):
    HAS_TINYGRAD_LLM = False

requires_tinygrad = pytest.mark.skipif(
    not HAS_TINYGRAD_LLM,
    reason="tinygrad LLM module not available (needs tinygrad with nn.llm support)",
)


@requires_tinygrad
class TestRealModel:
    """E2E tests with a real GGUF model via tinygrad."""

    @staticmethod
    def _build_client(model_path):
        from omnitool.backends.tinygrad import TinygradBackend
        from omnitool.orchestrator import Orchestrator
        from omnitool.api.middleware import create_app
        from starlette.testclient import TestClient

        backend = TinygradBackend()
        backend.load_model(str(model_path))
        orch = Orchestrator(backend)
        return TestClient(create_app(orch))

    def test_content_generation(self, gguf_model_path):
        client = self._build_client(gguf_model_path)
        resp = client.post("/v1/chat/completions", json=_chat_body(
            "Say hello.", stream=False, max_tokens=32, temperature=0.0,
        ))
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_streaming_format(self, gguf_model_path):
        client = self._build_client(gguf_model_path)
        resp = client.post("/v1/chat/completions", json=_chat_body(
            "Say hi.", stream=True, max_tokens=16, temperature=0.0,
        ))
        assert resp.status_code == 200
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        assert lines[-1] == "data: [DONE]"
        for line in lines:
            assert line.startswith("data: ")

    def test_tool_request_no_crash(self, gguf_model_path):
        client = self._build_client(gguf_model_path)
        resp = client.post("/v1/chat/completions", json=_chat_body(
            "What is the weather in Paris?",
            stream=False, max_tokens=64, temperature=0.0,
            tools=WEATHER_TOOL,
        ))
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] in ("stop", "tool_calls", "length")
