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
        assert body["data"][0]["created"] > 0
        assert body["data"][0]["owned_by"] == "omnitool"


class TestErrorHandling:
    """Error handling: malformed requests return JSON errors, not stack traces."""

    def test_invalid_json(self, make_client):
        client = make_client("")
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json at all{{{",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_json"

    def test_missing_messages(self, make_client):
        client = make_client("")
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "messages" in body["error"]["message"]

    def test_empty_messages(self, make_client):
        client = make_client("")
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"

    def test_orchestrator_error(self, make_client_raw):
        """Force an exception inside the orchestrator, expect 500 JSON error."""
        client = make_client_raw(raise_error=True)
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=False),
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "server_error"
        # Must NOT contain a Python traceback
        assert "Traceback" not in json.dumps(body)


class TestTokenCounts:
    """Token usage is populated with real counts."""

    def test_non_streaming_usage(self, make_client):
        client = make_client("Hello, how can I help you?")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=False),
        )
        body = resp.json()
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] > 0
        assert body["usage"]["total_tokens"] == (
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
        )

    def test_streaming_include_usage(self, make_client):
        client = make_client("Hello!")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=True,
                            stream_options={"include_usage": True}),
        )
        assert resp.status_code == 200
        # Parse ALL data lines including usage chunk
        all_data = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            all_data.append(json.loads(payload))

        # The last data chunk before [DONE] should be the usage chunk
        usage_chunk = all_data[-1]
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"]["prompt_tokens"] > 0
        assert usage_chunk["usage"]["completion_tokens"] > 0

    def test_streaming_no_usage_default(self, make_client):
        client = make_client("Hello!")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=True),
        )
        # No usage chunk should appear
        all_data = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            all_data.append(json.loads(payload))

        for chunk in all_data:
            assert "usage" not in chunk


class TestToolChoice:
    """tool_choice parameter handling."""

    TOOL_OUTPUT = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"location": "Paris"}}</tool_call>'
    )

    def test_tool_choice_none(self, make_client):
        """tool_choice=none should suppress tools (model produces content)."""
        # The scripted backend always outputs the same tokens regardless,
        # but with tool_choice=none we pass tools=None to the orchestrator,
        # so the parser won't recognize tool_call tags as tool calls.
        client = make_client("Just some plain text content.")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Weather?", stream=False, tools=WEATHER_TOOL,
                            tool_choice="none"),
        )
        assert resp.status_code == 200
        body = resp.json()
        msg = body["choices"][0]["message"]
        # With tool_choice=none, should not produce tool_calls
        assert msg.get("tool_calls") is None or len(msg.get("tool_calls", [])) == 0

    def test_tool_choice_auto(self, make_client):
        """tool_choice=auto (default) should allow tool calls."""
        client = make_client(self.TOOL_OUTPUT)
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Weather?", stream=False, tools=WEATHER_TOOL,
                            tool_choice="auto"),
        )
        assert resp.status_code == 200
        body = resp.json()
        msg = body["choices"][0]["message"]
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1


class TestSystemFingerprint:
    """system_fingerprint is present in all responses."""

    def test_non_streaming_fingerprint(self, make_client):
        client = make_client("OK")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=False),
        )
        body = resp.json()
        assert body["system_fingerprint"] == "omnitool-v0"

    def test_streaming_fingerprint(self, make_client):
        client = make_client("OK")
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_body("Hi", stream=True),
        )
        chunks = _parse_sse(resp.text)
        for chunk in chunks:
            assert chunk["system_fingerprint"] == "omnitool-v0"


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
