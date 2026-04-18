"""JSON mode / structured output: OpenAI ``response_format`` support.

The client sends ``response_format={"type": "json_object"}`` or
``response_format={"type": "json_schema", "json_schema": {...}}``
to constrain the model's output to valid JSON.  egg-toolbox
translates this to a GBNF grammar attached to the CompiledRequest.

Without a grammar-aware backend plugged in (llama-cpp-python, future
tinygrad sampler) the grammar is effectively a hint -- the backend
ignores it and returns a best-effort completion.  These tests pin the
API contract: the request is accepted, the grammar is generated, the
CompiledRequest carries it.  Backend-level enforcement is a Phase 3
concern.
"""
from __future__ import annotations

import pytest


class _CapturingBackend:
    """StepBackend stand-in that records the CompiledRequest it's given
    and emits a scripted short response.  Lets us peek at what the
    orchestrator built without needing a real model."""

    def __init__(self, tokens: list[int] | None = None):
        self._tokens = tokens or [10, 11, 12]
        self.last_request = None

    def tokenizer(self):
        return _CapturingTokenizer()

    def chat_template_source(self) -> str:
        # Minimal pass-through template: just concatenate message
        # contents.  No tool markers, no reasoning markers.
        return "{% for m in messages %}{{ m.content or '' }}{% endfor %}"

    def model_name(self) -> str:
        return "fake-model"

    def generate_tokens(self, request):
        self.last_request = request
        for t in self._tokens:
            yield t


class _CapturingTokenizer:
    eos_token_id = 0
    bos_token_id = None
    vocab_size = 100

    def encode(self, text: str) -> list[int]:
        # One token per char for simplicity.
        return [ord(c) & 0xFF for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def decode_single(self, token_id: int) -> str:
        if token_id == 0:
            return ""
        return chr(token_id)


def _make_orchestrator():
    """Orchestrator wired to the capturing fake backend."""
    from egg_toolbox.orchestrator import Orchestrator
    from egg_toolbox.backends.base import StepBackend

    backend = _CapturingBackend()
    # Register the backend with the StepBackend virtual-subclass so
    # isinstance() picks the right orchestrator branch.
    StepBackend.register(_CapturingBackend)
    orch = Orchestrator(backend)
    return orch, backend


@pytest.mark.asyncio
async def test_json_object_mode_generates_grammar():
    orch, backend = _make_orchestrator()
    from egg_toolbox.types import ChatMessage

    messages = [ChatMessage(role="user", content="hi")]
    async for _ in orch.chat_completion(
        messages,
        response_format={"type": "json_object"},
    ):
        pass

    assert backend.last_request is not None
    assert backend.last_request.grammar is not None, (
        "response_format=json_object must populate CompiledRequest.grammar"
    )
    # The grammar should start the response as JSON (object or array).
    assert "json-object" in backend.last_request.grammar
    assert "json-array" in backend.last_request.grammar


@pytest.mark.asyncio
async def test_json_schema_mode_generates_schema_grammar():
    orch, backend = _make_orchestrator()
    from egg_toolbox.types import ChatMessage

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    messages = [ChatMessage(role="user", content="hi")]
    async for _ in orch.chat_completion(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "person"},
        },
    ):
        pass

    assert backend.last_request.grammar is not None
    # Grammar must reference the schema's property names.
    assert '"name"' in backend.last_request.grammar or '\\"name\\"' in backend.last_request.grammar
    assert '"age"' in backend.last_request.grammar or '\\"age\\"' in backend.last_request.grammar
    # json_schema field on the request is populated too (for
    # structured_outputs backends like vLLM / SGLang).
    assert backend.last_request.json_schema == schema


@pytest.mark.asyncio
async def test_text_mode_produces_no_grammar():
    """response_format={'type':'text'} is the default / no-op case."""
    orch, backend = _make_orchestrator()
    from egg_toolbox.types import ChatMessage

    messages = [ChatMessage(role="user", content="hi")]
    async for _ in orch.chat_completion(
        messages,
        response_format={"type": "text"},
    ):
        pass

    assert backend.last_request.grammar is None


@pytest.mark.asyncio
async def test_no_response_format_produces_no_grammar():
    orch, backend = _make_orchestrator()
    from egg_toolbox.types import ChatMessage

    messages = [ChatMessage(role="user", content="hi")]
    async for _ in orch.chat_completion(messages):
        pass

    assert backend.last_request.grammar is None


# ---------- API-level parsing ----------

def test_openai_api_accepts_response_format():
    """POST /v1/chat/completions must pass response_format through to
    the orchestrator without rejecting the request."""
    from starlette.testclient import TestClient
    from egg_toolbox.api.middleware import create_app
    from egg_toolbox.orchestrator import Orchestrator
    from egg_toolbox.backends.base import StepBackend

    backend = _CapturingBackend()
    StepBackend.register(_CapturingBackend)
    orch = Orchestrator(backend)
    client = TestClient(create_app(orch))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "say hi as json"}],
            "stream": False,
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200, resp.text
    # The orchestrator captured the request; grammar must be set.
    assert backend.last_request is not None
    assert backend.last_request.grammar is not None


def test_openai_api_response_format_json_schema():
    from starlette.testclient import TestClient
    from egg_toolbox.api.middleware import create_app
    from egg_toolbox.orchestrator import Orchestrator
    from egg_toolbox.backends.base import StepBackend

    backend = _CapturingBackend()
    StepBackend.register(_CapturingBackend)
    orch = Orchestrator(backend)
    client = TestClient(create_app(orch))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "go"}],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                    },
                },
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert backend.last_request.grammar is not None
    assert backend.last_request.json_schema == {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
    }
