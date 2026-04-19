"""Reasoning/thinking round-trip across the OpenAI and Anthropic APIs.

The goal: when the model emits a ``<think>...</think>`` block, both APIs
must (a) surface the reasoning to the client in the expected field, and
(b) accept that same reasoning back on a subsequent turn so the rendered
history matches the bytes the model originally generated.

Naming conventions enforced by these tests:

- OpenAI:  ``message.reasoning_content``  (DeepSeek/Qwen/vLLM convention)
- Anthropic: content block with ``{"type": "thinking", "thinking": ..., "signature": ...}``
             native schema

Clients send reasoning back at the same key on assistant history messages.
"""
from __future__ import annotations

import json

import pytest

from starlette.testclient import TestClient

from egg_toolbox.api.anthropic import _BlockProjector, _parse_messages as anthropic_parse_messages, _synth_signature
from egg_toolbox.api.middleware import create_app
from egg_toolbox.api.openai import _parse_messages as openai_parse_messages
from egg_toolbox.orchestrator import Orchestrator
from egg_toolbox.template import ChatTemplate
from egg_toolbox.types import (
    ChatMessage,
    EventKind,
    SemanticEvent,
    StopReason,
)

from ..conftest import ScriptedBackend


# =================================================================== #
# ChatMessage now carries a ``reasoning`` field                       #
# =================================================================== #

class TestChatMessageReasoningField:
    def test_default_none(self):
        m = ChatMessage(role="assistant", content="hi")
        assert m.reasoning is None

    def test_explicit(self):
        m = ChatMessage(role="assistant", content="hi", reasoning="let me think")
        assert m.reasoning == "let me think"


# =================================================================== #
# OpenAI input: reasoning_content -> ChatMessage.reasoning            #
# =================================================================== #

class TestOpenAIInputParsing:
    def test_reasoning_content_captured_on_assistant(self):
        """``message.reasoning_content`` on assistant history populates
        ChatMessage.reasoning."""
        msgs = openai_parse_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello",
             "reasoning_content": "the user greeted me, so I greet back"},
            {"role": "user", "content": "continue"},
        ])
        assert msgs[1].reasoning == "the user greeted me, so I greet back"
        # Other roles' reasoning is ignored (not meaningful there).
        assert msgs[0].reasoning is None
        assert msgs[2].reasoning is None

    def test_reasoning_alias_accepted(self):
        """Accept ``reasoning`` as a lenient alternative spelling."""
        msgs = openai_parse_messages([
            {"role": "assistant", "content": "hi", "reasoning": "short alias"},
        ])
        assert msgs[0].reasoning == "short alias"

    def test_reasoning_absent_when_user_sends_it(self):
        """Only assistant messages round-trip reasoning -- a user
        message with reasoning_content is ignored."""
        msgs = openai_parse_messages([
            {"role": "user", "content": "hi", "reasoning_content": "user's thoughts"},
        ])
        assert msgs[0].reasoning is None

    def test_no_reasoning_key_no_error(self):
        """Messages without reasoning_content work unchanged."""
        msgs = openai_parse_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert all(m.reasoning is None for m in msgs)


# =================================================================== #
# Anthropic input: {type: thinking} block -> ChatMessage.reasoning    #
# =================================================================== #

class TestAnthropicInputParsing:
    def test_thinking_block_extracted_on_assistant(self):
        msgs = anthropic_parse_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning body", "signature": "sig"},
                {"type": "text", "text": "hello"},
            ]},
            {"role": "user", "content": "continue"},
        ], system=None)
        assert msgs[1].reasoning == "reasoning body"
        assert msgs[1].content == "hello"

    def test_thinking_block_on_user_dropped(self):
        """Thinking blocks only round-trip on assistant messages."""
        msgs = anthropic_parse_messages([
            {"role": "user", "content": [
                {"type": "thinking", "thinking": "user cannot think natively"},
                {"type": "text", "text": "hi"},
            ]},
        ], system=None)
        assert msgs[0].reasoning is None

    def test_multiple_thinking_blocks_concatenated(self):
        """Anthropic allows multiple thinking blocks in one message;
        concatenate them into ChatMessage.reasoning so the round-trip
        is faithful."""
        msgs = anthropic_parse_messages([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "first part. "},
                {"type": "thinking", "thinking": "second part."},
                {"type": "text", "text": "answer"},
            ]},
        ], system=None)
        assert msgs[0].reasoning == "first part. second part."

    def test_thinking_with_tool_use_roundtrips(self):
        """Thinking + tool_use + text should all survive into the
        internal ChatMessage."""
        msgs = anthropic_parse_messages([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should use the tool"},
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_01", "name": "search",
                 "input": {"q": "x"}},
            ]},
        ], system=None)
        msg = msgs[0]
        assert msg.reasoning == "I should use the tool"
        assert msg.content == "let me check"
        assert msg.tool_calls is not None and len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "search"

    def test_redacted_thinking_preserved_as_text(self):
        """Anthropic's ``redacted_thinking`` blocks (opaque) are
        preserved as raw text so the round-trip remains faithful."""
        msgs = anthropic_parse_messages([
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "opaque-blob"},
                {"type": "text", "text": "out"},
            ]},
        ], system=None)
        assert msgs[0].reasoning == "opaque-blob"


# =================================================================== #
# Template rendering: reasoning_content exposed per-message           #
# =================================================================== #

# Template that exercises message.reasoning_content -- mimics what
# Qwen3's native template does with the thinking key.
_REASONING_TEMPLATE = """\
{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
{%- if message.role == 'assistant' and message.reasoning_content -%}
<think>{{ message.reasoning_content }}</think>
{%- endif -%}
{{ message.content or '' }}
<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""


class TestTemplateRendering:
    def test_reasoning_content_appears_in_rendered_output(self):
        """A ChatMessage with .reasoning renders as ``<think>...</think>``
        bytes in the output, via the template's reasoning_content key."""
        t = ChatTemplate(_REASONING_TEMPLATE)
        rendered = t.render([
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello",
                        reasoning="I'm thinking about how to greet"),
        ])
        assert "<think>I'm thinking about how to greet</think>hello" in rendered

    def test_no_reasoning_no_think_block(self):
        """An assistant message without reasoning renders without a
        <think> block."""
        t = ChatTemplate(_REASONING_TEMPLATE)
        rendered = t.render([
            ChatMessage(role="assistant", content="hello"),
        ])
        assert "<think>" not in rendered

    def test_reasoning_passthrough_even_if_template_ignores(self):
        """A template that doesn't reference reasoning_content must
        not error when messages carry it -- the dict key is just
        unused."""
        plain_template = """\
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{% endfor -%}
"""
        t = ChatTemplate(plain_template)
        rendered = t.render([
            ChatMessage(role="assistant", content="hello", reasoning="ignored"),
        ])
        assert "hello" in rendered
        assert "ignored" not in rendered  # template didn't use it




# =================================================================== #
# Anthropic streaming: signature_delta emitted before block_stop      #
# =================================================================== #

class TestAnthropicStreamingSignature:
    def test_signature_emitted_on_thinking_close(self):
        """A thinking block close sequence must be:
        content_block_delta (thinking_delta)...,
        content_block_delta (signature_delta),
        content_block_stop.
        """
        p = _BlockProjector()
        events = []
        events.extend(p.feed(SemanticEvent(
            kind=EventKind.REASONING_DELTA, text="let me ")))
        events.extend(p.feed(SemanticEvent(
            kind=EventKind.REASONING_DELTA, text="think")))
        # Start a text block, which closes the thinking block.
        events.extend(p.feed(SemanticEvent(
            kind=EventKind.CONTENT_DELTA, text="answer")))

        # Extract just the thinking-related events.
        thinking_deltas = [
            e for e in events
            if e[0] == "content_block_delta"
            and e[1]["delta"].get("type") == "thinking_delta"
        ]
        sig_deltas = [
            e for e in events
            if e[0] == "content_block_delta"
            and e[1]["delta"].get("type") == "signature_delta"
        ]
        block_stops = [e for e in events if e[0] == "content_block_stop"]

        assert len(thinking_deltas) == 2
        assert len(sig_deltas) == 1
        # Signature deterministic over concatenated thinking text.
        assert sig_deltas[0][1]["delta"]["signature"] == _synth_signature(
            "let me think"
        )
        # Signature must come before block_stop.
        sig_idx = events.index(sig_deltas[0])
        stop_idx = events.index(block_stops[0])
        assert sig_idx < stop_idx

    def test_no_thinking_no_signature(self):
        """If no reasoning was emitted, no signature_delta fires."""
        p = _BlockProjector()
        events = []
        events.extend(p.feed(SemanticEvent(
            kind=EventKind.CONTENT_DELTA, text="hello")))
        events.extend(p.close())
        sig_deltas = [
            e for e in events
            if e[0] == "content_block_delta"
            and e[1]["delta"].get("type") == "signature_delta"
        ]
        assert sig_deltas == []

    def test_signature_stable_across_identical_text(self):
        """The synthesised signature is a function of the thinking
        text only -- two identical bodies produce identical sigs."""
        assert _synth_signature("abc") == _synth_signature("abc")
        assert _synth_signature("abc") != _synth_signature("abd")


# =================================================================== #
# End-to-end: OpenAI API surface                                       #
# =================================================================== #

_SCRIPTED_WITH_THINK = "<think>okay let me think</think>final answer"


# Template that both (a) contains ``<tool_call>`` so the Hermes tool
# detector activates and (b) references ``reasoning_content`` so the
# reasoning detector sets ``ReasoningMode.TAG_BASED``.  Otherwise the
# streaming parser treats ``<think>...</think>`` as plain content and
# the reasoning never separates.
_HERMES_WITH_REASONING_TEMPLATE = """\
{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
{%- if message.role == 'assistant' and message.reasoning_content -%}
<think>{{ message.reasoning_content }}</think>
{%- endif -%}
{{ message.get('content', '') or '' }}
<|im_end|>
{% endfor -%}
{%- if tools is defined and tools -%}
Tools: {{ tools | tojson }}
Use <tool_call>{"name":"fn","arguments":{...}}</tool_call> to call tools.
{% endif -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""


def _make_reasoning_client(output: str) -> TestClient:
    """TestClient whose backend uses a reasoning-aware Hermes template.
    Needed for E2E tests because the default HERMES_TEMPLATE in conftest
    has no ``reasoning_content`` keyword, so the detector falls back to
    ``ReasoningMode.NONE`` and <think>…</think> is parsed as content."""
    backend = ScriptedBackend(output, template=_HERMES_WITH_REASONING_TEMPLATE)
    orch = Orchestrator(backend)
    return TestClient(create_app(orch))


class TestOpenAIEndToEnd:
    def test_non_streaming_response_has_reasoning_content(self):
        """Scripted output ``<think>X</think>Y`` must produce
        ``message.reasoning_content=X`` and ``message.content=Y``."""
        client = _make_reasoning_client(_SCRIPTED_WITH_THINK)
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        msg = resp.json()["choices"][0]["message"]
        # Exact check: reasoning_content is the field name we
        # committed to, not "reasoning".
        assert "reasoning_content" in msg
        assert msg["reasoning_content"] == "okay let me think"
        assert msg["content"] == "final answer"

    def test_streaming_delta_uses_reasoning_content(self):
        client = _make_reasoning_client(_SCRIPTED_WITH_THINK)
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        reasoning_pieces = []
        content_pieces = []
        for line in resp.text.split("\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                payload = json.loads(line[len("data: "):])
                for ch in payload.get("choices", []):
                    delta = ch.get("delta", {})
                    if "reasoning_content" in delta:
                        reasoning_pieces.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"]:
                        content_pieces.append(delta["content"])
        assert "".join(reasoning_pieces) == "okay let me think"
        assert "".join(content_pieces) == "final answer"


# =================================================================== #
# End-to-end: Anthropic API surface                                    #
# =================================================================== #

class TestAnthropicEndToEnd:
    def test_non_streaming_emits_thinking_block_with_signature(self):
        client = _make_reasoning_client(_SCRIPTED_WITH_THINK)
        resp = client.post("/v1/messages", json={
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        content = body["content"]
        thinking = [b for b in content if b["type"] == "thinking"]
        text = [b for b in content if b["type"] == "text"]
        assert len(thinking) == 1
        assert thinking[0]["thinking"] == "okay let me think"
        # Signature present and deterministic.
        assert thinking[0]["signature"] == _synth_signature("okay let me think")
        assert text[0]["text"] == "final answer"
        # Thinking block must precede text per Anthropic's ordering.
        assert content.index(thinking[0]) < content.index(text[0])

    def test_streaming_emits_thinking_delta_and_signature_delta(self):
        client = _make_reasoning_client(_SCRIPTED_WITH_THINK)
        resp = client.post("/v1/messages", json={
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        events = []
        current_event = None
        for line in resp.text.split("\n"):
            line = line.rstrip()
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: ") and current_event is not None:
                events.append((current_event, json.loads(line[len("data: "):])))

        thinking_deltas = [
            e for e in events
            if e[0] == "content_block_delta"
            and e[1].get("delta", {}).get("type") == "thinking_delta"
        ]
        sig_deltas = [
            e for e in events
            if e[0] == "content_block_delta"
            and e[1].get("delta", {}).get("type") == "signature_delta"
        ]
        assert "".join(d[1]["delta"]["thinking"] for d in thinking_deltas) == (
            "okay let me think"
        )
        assert len(sig_deltas) == 1
        assert sig_deltas[0][1]["delta"]["signature"] == _synth_signature(
            "okay let me think"
        )
