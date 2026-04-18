"""Round-trip tests: assistant tool_calls and role=tool results
must reach the LLM correctly formatted."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from egg_toolbox.template import ChatTemplate, _parse_tool_arguments
from egg_toolbox.types import ChatMessage, ToolCall, ToolCallFunction


_QWEN_GGUF = Path(
    "/home/albert/Private/Projekti/ai/omnitool/egg-toolbox/tests/fixtures/models/"
    "qwen2.5-0.5b-instruct-q4_0.gguf"
)


@pytest.fixture(scope="module")
def qwen_template() -> ChatTemplate:
    if not _QWEN_GGUF.exists():
        pytest.skip("Qwen2.5-0.5B GGUF fixture not present")
    return ChatTemplate.from_gguf(_QWEN_GGUF)


def _conversation() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="What is the weather in Zagreb?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_0", type="function",
                         function=ToolCallFunction(
                             name="get_weather",
                             arguments='{"city": "Zagreb"}')),
            ),
        ),
        ChatMessage(role="tool", content="sunny, 22 C", tool_call_id="call_0"),
    ]


class TestPriorToolCallIsObjectNotString:
    """Regression: OpenAI-compat 'arguments' is a JSON string but
    chat templates invariably expect an object.  _msg_to_dict must
    parse the string so the rendered prompt contains a real JSON
    object, not a JSON-encoded string."""

    def test_parse_helper_json_object(self):
        assert _parse_tool_arguments('{"a": 1}') == {"a": 1}

    def test_parse_helper_passes_through_dict(self):
        assert _parse_tool_arguments({"a": 1}) == {"a": 1}

    def test_parse_helper_falls_back_on_malformed(self):
        assert _parse_tool_arguments("{not json") == "{not json"

    def test_parse_helper_none(self):
        assert _parse_tool_arguments(None) is None

    def test_qwen_template_renders_arguments_as_object(self, qwen_template):
        prompt = qwen_template.render(
            messages=_conversation(),
            tools=None,
            add_generation_prompt=True,
        )
        # The bad render looks like:  "arguments": "{\"city\": \"Zagreb\"}"
        # The good render looks like: "arguments": {"city": "Zagreb"}
        assert '"arguments": {"city": "Zagreb"}' in prompt
        assert '"arguments": "{\\"city\\"' not in prompt


class TestToolResultReachesPrompt:
    def test_tool_result_wrapped_in_response_block(self, qwen_template):
        prompt = qwen_template.render(
            messages=_conversation(),
            tools=None,
            add_generation_prompt=True,
        )
        # Qwen2.5 wraps role=tool content inside a <tool_response> block
        # attached to a user turn; we only care that the result text
        # lands somewhere the model will see it.
        assert "sunny, 22 C" in prompt
        assert "<tool_response>" in prompt
        assert "</tool_response>" in prompt

    def test_assistant_turn_preserves_tool_call(self, qwen_template):
        prompt = qwen_template.render(
            messages=_conversation(),
            tools=None,
            add_generation_prompt=True,
        )
        assert "<tool_call>" in prompt
        assert "get_weather" in prompt

    def test_generation_prompt_at_end(self, qwen_template):
        prompt = qwen_template.render(
            messages=_conversation(),
            tools=None,
            add_generation_prompt=True,
        )
        # Model should now be prompted to speak as assistant, AFTER
        # seeing the tool result.
        assert prompt.rstrip().endswith("<|im_start|>assistant")


class TestAssistantWithOnlyToolCallsRenders:
    """Regression: some chat templates (notably Qwen3's) do a bare
    ``message.content`` attribute access which Jinja translates to
    dict-key lookup.  If the key is absent (because our assistant
    message carries only tool_calls), the template raises
    UndefinedError -- the HTTP handler catches it as a 500.

    _msg_to_dict must always include a 'content' key, substituting
    an empty string when ChatMessage.content is None."""

    def test_no_undefined_error_on_tool_call_only_assistant(self, qwen_template):
        from egg_toolbox.types import ChatMessage, ToolCall, ToolCallFunction
        msgs = [
            ChatMessage(role="user", content="Weather?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(
                    ToolCall(id="call_0", type="function",
                             function=ToolCallFunction(
                                 name="get_weather",
                                 arguments='{"city": "Zagreb"}')),
                ),
            ),
            ChatMessage(role="tool", content="sunny", tool_call_id="call_0"),
        ]
        # Must not raise.
        prompt = qwen_template.render(messages=msgs, tools=None,
                                      add_generation_prompt=True)
        # Tool call is rendered despite empty content.
        assert "get_weather" in prompt
        # Tool result is rendered.
        assert "sunny" in prompt

    def test_msg_to_dict_always_includes_content_key(self):
        from egg_toolbox.template import ChatTemplate
        from egg_toolbox.types import ChatMessage
        d = ChatTemplate._msg_to_dict(ChatMessage(role="assistant", content=None))
        assert "content" in d
        assert d["content"] == ""


class TestMultipleToolResultsInOrder:
    """Parallel tool calls followed by their paired results."""

    def test_two_tools_two_results(self, qwen_template):
        msgs = [
            ChatMessage(role="user", content="Weather in Zagreb and Paris."),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(
                    ToolCall(id="call_0", type="function",
                             function=ToolCallFunction(
                                 name="get_weather",
                                 arguments='{"city": "Zagreb"}')),
                    ToolCall(id="call_1", type="function",
                             function=ToolCallFunction(
                                 name="get_weather",
                                 arguments='{"city": "Paris"}')),
                ),
            ),
            ChatMessage(role="tool", content="sunny 22C", tool_call_id="call_0"),
            ChatMessage(role="tool", content="cloudy 15C", tool_call_id="call_1"),
        ]
        prompt = qwen_template.render(messages=msgs, tools=None, add_generation_prompt=True)
        # Both results present, in the right order.
        zagreb_pos = prompt.find("sunny 22C")
        paris_pos = prompt.find("cloudy 15C")
        assert zagreb_pos >= 0 and paris_pos >= 0
        assert zagreb_pos < paris_pos
        # Both original calls present.
        assert prompt.count('"name": "get_weather"') == 2
