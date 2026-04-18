"""Tests for egg_toolbox.grammar -- GBNF grammar generation from
tool definitions.

These are structural/shape tests: we assert the grammar contains the
rules and literals we expect, not that it round-trips through a real
llama.cpp grammar parser (that would require an integration backend).
The generator's job is to produce a string; validation against an
actual engine is Phase 3 work.
"""
from __future__ import annotations

import pytest

from egg_toolbox.grammar import (
    generate_gbnf,
    _json_schema_to_gbnf,
    _hermes_gbnf,
    _generic_json_gbnf,
)
from egg_toolbox.types import (
    FormatAnalysis, ReasoningMode, Tool, ToolFormatMode, ToolFunction,
    ToolParameter,
)


def _analysis_hermes() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.HERMES,
        reasoning_mode=ReasoningMode.NONE,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        name_field="name",
        args_field="arguments",
    )


def _analysis_generic() -> FormatAnalysis:
    return FormatAnalysis(
        tool_mode=ToolFormatMode.GENERIC_JSON,
        reasoning_mode=ReasoningMode.NONE,
    )


def _tool(
    name: str,
    parameters: dict[str, ToolParameter] | None = None,
    required: tuple[str, ...] = (),
) -> Tool:
    return Tool(
        type="function",
        function=ToolFunction(
            name=name,
            description="",
            parameters=parameters or {},
            required=required,
        ),
    )


def _weather_tool() -> Tool:
    return _tool(
        "get_weather",
        parameters={
            "city": ToolParameter(name="city", type="string", required=True),
            "unit": ToolParameter(
                name="unit", type="string", required=False,
                enum=("celsius", "fahrenheit"),
            ),
        },
        required=("city",),
    )


# ---------- Primitives ----------

def test_json_primitives_present():
    schema = {"type": "string"}
    rules = _json_schema_to_gbnf(schema, "my-str")
    joined = "\n".join(rules)
    assert "my-str" in joined
    # Must reference or inline a JSON string production.
    assert "string" in joined or '"\\""' in joined


def test_schema_integer_and_number():
    """Integer and number schemas must resolve to distinct primitive
    rules -- either inlined or referenced."""
    int_rules = _json_schema_to_gbnf({"type": "integer"}, "r-int")
    num_rules = _json_schema_to_gbnf({"type": "number"}, "r-num")
    int_joined = "\n".join(int_rules)
    num_joined = "\n".join(num_rules)
    assert "r-int" in int_joined
    assert "r-num" in num_joined
    # The two must not be literally identical -- number permits decimals.
    assert int_joined != num_joined


def test_schema_boolean():
    """Boolean schema must resolve to a rule that accepts either
    literal 'true' or 'false' (inline or via reference)."""
    rules = _json_schema_to_gbnf({"type": "boolean"}, "r-bool")
    joined = "\n".join(rules)
    # Either the target rule inlines them or references json-bool.
    assert ("true" in joined and "false" in joined) or "json-bool" in joined


def test_schema_enum_emits_literal_alternatives():
    rules = _json_schema_to_gbnf(
        {"type": "string", "enum": ["red", "green", "blue"]},
        "r-color",
    )
    joined = "\n".join(rules)
    assert '"red"' in joined or '\\"red\\"' in joined
    assert '"green"' in joined or '\\"green\\"' in joined
    assert '"blue"' in joined or '\\"blue\\"' in joined


# ---------- Objects ----------

def test_schema_object_with_properties():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["city"],
    }
    rules = _json_schema_to_gbnf(schema, "args")
    joined = "\n".join(rules)
    # Must produce rules that reference both property names.
    assert '"city"' in joined or '\\"city\\"' in joined
    assert '"count"' in joined or '\\"count\\"' in joined
    # Must produce an args rule.
    assert "args " in joined or "args::=" in joined.replace(" ", "") or joined.startswith("args")


def test_schema_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            },
        },
    }
    rules = _json_schema_to_gbnf(schema, "root-obj")
    joined = "\n".join(rules)
    assert '"outer"' in joined or '\\"outer\\"' in joined
    assert '"inner"' in joined or '\\"inner\\"' in joined


def test_schema_array_items():
    schema = {
        "type": "array",
        "items": {"type": "integer"},
    }
    rules = _json_schema_to_gbnf(schema, "ints")
    joined = "\n".join(rules)
    assert "ints" in joined
    # array syntax brackets.
    assert "[" in joined and "]" in joined


# ---------- Full Hermes grammar ----------

def test_hermes_grammar_wraps_tool_call_tags():
    tools = [_weather_tool()]
    grammar = _hermes_gbnf(tools, _analysis_hermes())
    # Must reference the opening and closing Hermes tags as literals.
    assert "<tool_call>" in grammar
    assert "</tool_call>" in grammar
    assert "get_weather" in grammar


def test_hermes_grammar_couples_name_with_its_own_args():
    """If two tools have different arg schemas, the grammar must
    prevent a cross-pair like name='a' with args matching b's schema.
    Simple check: each tool name appears in its own rule alongside
    its own args rule, not the other tool's args rule.
    """
    tools = [
        _tool("alpha", parameters={"x": ToolParameter(name="x", type="integer")}),
        _tool("beta",  parameters={"y": ToolParameter(name="y", type="string")}),
    ]
    grammar = _hermes_gbnf(tools, _analysis_hermes())
    # Heuristic: line containing "alpha" must not also contain "y" as a
    # top-level property name (which would mean alpha's body permits
    # beta's schema).
    alpha_lines = [l for l in grammar.split("\n") if "alpha" in l]
    assert alpha_lines, "alpha must appear in some rule"
    for l in alpha_lines:
        # The alpha line may reference args-0 or similar; just make sure
        # it doesn't literally contain "\"y\"" which would indicate
        # inlined beta schema.
        assert '\\"y\\"' not in l


def test_hermes_grammar_two_tools_produces_union_at_top():
    """With two tools, the top-level tool-call rule should be an
    alternation so either can appear."""
    tools = [
        _tool("foo", parameters={"a": ToolParameter(name="a", type="string")}),
        _tool("bar", parameters={"b": ToolParameter(name="b", type="string")}),
    ]
    grammar = _hermes_gbnf(tools, _analysis_hermes())
    # '|' must appear somewhere in the grammar (alternation).
    assert "|" in grammar
    assert "foo" in grammar
    assert "bar" in grammar


def test_generic_json_grammar_no_wrapper_tags():
    tools = [_weather_tool()]
    grammar = _generic_json_gbnf(tools, _analysis_generic())
    # Must NOT include Hermes-style wrappers.
    assert "<tool_call>" not in grammar
    assert "</tool_call>" not in grammar
    # Still references the tool name.
    assert "get_weather" in grammar


def test_generate_gbnf_dispatches_on_mode():
    tools = [_weather_tool()]
    # Hermes: contains the Hermes wrappers.
    g1 = generate_gbnf(tools, _analysis_hermes())
    assert "<tool_call>" in g1
    # Generic: does not.
    g2 = generate_gbnf(tools, _analysis_generic())
    assert "<tool_call>" not in g2


def test_generate_gbnf_empty_tools_returns_empty_grammar():
    """Empty tools list is a legal no-tools request; grammar generator
    must not crash and must not leave dangling alternations."""
    grammar = generate_gbnf([], _analysis_hermes())
    # Should be either an empty string, or a grammar that accepts only
    # content (no tool-call alternation at all).  Either way, no hanging
    # '|' with nothing before/after it.
    assert not any(
        ln.strip().endswith("|") or ln.strip().startswith("|") or " | |" in ln
        for ln in grammar.split("\n")
    )


# ---------- Grammar wiring ----------

def test_hermes_handler_generate_grammar_returns_non_none_with_tools():
    """Integration: HermesHandler.generate_grammar should now return a
    usable grammar string when tools are provided."""
    from egg_toolbox.formats.hermes import HermesHandler
    handler = HermesHandler(_analysis_hermes())
    grammar = handler.generate_grammar([_weather_tool()])
    assert grammar is not None
    assert isinstance(grammar, str)
    assert "get_weather" in grammar


def test_unsupported_schema_features_raise():
    """$ref, oneOf, anyOf, allOf are explicitly out of scope; the
    generator should raise rather than silently produce a wrong
    grammar."""
    with pytest.raises((ValueError, NotImplementedError)):
        _json_schema_to_gbnf({"$ref": "#/definitions/X"}, "r")
    with pytest.raises((ValueError, NotImplementedError)):
        _json_schema_to_gbnf({"oneOf": [{"type": "string"}, {"type": "number"}]}, "r")
