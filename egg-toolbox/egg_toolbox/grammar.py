"""GBNF grammar generation from tool definitions.

Produces llama.cpp-compatible GBNF grammar strings that constrain the
model's output to syntactically valid tool calls.  Consumed by step
backends that support grammar-guided decoding (llama-cpp-python,
future tinygrad grammar sampler, vendored xgrammar).

Scope:
- Primitives: ``string``, ``number``, ``integer``, ``boolean``, ``null``.
- ``enum`` constraint on primitives.
- ``object`` with named ``properties``.
- ``array`` with homogeneous ``items``.

Out of scope (raises ``NotImplementedError``):
- ``$ref``, ``oneOf`` / ``anyOf`` / ``allOf`` -- uncommon in tool
  schemas and would pull in a full JSON-Schema engine.  Models rarely
  emit these shapes for tool args anyway.

The grammar produced is *lenient about property order and presence*:
any subset of defined properties may appear in any order.  That's
the practical middle ground -- strict required-field enforcement
would require combinatorial rule generation, and our streaming parser
already handles missing fields gracefully at commit time.
"""
from __future__ import annotations

from .types import FormatAnalysis, Tool, ToolFormatMode


# -------- public API --------

def generate_gbnf(tools: list[Tool], analysis: FormatAnalysis) -> str:
    """Generate a GBNF grammar for the given tools.

    Returns a GBNF string; empty string if ``tools`` is empty (no tool
    calls are permitted by the grammar in that case -- content only).
    Dispatches on the format's ``tool_mode``.
    """
    if not tools:
        return ""
    mode = analysis.tool_mode
    if mode == ToolFormatMode.HERMES:
        return _hermes_gbnf(tools, analysis)
    if mode == ToolFormatMode.LLAMA3:
        return _llama3_gbnf(tools, analysis)
    # Everything else collapses to bare-JSON for now.
    return _generic_json_gbnf(tools, analysis)


# -------- dispatchers --------

def _hermes_gbnf(tools: list[Tool], analysis: FormatAnalysis) -> str:
    """Hermes wrapper: ``<tool_call>{json}</tool_call>``.

    Each tool gets its own rule that couples the ``name`` literal with
    its own args rule, so the grammar can't mix tool A's name with
    tool B's schema.
    """
    start = analysis.tool_call_start or "<tool_call>"
    end = analysis.tool_call_end or "</tool_call>"
    name_key = analysis.name_field or "name"
    args_key = analysis.args_field or "arguments"

    lines: list[str] = []
    # Any amount of free-form content, optionally wrapped with tool calls.
    # content is "anything that isn't the open bracket of our tag" so the
    # parser gets a chance to switch rules at '<'.
    lines.append(f'root ::= content ( tool-wrap content )*')
    lines.append(
        f'tool-wrap ::= "{_lit(start)}" ws tool-call ws "{_lit(end)}"'
    )

    tool_rule_names: list[str] = []
    for i, tool in enumerate(tools):
        tool_rule = f"tool-{i}"
        args_rule = f"args-{i}"
        tool_rule_names.append(tool_rule)
        lines.append(
            f'{tool_rule} ::= "{{" ws '
            f'"{_lit_json_key(name_key)}" ws ":" ws "{_lit_json_string(tool.function.name)}" ws '
            f'"," ws '
            f'"{_lit_json_key(args_key)}" ws ":" ws {args_rule} ws '
            f'"}}"'
        )
        lines.extend(_json_schema_to_gbnf(tool.to_json_schema(), args_rule))

    lines.append(f'tool-call ::= {" | ".join(tool_rule_names)}')

    lines.extend(_json_primitives_gbnf())
    lines.append('content ::= [^<]*')
    return "\n".join(lines)


def _llama3_gbnf(tools: list[Tool], analysis: FormatAnalysis) -> str:
    """Llama 3.1+ python-tag form: ``<|python_tag|>{json}<|eom_id|>``.

    Same JSON body as Hermes but with different wrappers.
    """
    lines: list[str] = []
    lines.append(
        'root ::= content ( "<|python_tag|>" ws tool-call ws "<|eom_id|>" content )*'
    )

    tool_rule_names: list[str] = []
    for i, tool in enumerate(tools):
        tool_rule = f"tool-{i}"
        args_rule = f"args-{i}"
        tool_rule_names.append(tool_rule)
        lines.append(
            f'{tool_rule} ::= "{{" ws '
            f'"\\"name\\"" ws ":" ws "{_lit_json_string(tool.function.name)}" ws '
            f'"," ws '
            f'"\\"parameters\\"" ws ":" ws {args_rule} ws '
            f'"}}"'
        )
        lines.extend(_json_schema_to_gbnf(tool.to_json_schema(), args_rule))

    lines.append(f'tool-call ::= {" | ".join(tool_rule_names)}')
    lines.extend(_json_primitives_gbnf())
    lines.append('content ::= [^<]*')
    return "\n".join(lines)


def _generic_json_gbnf(tools: list[Tool], analysis: FormatAnalysis) -> str:
    """Bare-JSON form: the whole response is a JSON object (no wrappers).

    ``root ::= tool-call`` -- the model emits only the JSON object.
    """
    lines: list[str] = []
    tool_rule_names: list[str] = []
    for i, tool in enumerate(tools):
        tool_rule = f"tool-{i}"
        args_rule = f"args-{i}"
        tool_rule_names.append(tool_rule)
        lines.append(
            f'{tool_rule} ::= "{{" ws '
            f'"\\"name\\"" ws ":" ws "{_lit_json_string(tool.function.name)}" ws '
            f'"," ws '
            f'"\\"arguments\\"" ws ":" ws {args_rule} ws '
            f'"}}"'
        )
        lines.extend(_json_schema_to_gbnf(tool.to_json_schema(), args_rule))

    lines.append(f'root ::= {" | ".join(tool_rule_names)}')
    lines.extend(_json_primitives_gbnf())
    return "\n".join(lines)


# -------- JSON schema -> GBNF --------

def _json_schema_to_gbnf(schema: dict, rule_name: str) -> list[str]:
    """Return a list of GBNF rule lines for ``schema``, rooted at
    ``rule_name``.  The top rule is always named ``rule_name``;
    sub-rules are suffixed (``<rule_name>-prop-<key>``, etc.)."""
    # Refuse unsupported shapes.
    for unsupported in ("$ref", "oneOf", "anyOf", "allOf"):
        if unsupported in schema:
            raise NotImplementedError(
                f"JSON Schema feature {unsupported!r} is not supported by "
                "the GBNF generator.  Flatten the schema or file a bead."
            )

    t = schema.get("type")
    if "enum" in schema:
        return [_enum_rule(rule_name, schema["enum"])]
    if t == "string":
        return [f'{rule_name} ::= json-string']
    if t == "integer":
        return [f'{rule_name} ::= json-integer']
    if t == "number":
        return [f'{rule_name} ::= json-number']
    if t == "boolean":
        return [f'{rule_name} ::= json-bool']
    if t == "null":
        return [f'{rule_name} ::= json-null']
    if t == "array":
        return _array_rule(rule_name, schema)
    if t == "object":
        return _object_rule(rule_name, schema)
    # Unknown type: accept anything JSON-ish.
    return [f'{rule_name} ::= json-value']


def _enum_rule(rule_name: str, values: list) -> str:
    """Fixed alternation of literal JSON values."""
    alts: list[str] = []
    for v in values:
        if isinstance(v, str):
            alts.append(f'"{_lit_json_string(v)}"')
        elif isinstance(v, bool):
            alts.append(f'"{str(v).lower()}"')
        elif v is None:
            alts.append('"null"')
        elif isinstance(v, (int, float)):
            alts.append(f'"{v}"')
        else:
            # Fall back to string-ification for oddities.
            alts.append(f'"{_lit_json_string(str(v))}"')
    return f'{rule_name} ::= {" | ".join(alts)}'


def _array_rule(rule_name: str, schema: dict) -> list[str]:
    items = schema.get("items")
    if items is None:
        # Untyped array: permit any JSON value.
        return [
            f'{rule_name} ::= "[" ws "]" | "[" ws json-value (ws "," ws json-value)* ws "]"',
        ]
    item_rule = f'{rule_name}-item'
    lines = _json_schema_to_gbnf(items, item_rule)
    lines.append(
        f'{rule_name} ::= "[" ws "]" | "[" ws {item_rule} (ws "," ws {item_rule})* ws "]"'
    )
    return lines


def _object_rule(rule_name: str, schema: dict) -> list[str]:
    properties = schema.get("properties") or {}
    if not properties:
        # Empty object permitted.
        return [f'{rule_name} ::= "{{" ws "}}"']

    lines: list[str] = []
    prop_alts: list[str] = []
    for key, sub_schema in properties.items():
        key_safe = _safe_rule_suffix(key)
        prop_rule = f'{rule_name}-prop-{key_safe}'
        val_rule = f'{rule_name}-val-{key_safe}'
        lines.extend(_json_schema_to_gbnf(sub_schema, val_rule))
        lines.append(
            f'{prop_rule} ::= "{_lit_json_string(key)}" ws ":" ws {val_rule}'
        )
        prop_alts.append(prop_rule)

    # Lenient: any property may appear, in any order, any number of
    # times (clients typically produce each once; strict enforcement
    # would need per-permutation rules).
    any_prop = f'{rule_name}-any-prop'
    lines.append(f'{any_prop} ::= {" | ".join(prop_alts)}')
    lines.append(
        f'{rule_name} ::= "{{" ws "}}" | '
        f'"{{" ws {any_prop} (ws "," ws {any_prop})* ws "}}"'
    )
    return lines


def _json_primitives_gbnf() -> list[str]:
    """Canonical JSON primitive rules, reused by every tool format."""
    return [
        'ws ::= [ \\t\\n\\r]*',
        # Permissive string: any char except unescaped quote/backslash,
        # plus single-char escapes.  No \\uXXXX support -- rare in tool
        # args and simplifies the grammar.
        'json-string ::= "\\"" ( [^"\\\\] | "\\\\" ["\\\\/bfnrt] )* "\\""',
        'json-integer ::= "-"? ("0" | [1-9] [0-9]*)',
        'json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
        'json-bool ::= "true" | "false"',
        'json-null ::= "null"',
        'json-value ::= json-string | json-number | json-bool | json-null | '
        '"{" ws "}" | "[" ws "]"',
    ]


# -------- literal escaping --------

def _lit(s: str) -> str:
    """Escape a string so it's safe inside a GBNF double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _lit_json_key(key: str) -> str:
    """Produce the GBNF representation of a JSON key literal.

    For key ``city`` the JSON literal is ``"city"``; inside a GBNF
    double-quoted string that becomes ``\\"city\\"``.
    """
    return f'\\"{_lit(key)}\\"'


def _lit_json_string(value: str) -> str:
    """Same as :func:`_lit_json_key` but semantically for values."""
    return f'\\"{_lit(value)}\\"'


def _safe_rule_suffix(s: str) -> str:
    """Make a string safe to use as a GBNF rule-name suffix."""
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append(f"x{ord(ch):x}")
    return "".join(out) or "unnamed"
