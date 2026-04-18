# Plan: `omnitool` -- Universal Tool Calling Middleware for Local LLMs

## Context

The user wants a standalone Python library that sits between local LLM inference backends (tinygrad, vllm, sglang, llama-cpp-python) and API consumers, providing universal tool calling with streaming. The core challenge: every model family has a different tool calling format (Hermes, Llama, Mistral, DeepSeek, Functionary, Command-R, Gemma 4, Harmony -- 13+ known formats). The library must abstract over all of them, expose both OpenAI-compatible and Anthropic-compatible HTTP APIs, and support streaming tool calls.

**Key design insight from llama.cpp**: Tool calling is solved by (1) using each model's Jinja2 chat template to format tools in the prompt, (2) differential template analysis to auto-detect the model's tool calling format, (3) per-format state machine parsers that extract tool calls from the token stream, and (4) grammar-based constrained generation where backends support it.

**Decisions made**: Standalone repo (not egg-mono), pure Python (no Rust), tinygrad as first backend.

---

## Architecture

```
                     API Consumers (OpenAI / Anthropic clients)
                               |
                    +----------+----------+
                    |     API Layer        |
                    |  openai.py           |
                    |  anthropic.py        |
                    +----------+----------+
                               |
                      SemanticEvent stream  <-- universal IR
                               |
                    +----------+----------+
                    |    Orchestrator       |  compile -> generate -> parse -> emit
                    +----------+----------+
                          /          \
              +-----------+    +------+--------+
              | Template   |    | StreamParser  |
              | Engine     |    | (state machine|
              | (Jinja2)   |    |  per format)  |
              +-----+------+    +------+--------+
                    |                  |
              +-----+------+   +------+------+
              | Detector   |   | Grammar     |
              | (diff      |   | Generator   |
              |  analysis) |   | (GBNF)      |
              +-----+------+   +------+------+
                    |                 |
              +-----+---------+------+
              |  Format Handlers     |
              |  hermes, llama3,     |
              |  mistral, deepseek,  |
              |  functionary,        |
              |  command_r, harmony, |
              |  generic             |
              +----------+-----------+
                         |
              +----------+-----------+
              |  Backend Abstraction |
              |  StepBackend (tinygrad, llamacpp)        -- we control token loop
              |  ConstraintBackend (vllm, sglang)        -- they control loop, we inject constraints
              +----------------------+
```

### Core Abstraction: SemanticEvent

All backends and format handlers produce the same event stream. API layers are lossless projections from it.

```python
class EventKind(enum.Enum):
    CONTENT_DELTA    = "content_delta"
    REASONING_DELTA  = "reasoning_delta"
    TOOL_CALL_START  = "tool_call_start"
    TOOL_CALL_NAME   = "tool_call_name"
    TOOL_ARGS_DELTA  = "tool_args_delta"
    TOOL_CALL_COMMIT = "tool_call_commit"
    DONE             = "done"
```

### Two Backend Modes

- **StepBackend** (tinygrad, llama-cpp-python): Exposes `generate_tokens() -> Iterator[int]`. Orchestrator controls the loop, decodes tokens, feeds parser, checks stops.
- **ConstraintBackend** (vLLM, SGLang): Exposes `generate_stream() -> AsyncIterator[str]`. Backend owns batching/sampling, we inject constraints (grammars, structured_outputs, stop strings) at request time.

---

## Directory Layout

```
omnitool/
  pyproject.toml
  omnitool/
    __init__.py
    __main__.py           # CLI entry point (argparse)
    types.py              # Frozen dataclasses: Tool, ChatMessage, SemanticEvent, SamplingParams, CompiledRequest
    orchestrator.py       # Request lifecycle: template -> generate -> parse -> emit
    template.py           # Jinja2 chat template engine (loads from GGUF / HF config)
    detector.py           # Format auto-detection via differential template analysis
    parser.py             # StreamingParser wrapper (gen-prompt stripping, accumulation)
    grammar.py            # JSON schema -> GBNF grammar generation
    formats/
      __init__.py
      base.py             # FormatHandler ABC + FormatParserState ABC
      hermes.py           # Hermes/Qwen <tool_call> format
      llama3.py           # Llama 3.x <|python_tag|> format
      mistral.py          # Mistral [TOOL_CALLS] format
      deepseek.py         # DeepSeek fullwidth-unicode format
      functionary.py      # Functionary >>> / <function=> formats
      command_r.py        # Command-R <|START_ACTION|> format
      harmony.py          # Harmony multi-channel format
      generic.py          # Fallback: raw JSON extraction
    backends/
      __init__.py
      base.py             # StepBackend + ConstraintBackend ABCs, Tokenizer ABC
      tinygrad.py         # Wraps Transformer.generate()
      llamacpp.py         # Wraps llama-cpp-python Llama class
      vllm.py             # Wraps vLLM AsyncLLM
      sglang.py           # Wraps SGLang Engine
    api/
      __init__.py
      openai.py           # /v1/chat/completions endpoint (streaming + non-streaming)
      anthropic.py        # /v1/messages endpoint (streaming + non-streaming)
      middleware.py        # Starlette app factory, CORS, routing
  tests/
    fixtures/
      templates/          # Real Jinja2 templates from popular models
      outputs/            # Recorded token sequences for parser tests
    test_formats/         # Per-format state machine tests
    test_api/             # SSE format projection tests
    ...
```

---

## Implementation Phases

### Phase 1: Foundation (tinygrad + Hermes + OpenAI streaming)

**Goal**: Load a GGUF model in tinygrad, handle tool-calling requests via OpenAI API.

1. **`types.py`** -- All frozen dataclasses and enums (Tool, ChatMessage, SemanticEvent, SamplingParams, CompiledRequest, FormatAnalysis, EventKind, StopReason)
2. **`template.py`** -- Jinja2 ChatTemplate class: load from GGUF metadata or file, render messages+tools, capability probing (supports_tools, supports_reasoning)
3. **`formats/base.py`** -- FormatHandler ABC (stop_strings, create_parser_state, generate_grammar) and FormatParserState ABC (feed_token, feed_text, finish)
4. **`formats/hermes.py`** -- Hermes state machine parser (CONTENT -> MAYBE_TAG -> IN_TOOL_TAG -> AFTER_TOOL_CALL). Handles `<tool_call>JSON</tool_call>`, partial match buffering, `<think>` reasoning blocks
5. **`parser.py`** -- StreamingParser wrapper: generation prompt stripping, event accumulation
6. **`backends/base.py`** -- StepBackend ABC (`generate_tokens -> Iterator[int]`), ConstraintBackend ABC, Tokenizer ABC
7. **`backends/tinygrad.py`** -- TinygradBackend: wraps `Transformer.generate()`, GGUF template extraction, TinygradTokenizer
8. **`orchestrator.py`** -- Request lifecycle: render template -> tokenize -> create compiled request -> generate tokens -> feed parser -> yield events. Step-backend flow with stop string partial matching (StopStringMatcher)
9. **`api/openai.py`** -- `/v1/chat/completions`: parse OpenAI request -> call orchestrator -> project SemanticEvents to OpenAI SSE chunks. Handles streaming + non-streaming
10. **`api/middleware.py`** -- Starlette app factory with CORS, `/v1/models`, `/health`
11. **`__main__.py`** -- argparse CLI: `omnitool model.gguf --backend tinygrad --port 8000`
12. **`detector.py`** -- Stub: keyword match on template source (return HERMES if `<tool_call>` found)
13. **Tests** -- Template rendering, Hermes parser with recorded token sequences, OpenAI SSE format correctness

**Validation**: `omnitool qwen3-8b.gguf --backend tinygrad`, then use OpenAI Python client to send a tool-calling request and receive streaming tool calls.

### Phase 2: Format Coverage + Anthropic API

1. **`detector.py`** -- Full differential analysis algorithm (port from llama.cpp autoparser): render template with sentinel values, diff outputs, extract structural markers, classify format
2. **Format handlers**: `llama3.py`, `mistral.py`, `deepseek.py`, `functionary.py`, `command_r.py`, `harmony.py`, `generic.py` -- each with its own state machine. Harmony uses a multi-channel state machine (IDLE/ANALYSIS/COMMENTARY/FINAL) mapping channels to SemanticEvent types (analysis→REASONING_DELTA, commentary→TOOL_CALL_*, final→CONTENT_DELTA), with a TypeScript namespace converter for tool definitions
3. **`api/anthropic.py`** -- `/v1/messages`: project SemanticEvents to Anthropic content_block_start/delta/stop SSE format
4. **`grammar.py`** -- JSON schema to GBNF conversion for Hermes format
5. **Tests** -- Auto-detect format for 20+ popular model templates, per-format parser tests

### Phase 3: Backend Expansion

1. **`backends/llamacpp.py`** -- llama-cpp-python Llama class, GBNF grammar support, lazy grammar activation
2. **`backends/vllm.py`** -- AsyncLLM, structured_outputs constraint injection, DELTA streaming
3. **`backends/sglang.py`** -- Engine class, Xgrammar integration
4. **`grammar.py`** -- Full GBNF for all formats, JSON schema generation for structured_outputs backends
5. **Speculative decoding (backend-transparent)** -- Works automatically for step backends since speculative decoding produces verified tokens via the same `Iterator[int]` interface. For llama-cpp-python, enable via `n_draft` parameter. For vLLM, enable via `--speculative-model` flag

### Phase 4: Production Hardening

1. Error recovery (malformed JSON, partial tool calls)
2. Request cancellation/timeouts
3. Parallel tool calls
4. Token usage tracking in responses
5. **Grammar-guided speculative decoding** -- `GrammarState` class with `is_valid(token_id) -> bool`, `valid_token_ids() -> set[int]`, `advance(token_id)`, `clone()` methods. Extends `StepBackend` ABC with optional `supports_speculative` property and `generate_tokens_speculative(request, grammar_state)` method. Allows draft-phase token rejection when grammar constraints are active
6. Documentation

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Pure Python | Hot path is in backends (GPU kernels). Middleware processes 1 token/transition -- trivially fast in Python |
| HTTP framework | Starlette (not FastAPI) | Lighter, no Pydantic dependency, native ASGI streaming. FastAPI is built on Starlette anyway |
| Type layer | Frozen dataclasses (not Pydantic) | Zero dependencies, enforced immutability, explicit JSON serialization in API layer |
| Backend ABCs | Two (Step + Constraint) | Fundamentally different control flow; single ABC would be either too broad or require runtime dispatch |
| Format detection | At model load (not per-request) | Template doesn't change between requests; detector does 7+ Jinja2 renders |
| Parser approach | Per-format state machines (not regex) | Streaming requires incremental processing; regex needs complete text |
| Universal IR | SemanticEvent enum | Both API formats are lossless projections; keeps format logic at edges |

## Dependencies

```
Required: jinja2, starlette, uvicorn
Extras:   tinygrad | vllm | sglang | llama-cpp-python
Dev:      pytest, pytest-asyncio, httpx
```

---

## Verification

1. **Unit tests** (no model): template rendering, format detection on collected templates, parser state machines with recorded token sequences, OpenAI/Anthropic SSE format projection
2. **Integration test** (small model): Load Qwen3-0.6B GGUF via tinygrad, send tool-calling request via OpenAI client, verify streaming tool calls arrive correctly
3. **Cross-backend test**: Same test suite passes on all backends
4. **Format coverage test**: Auto-detect format correctly for 20+ popular model templates
