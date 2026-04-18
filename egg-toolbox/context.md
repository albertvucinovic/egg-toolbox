# egg-toolbox - Context & Architecture Overview

## Purpose
Universal tool calling middleware for local LLM backends. Sits between inference engines (tinygrad, vllm, sglang, llama-cpp-python) and API consumers, adding tool call detection/parsing and exposing OpenAI + Anthropic compatible streaming APIs.

## Module Architecture

### Core Types (`egg_toolbox/types.py`)
- **Frozen dataclasses**: `Tool`, `ToolFunction`, `ToolParameter`, `ChatMessage`, `ContentPart`, `ToolCall`, `ToolCallFunction`, `SemanticEvent`, `SamplingParams`, `CompiledRequest`, `FormatAnalysis`
- **Enums**: `EventKind`, `StopReason`, `ToolFormatMode`, `ReasoningMode`
- `SemanticEvent` is the universal IR -- both OpenAI and Anthropic APIs are lossless projections from it
- `SemanticEvent.prompt_tokens` / `completion_tokens`: populated on DONE events by orchestrator

### Template Engine (`egg_toolbox/template.py`)
- `ChatTemplate` class: loads Jinja2 chat templates from GGUF metadata or HF tokenizer_config.json
- Renders messages + tools into model-specific prompt format
- Capability probing: `supports_tools()`, `supports_reasoning()`, `supports_parallel_tools()`
- Full GGUF metadata parser for extracting chat_template, BOS/EOS tokens

### Format Detection (`egg_toolbox/detector.py`)
- `detect_format(template) -> FormatAnalysis`: identifies which tool calling format a model uses
- **Phase 1 (current)**: Keyword matching on template source
- **Phase 2 (planned)**: Differential template analysis (port of llama.cpp autoparser)

### Format Handlers (`egg_toolbox/formats/`)
- `base.py`: `FormatHandler` ABC and `FormatParserState` ABC
- `hermes.py`: Hermes/Qwen `<tool_call>JSON</tool_call>` format (Phase 1)
- `__init__.py`: `get_handler_for_format()` factory function
- Planned (Phase 2): llama3, mistral, deepseek, functionary, command_r, harmony, generic

### Streaming Parser (`egg_toolbox/parser.py`)
- `StreamingParser`: wraps a `FormatParserState`, handles generation prompt stripping, accumulates results
- Properties: `content`, `reasoning`, `tool_calls` for non-streaming access
- `trigger_detected()` for lazy grammar activation

### Backend Abstraction (`egg_toolbox/backends/`)
- `base.py`: `Tokenizer` ABC, `StepBackend` ABC (token-by-token), `ConstraintBackend` ABC (text chunks)
- `tinygrad.py`: `TinygradBackend` using `tinygrad.apps.llm.Transformer.from_gguf()` + `SimpleTokenizer.from_gguf_kv()` (Phase 1). Includes `_from_gguf_with_qkv_bias` helper that attaches attn_{q,k,v}.bias tensors (needed by Qwen2/Qwen2.5) which upstream tinygrad silently drops.
- Planned (Phase 3): llamacpp, vllm, sglang

### Orchestrator (`egg_toolbox/orchestrator.py`)
- `Orchestrator`: central coordinator, one per model
- `chat_completion()`: render template -> tokenize -> generate -> parse -> yield SemanticEvents
- `StopStringMatcher`: partial-match stop string detection for step backends
- Handles both step and constraint backend flows

### API Layer (`egg_toolbox/api/`)
- `openai.py`: `/v1/chat/completions` -- streaming + non-streaming, SemanticEvent to OpenAI SSE projection
  - Accepts pre-parsed body dict (JSON parsing/validation done in middleware)
  - `tool_choice`: `"auto"` (default), `"none"` (suppresses tools), `"required"`/dict (treated as auto for MVP)
  - `stream_options`: `{include_usage: true}` emits usage-only chunk before `[DONE]`
  - `system_fingerprint`: `"egg-toolbox-v0"` in all responses
  - Real token usage from orchestrator DONE events
- `middleware.py`: Starlette app factory with CORS, `/v1/models`, `/health`, error handling
  - JSON parse errors -> 400 with `invalid_request_error`
  - Missing/invalid `messages` -> 400
  - Orchestrator exceptions -> 500 with `server_error` (no stack traces)
- Planned (Phase 2): `anthropic.py` for `/v1/messages`

### CLI (`egg_toolbox/__main__.py`)
- argparse entry point: `egg-toolbox model.gguf --backend tinygrad --port 8000`
- Backend factory with lazy imports

## Data Flow
```
API Request -> parse_messages/parse_tools/parse_sampling
  -> Orchestrator.chat_completion()
    -> ChatTemplate.render() -> prompt string
    -> Tokenizer.encode() -> prompt tokens
    -> CompiledRequest (tokens + sampling + stops + grammar)
    -> Backend.generate_tokens() / generate_stream()
    -> StreamingParser.feed_token() / feed_text()
    -> FormatParserState state machine
    -> SemanticEvent stream
  -> API layer projects to OpenAI/Anthropic SSE format
```

## Dependencies
- Required: jinja2, starlette, uvicorn
- Backend extras: tinygrad | vllm | sglang | llama-cpp-python
- Dev: pytest, pytest-asyncio, httpx

## Testing (`tests/`)
- `conftest.py`: `ScriptedBackend`, `ErrorBackend`, `CharTokenizer`, `make_client`/`make_client_raw` fixtures, `gguf_model_path` fixture
- `test_e2e.py`: 23 tests total
  - `TestContentOnly`: streaming + non-streaming content (scripted)
  - `TestToolCalling`: streaming + non-streaming tool calls (scripted)
  - `TestSSEFormat`: SSE line format, chunk JSON structure, [DONE] sentinel (scripted)
  - `TestAuxEndpoints`: /health, /v1/models with created>0 and owned_by=egg-toolbox (scripted)
  - `TestErrorHandling`: invalid JSON, missing messages, empty messages, orchestrator error (scripted)
  - `TestTokenCounts`: non-streaming usage, streaming include_usage, streaming no usage default (scripted)
  - `TestToolChoice`: tool_choice=none suppresses tools, tool_choice=auto allows them (scripted)
  - `TestSystemFingerprint`: system_fingerprint in streaming + non-streaming (scripted)
  - `TestRealModel`: content generation, streaming format, tool request no-crash (real tinygrad + GGUF)
- Note: `CACHELEVEL=0` set in conftest to avoid tinygrad SQLite thread issues with TestClient

## Implementation Status
- **Phase 1 (COMPLETE)**: types, template, detector (stub), hermes format, parser, backends/base, backends/tinygrad, orchestrator, api/openai, api/middleware, __main__, E2E tests
- **Phase 2 (TODO)**: Full detector, all format handlers, Anthropic API, grammar generation
- **Phase 3 (TODO)**: llama-cpp-python, vLLM, SGLang backends
- **Phase 4 (TODO)**: Error recovery, timeouts, parallel tools, grammar-guided speculative decoding
- **Production API (COMPLETE)**: Token tracking, error handling, tool_choice, stream_options, system_fingerprint
