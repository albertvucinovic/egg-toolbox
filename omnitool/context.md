# omnitool - Context & Architecture Overview

## Purpose
Universal tool calling middleware for local LLM backends. Sits between inference engines (tinygrad, vllm, sglang, llama-cpp-python) and API consumers, adding tool call detection/parsing and exposing OpenAI + Anthropic compatible streaming APIs.

## Module Architecture

### Core Types (`omnitool/types.py`)
- **Frozen dataclasses**: `Tool`, `ToolFunction`, `ToolParameter`, `ChatMessage`, `ContentPart`, `ToolCall`, `ToolCallFunction`, `SemanticEvent`, `SamplingParams`, `CompiledRequest`, `FormatAnalysis`
- **Enums**: `EventKind`, `StopReason`, `ToolFormatMode`, `ReasoningMode`
- `SemanticEvent` is the universal IR -- both OpenAI and Anthropic APIs are lossless projections from it

### Template Engine (`omnitool/template.py`)
- `ChatTemplate` class: loads Jinja2 chat templates from GGUF metadata or HF tokenizer_config.json
- Renders messages + tools into model-specific prompt format
- Capability probing: `supports_tools()`, `supports_reasoning()`, `supports_parallel_tools()`
- Full GGUF metadata parser for extracting chat_template, BOS/EOS tokens

### Format Detection (`omnitool/detector.py`)
- `detect_format(template) -> FormatAnalysis`: identifies which tool calling format a model uses
- **Phase 1 (current)**: Keyword matching on template source
- **Phase 2 (planned)**: Differential template analysis (port of llama.cpp autoparser)

### Format Handlers (`omnitool/formats/`)
- `base.py`: `FormatHandler` ABC and `FormatParserState` ABC
- `hermes.py`: Hermes/Qwen `<tool_call>JSON</tool_call>` format (Phase 1)
- `__init__.py`: `get_handler_for_format()` factory function
- Planned (Phase 2): llama3, mistral, deepseek, functionary, command_r, harmony, generic

### Streaming Parser (`omnitool/parser.py`)
- `StreamingParser`: wraps a `FormatParserState`, handles generation prompt stripping, accumulates results
- Properties: `content`, `reasoning`, `tool_calls` for non-streaming access
- `trigger_detected()` for lazy grammar activation

### Backend Abstraction (`omnitool/backends/`)
- `base.py`: `Tokenizer` ABC, `StepBackend` ABC (token-by-token), `ConstraintBackend` ABC (text chunks)
- `tinygrad.py`: `TinygradBackend` wrapping `Transformer.generate()` (Phase 1)
- Planned (Phase 3): llamacpp, vllm, sglang

### Orchestrator (`omnitool/orchestrator.py`)
- `Orchestrator`: central coordinator, one per model
- `chat_completion()`: render template -> tokenize -> generate -> parse -> yield SemanticEvents
- `StopStringMatcher`: partial-match stop string detection for step backends
- Handles both step and constraint backend flows

### API Layer (`omnitool/api/`)
- `openai.py`: `/v1/chat/completions` -- streaming + non-streaming, SemanticEvent to OpenAI SSE projection
- `middleware.py`: Starlette app factory with CORS, `/v1/models`, `/health`
- Planned (Phase 2): `anthropic.py` for `/v1/messages`

### CLI (`omnitool/__main__.py`)
- argparse entry point: `omnitool model.gguf --backend tinygrad --port 8000`
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

## Implementation Status
- **Phase 1 (COMPLETE)**: types, template, detector (stub), hermes format, parser, backends/base, backends/tinygrad, orchestrator, api/openai, api/middleware, __main__
- **Phase 2 (TODO)**: Full detector, all format handlers, Anthropic API, grammar generation
- **Phase 3 (TODO)**: llama-cpp-python, vLLM, SGLang backends
- **Phase 4 (TODO)**: Error recovery, timeouts, parallel tools, token tracking, grammar-guided speculative decoding
