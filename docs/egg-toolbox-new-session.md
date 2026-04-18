# egg-toolbox -- Session Continuation Instructions

## What is egg-toolbox?

A standalone Python library providing universal tool calling middleware for local LLM backends. It sits between inference backends (tinygrad, vllm, sglang, llama-cpp-python) and API consumers, adding tool call detection/parsing and exposing OpenAI + Anthropic compatible streaming APIs.

## Documents to read first

1. **`./egg-toolbox-plan.md`** -- Condensed implementation plan with 4 phases, directory layout, and design decisions
2. **`./egg-toolbox-architecture.md`** -- Full architecture specification (~2500 lines) with complete code for all core components

These two files contain the entire design. Read them before doing anything.

## Decisions already made

- **Standalone repo** (not part of egg-mono monorepo)
- **Pure Python** (no Rust, no C extensions)
- **tinygrad as first backend**
- **Starlette** (not FastAPI) for HTTP
- **Frozen dataclasses** (not Pydantic) for types
- **SemanticEvent** as universal IR (both OpenAI and Anthropic are lossless projections)
- **Two backend ABCs**: StepBackend (tinygrad, llama-cpp-python) vs ConstraintBackend (vllm, sglang)
- **Per-format state machine parsers** (not regex)
- **Format detection at model load** (not per-request)

## What's been designed (not yet implemented)

Everything in the architecture doc is design only. No code has been written yet. The implementation follows 4 phases:

### Phase 1 (start here): Foundation
- `types.py`, `template.py`, `formats/base.py`, `formats/hermes.py`, `parser.py`
- `backends/base.py`, `backends/tinygrad.py`
- `orchestrator.py`, `api/openai.py`, `api/middleware.py`, `__main__.py`
- `detector.py` (stub: keyword match only)
- Validation: load Qwen GGUF via tinygrad, tool-calling request via OpenAI client

### Phase 2: Format coverage
- Full detector (differential template analysis)
- All format handlers: llama3, mistral, deepseek, functionary, command_r, **harmony**, generic
- Anthropic API, grammar generation

### Phase 3: Backend expansion
- llama-cpp-python, vllm, sglang backends
- Backend-transparent speculative decoding support

### Phase 4: Production hardening
- Error recovery, timeouts, parallel tool calls, token tracking
- Grammar-guided speculative decoding (`GrammarState` class)

## Key architectural questions resolved in prior session

1. **Harmony support**: Yes -- uses multi-channel state machine (IDLE/ANALYSIS/COMMENTARY/FINAL). Channels map to SemanticEvent types: analysis→REASONING_DELTA, commentary→TOOL_CALL_*, final→CONTENT_DELTA. Includes TypeScript namespace converter for tool definitions. Full handler code is in the architecture doc.

2. **Speculative decoding**: Two levels:
   - Backend-transparent (Phase 3): Works automatically since backends emit verified tokens through same `Iterator[int]`/`AsyncIterator[str]` interfaces
   - Grammar-guided (Phase 4): `GrammarState` class for draft-phase token rejection. See architecture doc Section 14.5

3. **Performance with vLLM/SGLang**: Near-zero overhead. egg-toolbox sits at request boundary (constraint injection) and output boundary (parse results), never inside the decode loop. Batching, scheduling, sampling all untouched.

4. **Value for existing projects**: The `formats/` and `detector.py` modules have zero backend dependency. vLLM/SGLang could `pip install egg-toolbox` and use only the parser layer. New format support would only need to be added once across all backends.

## To begin implementation

```
mkdir egg-toolbox && cd egg-toolbox
# Start with Phase 1, file by file, following the architecture doc
# The architecture doc contains complete code for every module
```

Implement in order: `types.py` → `template.py` → `formats/base.py` → `formats/hermes.py` → `parser.py` → `backends/base.py` → `backends/tinygrad.py` → `orchestrator.py` → `api/openai.py` → `api/middleware.py` → `__main__.py` → `detector.py` (stub)
