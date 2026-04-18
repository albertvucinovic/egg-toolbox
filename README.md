# egg-toolbox

Universal tool calling middleware for local LLMs. A drop-in OpenAI-compatible API server that adds structured tool calling to any model whose chat template supports a tool-call format (Hermes, Llama 3, Mistral, DeepSeek, etc.) — without requiring that model to be fine-tuned for OpenAI-style `tool_calls` output.

## Status

- **Phase 1 complete**: types, template engine, streaming parser, tinygrad backend, OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming), token-usage accounting, `tool_choice`, `stream_options`, error handling.
- **Phase 2 in progress**: format handlers for non-Hermes models. Hermes (Qwen, NousResearch Hermes fine-tunes) and Llama 3 formats are implemented; Mistral / DeepSeek / Functionary / Command-R / Harmony / Generic still TODO.
- **End-to-end verified**: `egg-mono` (OpenAI-compat client) → `egg-toolbox` → tinygrad → Qwen2.5-0.5B-Instruct with clean tool-call round-trips.

## Repository layout

```
.
├── README.md            ← you are here
├── CLAUDE.md            project conventions / build & test notes
├── AGENTS.md            beads workflow reminders for coding agents
├── docs/                design docs (plan, architecture, session continuation)
├── egg-toolbox/         the Python project
│   ├── pyproject.toml
│   ├── egg_toolbox/     the importable package
│   ├── tests/
│   └── context.md       per-module architecture overview
└── .beads/              bd (beads) issue tracker database
```

## Quick start

Install in editable mode (from the repo root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "./egg-toolbox[tinygrad,dev]"
```

Start the server with a GGUF model:

```bash
python -m egg_toolbox path/to/model.gguf --backend tinygrad --host 127.0.0.1 --port 8000
```

The server listens on `http://127.0.0.1:8000` and exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/v1/models` | GET | List the loaded model |
| `/health` | GET | Health check |

## Example — tool call via `curl`

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "What is the weather in Zagreb?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

Qwen2.5-0.5B emits a `<tool_call>` block which egg-toolbox projects to the OpenAI-compat `tool_calls` object:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Zagreb\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Example — OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(resp.choices[0].message.tool_calls)
```

## CLI options

```
python -m egg_toolbox MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `MODEL` | *(required)* | Path to model (GGUF file or HF model ID) |
| `--backend` | `tinygrad` | Backend: `tinygrad` (shipped), `vllm` / `sglang` / `llamacpp` (planned) |
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |
| `--chat-template` | *(auto)* | Override chat template (path to .jinja file) |
| `--tool-format` | *(auto)* | Force format: `hermes`, `llama3`, `mistral`, `deepseek`, `functionary`, `command_r`, `generic` |
| `--temperature` | `0.0` | Default sampling temperature |
| `--max-tokens` | *(none)* | Default max tokens |
| `--gpu-layers` | `-1` | GPU layers for llamacpp (all layers on GPU) |
| `--tensor-parallel` | `1` | Tensor parallel size for vllm/sglang |

## Backends

| Backend | Status | Install extra |
|---|---|---|
| tinygrad | Shipped (includes Qwen2/Qwen2.5 attn-bias fix) | `pip install -e "./egg-toolbox[tinygrad]"` |
| vLLM | Planned | `pip install -e "./egg-toolbox[vllm]"` |
| SGLang | Planned | `pip install -e "./egg-toolbox[sglang]"` |
| llama-cpp-python | Planned | `pip install -e "./egg-toolbox[llamacpp]"` |

## Architecture

```
API request
  → Orchestrator
    → ChatTemplate.render()        (Jinja2 chat template from GGUF metadata)
    → Backend.generate_tokens()    (per-model generator loop)
    → StreamingParser              (per-format state machine)
    → SemanticEvent stream         (universal IR)
  → API layer (OpenAI SSE projection; Anthropic planned)
```

The universal IR is `SemanticEvent` — OpenAI and Anthropic formats are lossless projections of it. Tool-call parsing uses per-format state machine parsers (not regex) for correct streaming behavior. See `docs/egg-toolbox-architecture.md` for the full spec and `egg-toolbox/context.md` for a module-level overview.

## Development

```bash
# From repo root after `pip install -e "./egg-toolbox[dev]"`:
python -m pytest egg-toolbox/tests/

# Quick import check
python -c "from egg_toolbox.types import *; print('OK')"
```

Issue tracking uses [beads](https://github.com/beads-dev/beads) (`bd`). Run `bd ready` for available work; see `AGENTS.md` for conventions.

## License

See `egg-toolbox/LICENSE` (when added) for details.
