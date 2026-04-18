# egg-toolbox

Universal tool calling middleware for local LLMs. Drop-in OpenAI-compatible API that adds structured tool calling to any model that supports a chat template with tool syntax (Hermes, Llama 3, Mistral, DeepSeek, etc.).

## Quick Start

```bash
pip install -e ".[tinygrad]"

# Start the server with a GGUF model
egg-toolbox path/to/model.gguf
```

The server starts on `http://0.0.0.0:8000` by default and exposes an OpenAI-compatible API.

## API Usage

### Non-streaming with tools

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
)
print(response.choices[0].message.tool_calls)
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |

## CLI Options

```
egg-toolbox MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `MODEL` | *(required)* | Path to model (GGUF file or HF model ID) |
| `--backend` | `tinygrad` | Backend to use: `tinygrad`, `vllm`, `sglang`, `llamacpp` |
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |
| `--chat-template` | *(auto)* | Override chat template (path to .jinja file) |
| `--tool-format` | *(auto)* | Override tool format: `hermes`, `llama3`, `mistral`, `deepseek`, `functionary`, `command_r`, `generic` |
| `--temperature` | `0.0` | Default sampling temperature |
| `--max-tokens` | *(none)* | Default max tokens |
| `--gpu-layers` | `-1` (all) | GPU layers for llamacpp backend |
| `--tensor-parallel` | `1` | Tensor parallel size for vllm/sglang |

## Backends

| Backend | Status | Install |
|---|---|---|
| tinygrad | Implemented | `pip install -e ".[tinygrad]"` |
| vLLM | Planned | `pip install -e ".[vllm]"` |
| SGLang | Planned | `pip install -e ".[sglang]"` |
| llama-cpp-python | Planned | `pip install -e ".[llamacpp]"` |

## Architecture

```
API Request → Orchestrator → ChatTemplate.render()
  → Backend.generate_tokens() → StreamingParser
  → SemanticEvent stream → API SSE projection
```

The universal IR is `SemanticEvent` — both OpenAI and Anthropic API formats are lossless projections of this internal representation. Tool call parsing uses per-format state machine parsers (not regex) for correct streaming behavior.

## Development

```bash
# Create venv and install dev deps
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Quick import check
python -c "from egg_toolbox.types import *; print('OK')"
```

## License

See [LICENSE](LICENSE) for details.
