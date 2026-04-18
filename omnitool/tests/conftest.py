"""Shared fixtures for E2E tests."""
from __future__ import annotations

import os
import urllib.request

# Disable tinygrad's SQLite disk cache before it's imported.
# TestClient runs the ASGI app in a separate thread which triggers
# SQLite's same-thread check against tinygrad's cached connection.
os.environ.setdefault("CACHELEVEL", "0")
from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from omnitool.types import CompiledRequest
from omnitool.backends.base import StepBackend, Tokenizer
from omnitool.orchestrator import Orchestrator
from omnitool.api.middleware import create_app


# ---------------------------------------------------------------------------
# Minimal Hermes chat template (must contain "<tool_call>" for detector)
# ---------------------------------------------------------------------------

HERMES_TEMPLATE = """\
{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
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


# ---------------------------------------------------------------------------
# ScriptedBackend: deterministic token generation for pipeline tests
# ---------------------------------------------------------------------------

class CharTokenizer(Tokenizer):
    """Character-level tokenizer for deterministic testing."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t > 0)

    def decode_single(self, token_id: int) -> str:
        return chr(token_id) if token_id > 0 else ""

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def vocab_size(self) -> int:
        return 256


class ScriptedBackend(StepBackend):
    """Yields tokens from a predetermined output string."""

    def __init__(self, output: str, template: str = HERMES_TEMPLATE):
        self._output = output
        self._template = template
        self._tok = CharTokenizer()

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        pass

    def tokenizer(self) -> Tokenizer:
        return self._tok

    def chat_template_source(self) -> str:
        return self._template

    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        for c in self._output:
            yield ord(c)

    def model_name(self) -> str:
        return "scripted-test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_client():
    """Factory fixture: returns a TestClient backed by ScriptedBackend(output)."""
    def _factory(output: str) -> TestClient:
        backend = ScriptedBackend(output)
        orch = Orchestrator(backend)
        return TestClient(create_app(orch))
    return _factory


# ---------------------------------------------------------------------------
# Real GGUF model download (for tinygrad E2E tests)
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "fixtures" / "models"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    "/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
)
DEFAULT_MODEL_FILE = "qwen2.5-0.5b-instruct-q4_0.gguf"


@pytest.fixture(scope="session")
def gguf_model_path() -> Path:
    """Download and cache a small GGUF model for real-backend tests."""
    env_path = os.environ.get("OMNITOOL_TEST_MODEL")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        pytest.skip(f"OMNITOOL_TEST_MODEL={env_path} not found")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / DEFAULT_MODEL_FILE
    if path.exists():
        return path

    try:
        print(f"\nDownloading {DEFAULT_MODEL_FILE} …")
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, path)
    except Exception as e:
        pytest.skip(f"Model download failed: {e}")

    return path
