"""Regression test: SSE streaming must flush chunks progressively,
not buffer until the backend completes.

This matters in practice with slow backends (tinygrad decoding an 8B
model at tens of ms per token) and especially with long prompt
compiles -- clients expect to see tokens appear as they are produced.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator

import httpx
import pytest
import uvicorn
import threading

from egg_toolbox.backends.base import StepBackend, Tokenizer
from egg_toolbox.orchestrator import Orchestrator
from egg_toolbox.api.middleware import create_app
from egg_toolbox.types import CompiledRequest


HERMES_TEMPLATE = """\
{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message.get('content', '') or '' }}
<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""


class _CharTokenizer(Tokenizer):
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


class _SlowBackend(StepBackend):
    """Simulates a slow token-generating backend (e.g. tinygrad).

    Each next() call on generate_tokens blocks for ``delay`` seconds
    before yielding, exactly like a real model's per-token decode.
    """

    def __init__(self, output: str, delay: float):
        self._output = output
        self._delay = delay
        self._tok = _CharTokenizer()

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        pass

    def tokenizer(self) -> Tokenizer:
        return self._tok

    def chat_template_source(self) -> str:
        return HERMES_TEMPLATE

    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        for c in self._output:
            time.sleep(self._delay)   # blocking sleep == GPU-decode analogue
            yield ord(c)

    def model_name(self) -> str:
        return "slow-test"


@pytest.fixture
def running_server():
    """Spin up a real uvicorn on a random port with a slow backend."""
    backend = _SlowBackend("ABCDE", delay=0.1)   # 5 tokens, 100 ms each
    orch = Orchestrator(backend)
    app = create_app(orch)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to bind.
    for _ in range(100):
        time.sleep(0.05)
        if server.started and server.servers:
            break

    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def test_event_loop_not_blocked_during_generation(running_server):
    """A second request fired while the first is mid-generation must
    not wait for the first to finish.  The sync backend blocks the
    thread for N x delay, so if it ran on the event loop the second
    request's response-start would be delayed by the same amount.
    With the thread bridge, the event loop stays free and request 2's
    first chunk arrives almost immediately.
    """
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "go"}],
        "stream": True,
        "max_tokens": 10,
    }

    first_chunk_times: list[float] = []

    def fire_and_time(start: float) -> float:
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", f"{running_server}/v1/chat/completions",
                               json=payload) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        return time.monotonic() - start
        return -1.0

    t_start = time.monotonic()
    t1 = threading.Thread(target=lambda: first_chunk_times.append(fire_and_time(t_start)))
    t1.start()
    # Small stagger so request 1 is genuinely mid-flight when 2 arrives.
    time.sleep(0.15)
    t2 = threading.Thread(target=lambda: first_chunk_times.append(fire_and_time(t_start)))
    t2.start()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert len(first_chunk_times) == 2
    earliest, latest = sorted(first_chunk_times)
    # Both first chunks should arrive within a few hundred ms of each
    # other despite total generation taking 500 ms.  If the event loop
    # were blocked, the second would wait for the first to finish.
    assert latest < 0.6, (
        f"Second request's first chunk arrived at {latest:.3f}s -- "
        "looks like the event loop was blocked while the first "
        "request was generating."
    )


def test_sse_chunks_arrive_progressively(running_server):
    """The time between the first and last content delta should be at
    least roughly the number-of-tokens * delay.  If chunks were
    buffered until the whole generation finished, we would instead see
    all deltas arrive in a burst at the end."""
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "go"}],
        "stream": True,
        "max_tokens": 10,
    }

    deltas: list[tuple[float, str]] = []
    t_start = time.monotonic()

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", f"{running_server}/v1/chat/completions",
                           json=payload) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    deltas.append((time.monotonic() - t_start, delta["content"]))

    # 5 tokens * 100 ms = 500 ms minimum; allow some slack but require
    # that at least two deltas are separated by >= 50 ms.
    # If chunks were buffered, deltas would cluster near the end and
    # consecutive gaps would be ~0 ms.
    gaps = [deltas[i + 1][0] - deltas[i][0] for i in range(len(deltas) - 1)]
    assert len(deltas) >= 3, f"Got too few deltas: {deltas}"
    assert max(gaps) >= 0.05, (
        f"All deltas clustered together (max gap {max(gaps):.3f}s); "
        f"SSE is NOT streaming progressively.  Deltas: {deltas}"
    )
    # First delta should arrive well before the last.
    assert deltas[-1][0] - deltas[0][0] >= 0.2, (
        f"First-to-last delta gap only {deltas[-1][0] - deltas[0][0]:.3f}s"
    )
