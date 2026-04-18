"""Regression: TinygradBackend must pin ALL tinygrad ops (model load
+ every generate_tokens) to a single worker thread.  Rationale: the
tinygrad kernel-compile cache opens one SQLite connection and
sqlite3 rejects cross-thread use.

We don't need a real GGUF or tinygrad install to test this -- we
monkey-patch the tinygrad-touching bits with a stand-in that
records which thread each call ran on.
"""
from __future__ import annotations

import threading

import pytest


def test_load_and_generate_run_on_same_thread(monkeypatch):
    """Simulates the traceback the user saw: load creates the
    SQLite connection on thread X, first request runs on thread Y,
    sqlite3 raises ProgrammingError.  With the backend executor, X
    and Y are the same thread and the connection is reusable."""
    from egg_toolbox.backends import tinygrad as tg_backend

    thread_ids: list[tuple[str, int]] = []

    def fake_from_gguf(gguf, **kwargs):
        thread_ids.append(("load", threading.get_ident()))

        class FakeKV(dict):
            def __getitem__(self, key):
                defaults = {
                    "general.architecture": "llama",
                    "tokenizer.ggml.tokens": ["<eos>"] * 100,
                    "tokenizer.ggml.eos_token_id": 0,
                }
                return defaults.get(key, 0)
            def get(self, key, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default

        class FakeModel:
            def generate(self, tokens):
                thread_ids.append(("generate", threading.get_ident()))
                for _ in range(3):
                    yield 1
        return FakeModel(), FakeKV()

    monkeypatch.setattr(tg_backend, "_from_gguf_with_qkv_bias", fake_from_gguf)

    # Monkey-patch tinygrad imports inside _do_load_model to stubs.
    import sys
    import types

    fake_tinygrad = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, *args, **kwargs):
            pass
    fake_tinygrad.Tensor = _FakeTensor
    monkeypatch.setitem(sys.modules, "tinygrad", fake_tinygrad)

    fake_apps_llm = types.ModuleType("tinygrad.apps.llm")
    class _FakeSimpleTokenizer:
        def __init__(self):
            pass
        @staticmethod
        def from_gguf_kv(kv):
            return _FakeSimpleTokenizer()
        def encode(self, t): return [1]
        def decode(self, ids): return ""
    fake_apps_llm.SimpleTokenizer = _FakeSimpleTokenizer
    monkeypatch.setitem(sys.modules, "tinygrad.apps.llm", fake_apps_llm)

    # Stub ChatTemplate.from_gguf to avoid touching the disk.
    import egg_toolbox.template as tpl_mod
    original_from_gguf = tpl_mod.ChatTemplate.from_gguf
    class _FakeCT:
        source = "{% for m in messages %}{{ m.content }}{% endfor %}"
    monkeypatch.setattr(tpl_mod.ChatTemplate, "from_gguf",
                        staticmethod(lambda p: _FakeCT()))

    backend = tg_backend.TinygradBackend()
    backend.load_model("fake.gguf")

    from egg_toolbox.types import CompiledRequest, SamplingParams
    req = CompiledRequest(prompt_tokens=(1, 2, 3), sampling=SamplingParams(),
                         stop_strings=(), stop_token_ids=())

    # Consume a few tokens from generate_tokens.
    tokens = list(backend.generate_tokens(req))
    assert len(tokens) == 3

    # Load AND generate must have run on the same worker thread,
    # which is NOT the caller's thread (the test runs on main).
    assert len(thread_ids) == 2
    assert thread_ids[0][0] == "load"
    assert thread_ids[1][0] == "generate"
    assert thread_ids[0][1] == thread_ids[1][1], (
        f"Load and generate ran on different threads: {thread_ids}"
    )
    assert thread_ids[0][1] != threading.get_ident(), (
        "Load ran on the caller's thread -- not pinned to a backend "
        "worker thread."
    )


def test_generate_tokens_stops_backend_when_caller_stops_iterating(monkeypatch):
    """Regression: when the orchestrator breaks out of its
    generate_tokens iteration (stop string matched, EOS, tool_call
    complete), the backend worker must stop producing tokens.
    Otherwise tinygrad keeps running forward passes for tokens
    nobody reads, locking the backend's single-thread executor for
    minutes and causing the 'stuck after tool_call' loop.
    """
    from egg_toolbox.backends import tinygrad as tg_backend

    tokens_produced: list[int] = []
    tokens_yielded_out = 0

    def fake_from_gguf(gguf, **kwargs):
        class FakeKV(dict):
            def get(self, key, default=None):
                return {
                    "general.architecture": "llama",
                    "tokenizer.ggml.tokens": ["<eos>"] * 100,
                    "tokenizer.ggml.eos_token_id": 0,
                }.get(key, default)
            def __getitem__(self, key):
                return self.get(key, 0)

        class FakeModel:
            def generate(self, prompt_tokens):
                # Simulate tinygrad's per-token GPU cost (~1 tok/s)
                # so the producer doesn't get thousands of tokens
                # ahead of the consumer -- the test's purpose is to
                # verify cancellation, not queue-throughput.
                import time as _t
                i = 0
                while i < 10_000:
                    i += 1
                    tokens_produced.append(i)
                    _t.sleep(0.005)   # 5 ms per token
                    yield i
        return FakeModel(), FakeKV()

    monkeypatch.setattr(tg_backend, "_from_gguf_with_qkv_bias", fake_from_gguf)

    import sys, types
    fake_tg = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, *args, **kwargs): pass
    fake_tg.Tensor = _FakeTensor
    monkeypatch.setitem(sys.modules, "tinygrad", fake_tg)

    fake_llm = types.ModuleType("tinygrad.apps.llm")
    class _FakeSimpleTokenizer:
        @staticmethod
        def from_gguf_kv(kv): return _FakeSimpleTokenizer()
    fake_llm.SimpleTokenizer = _FakeSimpleTokenizer
    monkeypatch.setitem(sys.modules, "tinygrad.apps.llm", fake_llm)

    import egg_toolbox.template as tpl_mod
    class _FakeCT:
        source = ""
    monkeypatch.setattr(tpl_mod.ChatTemplate, "from_gguf",
                        staticmethod(lambda p: _FakeCT()))

    backend = tg_backend.TinygradBackend()
    backend.load_model("fake.gguf")

    from egg_toolbox.types import CompiledRequest, SamplingParams
    req = CompiledRequest(prompt_tokens=(1,), sampling=SamplingParams(),
                         stop_strings=(), stop_token_ids=())

    # Simulate the orchestrator consuming 5 tokens, then breaking out.
    gen = backend.generate_tokens(req)
    nonlocal_count = 0
    for tid in gen:
        nonlocal_count += 1
        if nonlocal_count >= 5:
            break

    # Explicitly close (orchestrator does this too via finally).
    gen.close()

    # Wait a moment for the worker to notice the cancellation flag
    # and exit.  If cancellation is broken, tokens_produced keeps
    # growing toward 100_000.
    import time
    time.sleep(0.25)
    produced_after_close = len(tokens_produced)
    time.sleep(0.25)
    # No further progress after a full second.
    assert len(tokens_produced) == produced_after_close, (
        f"Backend kept producing tokens after close(): {produced_after_close} -> {len(tokens_produced)}"
    )
    # Some slack: within ~a dozen extra tokens due to the check
    # happening *after* each yield.
    assert produced_after_close < 100, (
        f"Backend produced {produced_after_close} tokens but caller "
        "only asked for 5 -- cancellation flag isn't being honoured."
    )


def test_multiple_generate_calls_share_backend_thread(monkeypatch):
    """Each generate_tokens call must land on the SAME backend
    worker thread so the sqlite connection opened during load is
    reusable."""
    from egg_toolbox.backends import tinygrad as tg_backend

    generate_thread_ids: list[int] = []

    def fake_from_gguf(gguf, **kwargs):
        class FakeKV(dict):
            def get(self, key, default=None):
                return {
                    "general.architecture": "llama",
                    "tokenizer.ggml.tokens": ["<eos>"] * 100,
                    "tokenizer.ggml.eos_token_id": 0,
                }.get(key, default)
            def __getitem__(self, key):
                return self.get(key, 0)

        class FakeModel:
            def generate(self, tokens):
                generate_thread_ids.append(threading.get_ident())
                yield 1
        return FakeModel(), FakeKV()

    monkeypatch.setattr(tg_backend, "_from_gguf_with_qkv_bias", fake_from_gguf)

    import sys, types
    fake_tinygrad = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, *args, **kwargs): pass
    fake_tinygrad.Tensor = _FakeTensor
    monkeypatch.setitem(sys.modules, "tinygrad", fake_tinygrad)

    fake_apps_llm = types.ModuleType("tinygrad.apps.llm")
    class _FakeSimpleTokenizer:
        @staticmethod
        def from_gguf_kv(kv): return _FakeSimpleTokenizer()
        def encode(self, t): return [1]
        def decode(self, ids): return ""
    fake_apps_llm.SimpleTokenizer = _FakeSimpleTokenizer
    monkeypatch.setitem(sys.modules, "tinygrad.apps.llm", fake_apps_llm)

    import egg_toolbox.template as tpl_mod
    class _FakeCT:
        source = ""
    monkeypatch.setattr(tpl_mod.ChatTemplate, "from_gguf",
                        staticmethod(lambda p: _FakeCT()))

    backend = tg_backend.TinygradBackend()
    backend.load_model("fake.gguf")

    from egg_toolbox.types import CompiledRequest, SamplingParams
    for _ in range(3):
        req = CompiledRequest(prompt_tokens=(1,), sampling=SamplingParams(),
                             stop_strings=(), stop_token_ids=())
        list(backend.generate_tokens(req))

    assert len(generate_thread_ids) == 3
    assert len(set(generate_thread_ids)) == 1, (
        f"Three generate calls ran on different threads: {generate_thread_ids}"
    )
