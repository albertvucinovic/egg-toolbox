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
