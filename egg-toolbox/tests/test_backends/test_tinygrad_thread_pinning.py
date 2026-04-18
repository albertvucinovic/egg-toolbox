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

        class _Jit:
            def reset(self):
                pass

        class FakeModel:
            max_context = 1024
            forward_jit = _Jit()
            def __call__(self, t, start_pos):
                thread_ids.append(("generate", threading.get_ident()))
                class _R:
                    def tolist(self_inner):
                        # One-hot logits -> argmax picks token 1.
                        return [[0.0, 1.0]]
                return _R()
        return FakeModel(), FakeKV()

    import egg_toolbox.models as models_pkg
    monkeypatch.setattr(
        models_pkg, "load_from_gguf",
        lambda path, **kw: fake_from_gguf(path, **kw),
    )

    # Monkey-patch tinygrad imports inside _do_load_model to stubs.
    import sys
    import types

    fake_tinygrad = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, data=None, *args, **kwargs):
            seq = data[0] if (isinstance(data, list) and data) else []
            self.shape = (1, len(seq) if isinstance(seq, list) else 1)
    class _FakeUOpVar:
        def bind(self, v):
            return v
    class _FakeUOp:
        @staticmethod
        def variable(name, lo, hi):
            return _FakeUOpVar()
    fake_tinygrad.Tensor = _FakeTensor
    fake_tinygrad.UOp = _FakeUOp
    fake_tinygrad.getenv = lambda name, default=1: default
    class _FakeTinyJit:
        def __init__(self, fn): self.fn = fn
        def reset(self): pass
        def __call__(self, *a, **k): return self.fn(*a, **k)
    fake_tinygrad.TinyJit = _FakeTinyJit
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

    # Consume a few tokens from generate_tokens, then stop.  The new
    # forward-driven path yields until max_context, so we break out
    # ourselves and let cancellation fire the producer's exit.
    tokens = []
    for tid in backend.generate_tokens(req):
        tokens.append(tid)
        if len(tokens) >= 3:
            break
    backend.cancel_generation()
    assert tokens == [1, 1, 1]

    # Load AND at least one generate forward must have run on the
    # same worker thread, which is NOT the caller's (test runs on
    # main).
    assert thread_ids[0][0] == "load"
    assert any(kind == "generate" for kind, _ in thread_ids)
    load_tid = thread_ids[0][1]
    gen_tids = [tid for kind, tid in thread_ids if kind == "generate"]
    assert all(tid == load_tid for tid in gen_tids), (
        f"Load and generate ran on different threads: {thread_ids}"
    )
    assert load_tid != threading.get_ident(), (
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

        class _Jit:
            def reset(self):
                pass

        class FakeModel:
            max_context = 10_000
            forward_jit = _Jit()
            _counter = [0]
            def __call__(self, t, start_pos):
                # Simulate tinygrad's per-token GPU cost (~200 tok/s)
                # so the producer doesn't get thousands of tokens
                # ahead of the consumer -- the test's purpose is to
                # verify cancellation, not queue-throughput.
                import time as _t
                self._counter[0] += 1
                tokens_produced.append(self._counter[0])
                _t.sleep(0.005)   # 5 ms per forward
                val = self._counter[0]
                class _R:
                    def tolist(self_inner):
                        # Dense one-hot logits: argmax picks `val`.
                        row = [0.0] * max(val + 1, 8)
                        row[val] = 1.0
                        return [row]
                return _R()
        return FakeModel(), FakeKV()

    import egg_toolbox.models as models_pkg
    monkeypatch.setattr(
        models_pkg, "load_from_gguf",
        lambda path, **kw: fake_from_gguf(path, **kw),
    )

    import sys, types
    fake_tg = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, data=None, *args, **kwargs):
            seq = data[0] if (isinstance(data, list) and data) else []
            self.shape = (1, len(seq) if isinstance(seq, list) else 1)
    class _FakeUOpVar:
        def bind(self, v): return v
    class _FakeUOp:
        @staticmethod
        def variable(name, lo, hi): return _FakeUOpVar()
    fake_tg.Tensor = _FakeTensor
    fake_tg.UOp = _FakeUOp
    fake_tg.getenv = lambda name, default=1: default
    class _FakeTinyJit:
        def __init__(self, fn): self.fn = fn
        def reset(self): pass
        def __call__(self, *a, **k): return self.fn(*a, **k)
    fake_tg.TinyJit = _FakeTinyJit
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

    # This is the exact shape of what the orchestrator's finally
    # block does: call cancel_generation() on the backend.  Crucially
    # we do NOT call gen.close() here -- that's cross-thread on the
    # iterating thread, Python raises "generator already executing",
    # and the previous implementation silently swallowed that, so
    # cancellation never fired and tinygrad ran to max_context.
    backend.cancel_generation()

    # Wait for the worker to notice the flag and exit.
    import time
    time.sleep(0.25)
    produced_after_cancel = len(tokens_produced)
    time.sleep(0.25)
    assert len(tokens_produced) == produced_after_cancel, (
        f"Backend kept producing tokens after cancel(): "
        f"{produced_after_cancel} -> {len(tokens_produced)}"
    )
    assert produced_after_cancel < 100, (
        f"Backend produced {produced_after_cancel} tokens but caller "
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

        class _Jit:
            def reset(self):
                pass

        class FakeModel:
            max_context = 1024
            forward_jit = _Jit()
            def __call__(self, t, start_pos):
                generate_thread_ids.append(threading.get_ident())
                class _R:
                    def tolist(self_inner):
                        return [[0.0, 1.0]]
                return _R()
        return FakeModel(), FakeKV()

    import egg_toolbox.models as models_pkg
    monkeypatch.setattr(
        models_pkg, "load_from_gguf",
        lambda path, **kw: fake_from_gguf(path, **kw),
    )

    import sys, types
    fake_tinygrad = types.ModuleType("tinygrad")
    class _FakeTensor:
        def __init__(self, data=None, *args, **kwargs):
            seq = data[0] if (isinstance(data, list) and data) else []
            self.shape = (1, len(seq) if isinstance(seq, list) else 1)
    class _FakeUOpVar:
        def bind(self, v): return v
    class _FakeUOp:
        @staticmethod
        def variable(name, lo, hi): return _FakeUOpVar()
    fake_tinygrad.Tensor = _FakeTensor
    fake_tinygrad.UOp = _FakeUOp
    fake_tinygrad.getenv = lambda name, default=1: default
    class _FakeTinyJit:
        def __init__(self, fn): self.fn = fn
        def reset(self): pass
        def __call__(self, *a, **k): return self.fn(*a, **k)
    fake_tinygrad.TinyJit = _FakeTinyJit
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
    for i in range(3):
        # Different prompt per request so prefix-cache path still
        # executes the forward for at least one token each time.
        req = CompiledRequest(prompt_tokens=(i + 10, i + 11),
                             sampling=SamplingParams(),
                             stop_strings=(), stop_token_ids=())
        for _ in backend.generate_tokens(req):
            break  # one token is enough to assert thread identity
        backend.cancel_generation()

    assert len(generate_thread_ids) >= 3
    assert len(set(generate_thread_ids)) == 1, (
        f"Three generate calls ran on different threads: {generate_thread_ids}"
    )
