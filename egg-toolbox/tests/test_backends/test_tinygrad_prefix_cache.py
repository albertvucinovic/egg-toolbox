"""Regression: TinygradBackend must reuse tinygrad's per-block
cache_kv tensor across sequential generate_tokens calls when the new
prompt shares a token prefix with the previous one.  Without this,
agentic chat re-prefills the entire conversation every turn.

These tests use a fake model that records every (T, start_pos) pair
passed to __call__.  That's the *contract* of prefix caching:

  First request, prompt of length P  -> one prefill at (P, 0),
                                        then (1, P), (1, P+1), ...

  Second request, prompt = first_prompt + first_response + tail ->
                                        one prefill at (len(tail), P'),
                                        then (1, P'+len(tail)), ...

If the second prefill's start_pos is 0, prefix caching is broken.
"""
from __future__ import annotations

import sys
import types

import pytest


class _FakeModel:
    """Stand-in for tinygrad.apps.llm.Transformer.

    Records every forward call.  __call__ returns the next token id
    from a scripted stream so tests can assert deterministic outputs.
    """

    def __init__(self, script: list[int], max_context: int = 1024):
        self.max_context = max_context
        self.calls: list[tuple[int, int]] = []  # (T, start_pos)
        self._script = list(script)
        self._idx = 0

        class _Jit:
            def reset(self_inner):  # noqa: ARG002
                pass

        self.forward_jit = _Jit()

    def __call__(self, t, start_pos):
        # t is either a list-like Tensor or a _FakeTensor with .shape.
        # We use the shape set by _FakeTensor below.
        T = t.shape[1]
        # start_pos may be a UOp-bound object; the fake accepts either
        # an int or anything with a ._bound_value attribute.
        if hasattr(start_pos, "_bound_value"):
            start_pos = start_pos._bound_value
        self.calls.append((T, int(start_pos)))

        next_id = self._script[self._idx % len(self._script)]
        self._idx += 1

        class _Result:
            def tolist(self_inner):  # noqa: ARG002
                # One-hot logits so argmax picks next_id.
                row = [0.0] * max(next_id + 1, 8)
                row[next_id] = 1.0
                return [row]

        return _Result()


class _FakeTensor:
    """Minimal Tensor stand-in with a .shape tuple so fake _model can
    read T = shape[1].  tinygrad's real Tensor has a richer API but
    prefix-cache logic only needs shape[1] (sequence length)."""

    def __init__(self, data, dtype=None):  # noqa: ARG002
        # data is [[a, b, c, ...]] -- we only care about the length.
        seq = data[0] if data else []
        self.shape = (1, len(seq))
        self._data = list(seq)


class _FakeUOp:
    @staticmethod
    def variable(name, lo, hi):  # noqa: ARG004
        return _FakeUOpVar()


class _FakeUOpVar:
    def bind(self, v):
        class _Bound:
            _bound_value = v
        return _Bound()


def _install_tinygrad_stub(monkeypatch):
    """Install fake tinygrad top-level module so the backend can
    ``from tinygrad import Tensor, UOp, getenv`` inside the
    prefix-cache path without a real install."""
    fake_tg = types.ModuleType("tinygrad")
    fake_tg.Tensor = _FakeTensor
    fake_tg.UOp = _FakeUOp
    fake_tg.getenv = lambda name, default=1: default
    fake_tg.nn = types.SimpleNamespace()  # for any stray access
    class _FakeTinyJit:
        def __init__(self, fn): self.fn = fn
        def reset(self): pass
        def __call__(self, *a, **k): return self.fn(*a, **k)
    fake_tg.TinyJit = _FakeTinyJit
    monkeypatch.setitem(sys.modules, "tinygrad", fake_tg)

    fake_apps_llm = types.ModuleType("tinygrad.apps.llm")

    class _FakeSimpleTokenizer:
        @staticmethod
        def from_gguf_kv(kv):  # noqa: ARG004
            return _FakeSimpleTokenizer()
        def encode(self, t):  # noqa: ARG002
            return [1]
        def decode(self, ids):  # noqa: ARG002
            return ""
    fake_apps_llm.SimpleTokenizer = _FakeSimpleTokenizer
    monkeypatch.setitem(sys.modules, "tinygrad.apps.llm", fake_apps_llm)


def _make_loaded_backend(monkeypatch, fake_model):
    """Build a TinygradBackend with _model already set to ``fake_model``
    so we don't have to run the real load path."""
    _install_tinygrad_stub(monkeypatch)

    from egg_toolbox.backends import tinygrad as tg_backend
    from egg_toolbox.backends.tinygrad import TinygradBackend, TinygradTokenizer

    backend = TinygradBackend()
    backend._model = fake_model
    backend._tokenizer = TinygradTokenizer(
        inner=types.SimpleNamespace(
            encode=lambda t: [1],
            decode=lambda ids: "",
        ),
        eos_id=0,
    )
    backend._chat_template = ""
    return backend, tg_backend


def _run_request(backend, prompt_tokens, take=5):
    """Run generate_tokens and consume ``take`` tokens."""
    from egg_toolbox.types import CompiledRequest, SamplingParams

    req = CompiledRequest(
        prompt_tokens=tuple(prompt_tokens),
        sampling=SamplingParams(),
        stop_strings=(),
        stop_token_ids=(),
    )
    out = []
    for tid in backend.generate_tokens(req):
        out.append(tid)
        if len(out) >= take:
            break
    # Drop into the generator's finally block so cancellation fires
    # and the worker thread exits cleanly between requests.
    import time
    backend.cancel_generation()
    time.sleep(0.05)
    return out


def test_first_request_full_prefill_then_incremental(monkeypatch):
    """Contract: first request prefills the whole prompt at start_pos=0,
    then decodes one token at a time with start_pos advancing."""
    script = [10, 11, 12, 13, 14, 15]
    fake = _FakeModel(script)
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
    got = _run_request(backend, prompt, take=4)

    assert got == script[:4]
    # First call: full prefill with T=8, start_pos=0.
    assert fake.calls[0] == (8, 0), f"expected prefill (8,0), got {fake.calls[0]}"
    # Following calls: T=1 decode at positions 8, 9, 10.
    assert fake.calls[1:4] == [(1, 8), (1, 9), (1, 10)], fake.calls


def test_second_request_reuses_common_prefix(monkeypatch):
    """The *whole point* of prefix caching: if the second request's
    prompt starts with the first request's prompt (plus maybe some of
    the first response and a new user turn), the second prefill must
    be at start_pos = len(common_prefix), not 0."""
    fake = _FakeModel(script=[100, 101, 102, 103, 104, 105, 106, 107])
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    # Request 1: prompt len 5, consume 3 tokens.
    prompt1 = [1, 2, 3, 4, 5]
    out1 = _run_request(backend, prompt1, take=3)
    assert out1 == [100, 101, 102]

    calls_after_req1 = list(fake.calls)
    fake.calls.clear()

    # Request 2: prompt1 + the 3 tokens we yielded + a new 2-token tail.
    # Everything up to and including the yielded tokens is what we fed
    # the model -- so the common prefix with cache is len(prompt1) + 3 = 8.
    prompt2 = prompt1 + out1 + [99, 98]  # total length 10
    out2 = _run_request(backend, prompt2, take=2)

    # Check request 1 did what we expected.
    assert calls_after_req1[0] == (5, 0), calls_after_req1

    # Request 2's prefill: the new 2-token tail at start_pos=8.
    # (Or, with cp -= 1 edge handling when prompt fully covered, a
    # 3-token refeed at start_pos=7 -- but cp<len(prompt) here so no
    # rollback, so strictly (2, 8).)
    assert fake.calls[0] == (2, 8), (
        f"expected second prefill at (2,8), got {fake.calls[0]} "
        f"-- prefix cache broken, re-prefilling from 0?"
    )


def test_unrelated_prompt_evicts_cache(monkeypatch):
    """When the new prompt shares no prefix with the cached sequence,
    we must re-prefill from 0.  The cached tokens get overwritten
    implicitly (cache_kv[0..N) is reassigned)."""
    fake = _FakeModel(script=[50, 51, 52, 53])
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    _run_request(backend, [1, 2, 3], take=2)
    fake.calls.clear()

    _run_request(backend, [99, 98, 97, 96], take=1)
    # Divergence at position 0 -> full prefill at start_pos=0.
    assert fake.calls[0] == (4, 0), fake.calls


def test_prompt_fully_covered_rolls_back_one_token(monkeypatch):
    """Edge case: the new prompt is a proper prefix of what's already
    cached.  We still need at least one token as input to produce
    logits, so cp is rolled back by 1 and we re-feed the last token."""
    fake = _FakeModel(script=[200, 201, 202])
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    # Prime cache with prompt + 2 generated tokens = [1,2,3] + [200,201].
    _run_request(backend, [1, 2, 3], take=2)
    fake.calls.clear()

    # New prompt exactly matches the cached sequence [1,2,3,200,201].
    _run_request(backend, [1, 2, 3, 200, 201], take=1)
    # cp starts at 5 (full match) -> rolled back to 4 -> refeed token 201.
    assert fake.calls[0] == (1, 4), (
        f"expected (1,4) rollback refeed, got {fake.calls[0]}"
    )


def test_mid_forward_exception_invalidates_cache(monkeypatch):
    """If the forward pass raises, the cache might be in an inconsistent
    state (some blocks' cache_kv written, others not).  The backend
    must invalidate _cache_tokens so the next request does a full
    prefill from 0, not a partial reuse of corrupt state."""
    boom = RuntimeError("synthetic GPU failure")

    class _BrokenModel:
        max_context = 1024
        calls: list[tuple[int, int]] = []

        class _Jit:
            def reset(self_inner):  # noqa: ARG002
                pass
        forward_jit = _Jit()

        def __call__(self, t, start_pos):
            self.calls.append((t.shape[1],
                               start_pos if isinstance(start_pos, int)
                               else getattr(start_pos, "_bound_value", -1)))
            raise boom

    fake = _BrokenModel()
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    from egg_toolbox.types import CompiledRequest, SamplingParams
    req = CompiledRequest(
        prompt_tokens=(1, 2, 3, 4),
        sampling=SamplingParams(),
        stop_strings=(),
        stop_token_ids=(),
    )
    with pytest.raises(RuntimeError, match="synthetic GPU failure"):
        for _ in backend.generate_tokens(req):
            pass

    assert backend._cache_tokens == [], (
        "backend must clear _cache_tokens after a forward exception; "
        f"got {backend._cache_tokens!r}"
    )

    # Next request: must do full prefill from 0 even though the failed
    # request "fed" some tokens.  Swap in a working model and verify.
    fake.calls.clear()
    working = _FakeModel(script=[7])

    class _ShimModel:
        max_context = 1024
        class _Jit:
            def reset(self_inner):  # noqa: ARG002
                pass
        forward_jit = _Jit()
        calls: list[tuple[int, int]] = []
        def __call__(self, t, start_pos):
            self.calls.append((t.shape[1],
                               start_pos if isinstance(start_pos, int)
                               else getattr(start_pos, "_bound_value", -1)))
            class _R:
                def tolist(self_inner):  # noqa: ARG002
                    row = [0.0] * 8
                    row[7] = 1.0
                    return [row]
            return _R()

    shim = _ShimModel()
    backend._model = shim
    _run_request(backend, [1, 2, 3, 4, 5], take=1)
    assert shim.calls[0] == (5, 0), (
        f"after invalidation, next request must prefill from 0; "
        f"got {shim.calls[0]}"
    )


def test_prefix_cache_can_be_disabled_via_env(monkeypatch):
    """EGG_PREFIX_CACHE=0 forces full prefill every request -- escape
    hatch for debugging or known-bad cases."""
    monkeypatch.setenv("EGG_PREFIX_CACHE", "0")
    fake = _FakeModel(script=[1, 2, 3, 4])
    backend, _ = _make_loaded_backend(monkeypatch, fake)

    _run_request(backend, [10, 20, 30], take=1)
    fake.calls.clear()
    _run_request(backend, [10, 20, 30, 1, 99], take=1)  # shared prefix of 4
    # With caching disabled, second prefill is full length 5 at pos 0.
    assert fake.calls[0] == (5, 0), fake.calls
