"""Chunked prefill — verify the backend feeds the prompt suffix to the
model as a series of fixed-T chunks plus a T=1 residual, not one big
T=len(suffix) call.

This is what lets tinygrad's on-disk kernel cache actually hit across
requests: varying T per prompt-length compiles new kernels every time;
fixed T=CHUNK reuses the same compiled kernels forever.

Uses the same fake-model pattern as test_tinygrad_prefix_cache.py: we
install stubs for tinygrad.Tensor/UOp/TinyJit so no GPU is required and
every ``model.__call__`` records the (T, start_pos) pair it saw.
"""
from __future__ import annotations

import sys
import time
import types

import pytest


class _FakeModel:
    """Records every forward call's (T, start_pos) pair, returns a fake
    logits tensor so sampling can pick a scripted next token."""

    def __init__(self, script: list[int], max_context: int = 4096):
        self.max_context = max_context
        self.calls: list[tuple[int, int]] = []
        self._script = list(script)
        self._idx = 0

        class _Jit:
            def reset(self_inner):  # noqa: ARG002
                pass
        self.forward_jit = _Jit()

    def __call__(self, t, start_pos):
        T = t.shape[1]
        if hasattr(start_pos, "_bound_value"):
            start_pos = start_pos._bound_value
        self.calls.append((T, int(start_pos)))
        next_id = self._script[self._idx % len(self._script)]
        self._idx += 1

        class _Result:
            def tolist(self_inner):  # noqa: ARG002
                row = [0.0] * max(next_id + 1, 8)
                row[next_id] = 1.0
                return [row]
        return _Result()


class _FakeTensor:
    def __init__(self, data, dtype=None):  # noqa: ARG002
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
    fake_tg = types.ModuleType("tinygrad")
    fake_tg.Tensor = _FakeTensor
    fake_tg.UOp = _FakeUOp
    fake_tg.getenv = lambda name, default=1: default
    fake_tg.nn = types.SimpleNamespace()

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
    _install_tinygrad_stub(monkeypatch)

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
    return backend


def _run_request(backend, prompt_tokens, take=1):
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
    backend.cancel_generation()
    time.sleep(0.05)
    return out


# =================================================================== #
# Chunked-prefill shape contract                                      #
# =================================================================== #

class TestChunkedPrefillShape:
    def test_default_chunk_is_128(self, monkeypatch):
        """Default chunk size (EGG_PREFILL_CHUNK unset) is 128."""
        monkeypatch.delenv("EGG_PREFILL_CHUNK", raising=False)
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        # 130-token prompt → one chunk of 128 + residual of 2 tokens.
        _run_request(backend, list(range(1, 131)), take=1)
        prefill_calls = [c for c in fake.calls if c[1] != 130]  # all except decode step
        # Actually let's be explicit: prefill happens for all calls where
        # we're still feeding the prompt, i.e. start_pos < 130.
        prefill = [c for c in fake.calls if c[1] < 130]
        # Expect: [(128, 0), (1, 128), (1, 129)]
        assert prefill == [(128, 0), (1, 128), (1, 129)], prefill

    def test_custom_chunk_size(self, monkeypatch):
        """EGG_PREFILL_CHUNK=64 → chunks of 64."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "64")
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        _run_request(backend, list(range(1, 201)), take=1)  # 200 tokens
        prefill = [c for c in fake.calls if c[1] < 200]
        # 200 / 64 = 3 full chunks (192 tokens) + 8 residual
        # Expect: (64, 0), (64, 64), (64, 128), then 8x (1, start).
        assert prefill[0] == (64, 0)
        assert prefill[1] == (64, 64)
        assert prefill[2] == (64, 128)
        # Residual: (1, 192), (1, 193), ... (1, 199)
        assert prefill[3:] == [(1, 192 + i) for i in range(8)]

    def test_chunk_zero_disables_chunking(self, monkeypatch):
        """EGG_PREFILL_CHUNK=0 falls back to a single-shot prefill
        of T=len(new_suffix).  Same as the pre-chunking behaviour."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "0")
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        _run_request(backend, list(range(1, 101)), take=1)  # 100 tokens
        prefill = [c for c in fake.calls if c[1] < 100]
        assert prefill == [(100, 0)], prefill

    def test_short_prompt_below_chunk_size_uses_only_residual(self, monkeypatch):
        """A 5-token prompt with CHUNK=128 just runs 5 T=1 residual
        forwards -- no full chunks."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "128")
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        _run_request(backend, list(range(1, 6)), take=1)  # 5 tokens
        prefill = [c for c in fake.calls if c[1] < 5]
        # 5 x (T=1) at positions 0..4.
        assert prefill == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)], prefill

    def test_exact_chunk_multiple_no_residual(self, monkeypatch):
        """A prompt of length 256 with CHUNK=128 -> exactly 2 chunks, 0 residual."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "128")
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        _run_request(backend, list(range(1, 257)), take=1)  # 256 tokens
        prefill = [c for c in fake.calls if c[1] < 256]
        assert prefill == [(128, 0), (128, 128)], prefill


# =================================================================== #
# Prefix-cache interaction                                             #
# =================================================================== #

class TestChunkedPrefillWithPrefixCache:
    def test_long_suffix_uses_chunks(self, monkeypatch):
        """Long suffix (>= chunk size) goes through the chunked path."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "64")
        fake = _FakeModel(script=[100])
        backend = _make_loaded_backend(monkeypatch, fake)

        # Prime: 100-token prompt.  After take=1, the fake's producer
        # may race through the decode loop and append many tokens to
        # cache_tokens, but only the first 100 are guaranteed prompt.
        _run_request(backend, list(range(1, 101)), take=1)
        # Truncate cache_tokens to the known-good prompt prefix so the
        # second request's prefix match behaves predictably.  (In real
        # GPU-speed conditions, cancellation fires fast and the decode
        # loop rarely runs beyond what was yielded.)
        backend._cache_tokens = list(range(1, 101))
        fake.calls.clear()

        # New prompt: 100 cached + 150 new = 250 total.  cp=100 after
        # full match; no rollback because len(prompt) (250) != cp (100).
        # Suffix length = 150 starting at pos 100.
        prompt2 = list(range(1, 101)) + list(range(1000, 1150))  # 250 tokens
        _run_request(backend, prompt2, take=1)

        # 150 / 64 = 2 full chunks (128 tokens) + 22 residual.
        prefill = [c for c in fake.calls if c[1] < 250]
        assert prefill[0] == (64, 100), prefill[0]
        assert prefill[1] == (64, 164), prefill[1]
        # Residual: 22 tokens at 228..249.
        assert prefill[2:] == [(1, 228 + i) for i in range(22)], prefill[2:]


# =================================================================== #
# Correctness: yielded tokens + cache invariant                        #
# =================================================================== #

class TestChunkedPrefillStartPosTyping:
    """Regression: tinygrad's TransformerBlock._attention constructs
    the causal mask via triu over a ``(1, 1, T, start_pos+T)`` shape,
    and ``triu`` rejects symbolic shape dims.  So T>1 chunk calls MUST
    pass an int start_pos; only T=1 residual/decode calls may pass a
    UOp.  This test enforces that contract against a stricter fake
    model that blows up if T>1 sees a UOp.
    """

    def _strict_fake(self, script):
        class _StrictFakeModel(_FakeModel):
            def __call__(self_inner, t, start_pos):
                T = t.shape[1]
                is_uop = hasattr(start_pos, "_bound_value")
                if T > 1 and is_uop:
                    raise AssertionError(
                        f"T={T}>1 must use int start_pos (symbolic "
                        f"shapes break triu); got UOp"
                    )
                return super().__call__(t, start_pos)
        return _StrictFakeModel(script)

    def test_chunks_use_int_start_pos(self, monkeypatch):
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "64")
        fake = self._strict_fake(script=[42])
        backend = _make_loaded_backend(monkeypatch, fake)
        # 200-token prompt => 3 chunks of 64 (T>1) + 8 residual (T=1).
        # If any chunk call passes a UOp, the strict fake raises.
        result = _run_request(backend, list(range(1, 201)), take=1)
        assert result == [42], result

    def test_unchunked_fallback_uses_int_start_pos(self, monkeypatch):
        """EGG_PREFILL_CHUNK=0 path must also pass int start_pos."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "0")
        fake = self._strict_fake(script=[42])
        backend = _make_loaded_backend(monkeypatch, fake)
        result = _run_request(backend, list(range(1, 51)), take=1)
        assert result == [42], result


class TestChunkedPrefillCorrectness:
    def test_first_yielded_token_comes_from_last_prefill_logits(self, monkeypatch):
        """The token we yield FIRST after prefill must be the sampled
        prediction from the last prefill forward (the T=1 residual's
        logits, or the last chunk's if no residual)."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "128")
        # Script: model produces tokens in the order feeds happen.
        # For a 130-token prompt: chunk(128) returns scripted[0]=42,
        # then residual (1,128) returns scripted[1]=43, residual (1,129)
        # returns scripted[2]=44.  That last sample is the first yield.
        fake = _FakeModel(script=[42, 43, 44, 45, 46])
        backend = _make_loaded_backend(monkeypatch, fake)

        yielded = _run_request(backend, list(range(1, 131)), take=1)
        assert yielded == [44], yielded

    def test_cache_tokens_prefix_equals_prompt_after_prefill(self, monkeypatch):
        """_cache_tokens[:len(prompt)] must equal prompt_list after
        prefill completes.  The fake model runs near-instantly so
        the decode loop may extend cache_tokens beyond len(prompt)
        before cancellation fires -- we only verify the prompt-prefix
        part, which is the invariant needed for prefix-cache hits."""
        monkeypatch.setenv("EGG_PREFILL_CHUNK", "64")
        fake = _FakeModel(script=[99])
        backend = _make_loaded_backend(monkeypatch, fake)

        prompt = list(range(1, 201))
        _run_request(backend, prompt, take=1)
        assert backend._cache_tokens[:len(prompt)] == prompt
