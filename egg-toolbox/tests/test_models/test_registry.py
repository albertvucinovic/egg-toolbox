"""Registry + dispatch contract for egg_toolbox.models.

These tests don't touch a real GGUF -- they verify the wiring:

- @register(...) puts the factory in the dispatch map
- load_from_gguf picks the right factory based on general.architecture
- Unknown archs fall through to the fallback
- Missing general.architecture is a clear error, not a silent fallback
"""
from __future__ import annotations

import pytest

from egg_toolbox.models import base as models_base
from egg_toolbox.models import (
    Architecture,
    load_from_gguf,
    register,
    registered_architectures,
)


def test_llama_qwen_registered():
    """The llama module self-registers for the standard families."""
    names = registered_architectures()
    assert "llama" in names
    assert "qwen2" in names
    assert "qwen3" in names


def test_register_requires_from_gguf_kv():
    """A class missing the factory classmethod can't be registered."""

    with pytest.raises(TypeError, match="from_gguf_kv"):

        @register("__bogus__")
        class _Bogus(Architecture):
            def forward(self, tokens, start_pos):  # pragma: no cover
                return None


def test_register_preserves_existing_registry():
    """Registering a test arch doesn't evict the real ones."""
    before = dict(models_base._REGISTRY)

    @register("__test_arch_preserves__")
    class _Test(Architecture):
        @classmethod
        def from_gguf_kv(cls, kv, state_dict, **kw):  # pragma: no cover
            return None, kv

        def forward(self, tokens, start_pos):  # pragma: no cover
            return None

    try:
        assert "llama" in models_base._REGISTRY
        assert "__test_arch_preserves__" in models_base._REGISTRY
    finally:
        del models_base._REGISTRY["__test_arch_preserves__"]
        assert models_base._REGISTRY == before


def test_dispatch_picks_registered_factory(monkeypatch):
    """load_from_gguf routes to the factory whose arch string matches."""
    called = {}

    @register("__test_dispatch_pick__")
    class _Picked(Architecture):
        @classmethod
        def from_gguf_kv(cls, kv, state_dict, **kw):
            called["args"] = (kv.get("general.architecture"), kw)
            return "PICKED", kv

        def forward(self, tokens, start_pos):  # pragma: no cover
            return None

    try:
        def fake_gguf_load(_tensor):
            return ({"general.architecture": "__test_dispatch_pick__"}, {})

        # Patch tinygrad.nn.state.gguf_load so we don't need a real file.
        import tinygrad.nn.state as state_mod
        monkeypatch.setattr(state_mod, "gguf_load", fake_gguf_load)

        # Also stub Tensor(Path(...)).to(None) so we don't try to mmap.
        import tinygrad
        class _FakeTensor:
            def __init__(self, *a, **k): pass
            def to(self, _): return self
        monkeypatch.setattr(tinygrad, "Tensor", _FakeTensor)

        result, kv = load_from_gguf("ignored.gguf", max_context=42)
        assert result == "PICKED"
        assert called["args"] == ("__test_dispatch_pick__", {"max_context": 42})
    finally:
        del models_base._REGISTRY["__test_dispatch_pick__"]


def test_dispatch_falls_back_for_unknown_arch(monkeypatch):
    """Unknown arch names route to the fallback class (LlamaArchitecture)."""
    seen_arch = []

    # Swap in a fake fallback factory so we can assert it was invoked.
    original_factory = models_base._REGISTRY["llama"]

    def fake_fallback(kv, state_dict, **kw):
        seen_arch.append(kv.get("general.architecture"))
        return "FALLBACK", kv

    models_base._REGISTRY["llama"] = fake_fallback
    try:
        def fake_gguf_load(_tensor):
            return ({"general.architecture": "nobody_registered_this"}, {})

        import tinygrad.nn.state as state_mod
        monkeypatch.setattr(state_mod, "gguf_load", fake_gguf_load)

        import tinygrad
        class _FakeTensor:
            def __init__(self, *a, **k): pass
            def to(self, _): return self
        monkeypatch.setattr(tinygrad, "Tensor", _FakeTensor)

        result, _ = load_from_gguf("ignored.gguf")
        assert result == "FALLBACK"
        assert seen_arch == ["nobody_registered_this"]
    finally:
        models_base._REGISTRY["llama"] = original_factory


def test_missing_arch_raises(monkeypatch):
    """A GGUF without general.architecture is a hard error, not a silent
    fallback -- that key is mandatory in a well-formed GGUF."""
    def fake_gguf_load(_tensor):
        return ({}, {})

    import tinygrad.nn.state as state_mod
    monkeypatch.setattr(state_mod, "gguf_load", fake_gguf_load)

    import tinygrad
    class _FakeTensor:
        def __init__(self, *a, **k): pass
        def to(self, _): return self
    monkeypatch.setattr(tinygrad, "Tensor", _FakeTensor)

    with pytest.raises(ValueError, match="general.architecture"):
        load_from_gguf("ignored.gguf")
