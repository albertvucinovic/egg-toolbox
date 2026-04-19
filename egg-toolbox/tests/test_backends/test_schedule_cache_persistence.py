"""Integration tests: tinygrad's schedule_cache round-trips through
pickle and through our save/load helpers.

These tests use real tinygrad (no fakes) because the whole point is
to verify that entries produced by a real scheduling pass survive
serialization and get reused on the next forward.  No GPU required --
tinygrad's default backend (CLANG/C) works fine for small ops.

If any of these tests fail, it almost certainly means tinygrad's
schedule-cache internals changed in a way that breaks our pickle
roundtrip.  Fix: either update our helpers to match the new format,
or disable EGG_SCHEDULE_CACHE until tinygrad itself ships
schedule_cache persistence (workspace-g71).
"""
from __future__ import annotations

import os
import pickle

import pytest


def _with_clean_cache():
    """Import tinygrad and clear schedule_cache so each test starts
    from a known state.  Returns the module-level dict reference."""
    from tinygrad.engine.schedule import schedule_cache
    schedule_cache.clear()
    return schedule_cache


def test_schedule_cache_entry_is_picklable():
    """A real schedule_cache entry produced by a simple forward must
    survive a pickle round-trip."""
    from tinygrad import Tensor

    sc = _with_clean_cache()
    (Tensor([1.0, 2.0, 3.0]) * 2 + 1).realize()
    assert len(sc) >= 1, "forward should populate schedule_cache"

    # Key + value must be picklable.
    snapshot = dict(sc)
    blob = pickle.dumps(snapshot)
    restored = pickle.loads(blob)
    assert restored.keys() == snapshot.keys()
    # Value types should match.
    for k in snapshot:
        assert type(restored[k]) is type(snapshot[k])


def test_restored_entry_serves_a_matching_forward():
    """The key correctness property: after clearing schedule_cache
    and restoring it from a pickle snapshot, an identical forward
    should HIT the cache (not grow the dict)."""
    from tinygrad import Tensor

    sc = _with_clean_cache()
    # First forward -- populate.
    (Tensor([1.0, 2.0, 3.0]) * 2 + 1).realize()
    baseline_size = len(sc)
    assert baseline_size >= 1

    # Snapshot + clear + restore.
    snapshot = pickle.dumps(dict(sc))
    sc.clear()
    assert len(sc) == 0
    sc.update(pickle.loads(snapshot))
    assert len(sc) == baseline_size

    # Identical forward on different tensor values.  Shape + op
    # structure match -> same cache key -> should hit.
    (Tensor([10.0, 20.0, 30.0]) * 2 + 1).realize()

    # If the restored entry served the hit, cache size is unchanged.
    # If the restored entry was ignored, size grows by one (fresh
    # schedule for this forward).
    assert len(sc) == baseline_size, (
        f"restored entry failed to serve a matching forward "
        f"(expected {baseline_size}, got {len(sc)})"
    )


def test_save_load_helpers_round_trip(tmp_path):
    """Our save/load helpers must round-trip real schedule_cache
    entries through the filesystem."""
    from tinygrad import Tensor

    from egg_toolbox.backends.tinygrad import (
        _load_schedule_cache_into_tinygrad,
        _save_schedule_cache_from_tinygrad,
    )

    sc = _with_clean_cache()
    # Run a few different forwards to get multiple entries.
    (Tensor([1.0, 2.0, 3.0]) * 2).realize()
    (Tensor([[1.0, 2.0], [3.0, 4.0]]).sum()).realize()

    before_size = len(sc)
    assert before_size >= 1

    cache_path = str(tmp_path / "schedule-cache.pkl")
    _save_schedule_cache_from_tinygrad(cache_path)
    assert os.path.exists(cache_path)

    # Clear + reload.
    sc.clear()
    n_restored = _load_schedule_cache_into_tinygrad(cache_path)
    assert n_restored == before_size
    assert len(sc) == before_size


def test_load_missing_file_is_noop(tmp_path):
    """A non-existent cache file returns 0 restored and doesn't touch
    tinygrad's schedule_cache."""
    from egg_toolbox.backends.tinygrad import _load_schedule_cache_into_tinygrad
    sc = _with_clean_cache()
    n = _load_schedule_cache_into_tinygrad(str(tmp_path / "nope.pkl"))
    assert n == 0
    assert len(sc) == 0


def test_load_malformed_file_does_not_raise(tmp_path, capsys):
    """A corrupt pickle file logs and returns 0 -- must not crash
    the server on startup."""
    from egg_toolbox.backends.tinygrad import _load_schedule_cache_into_tinygrad
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"definitely not a pickle blob")
    sc = _with_clean_cache()
    n = _load_schedule_cache_into_tinygrad(str(path))
    assert n == 0
    assert len(sc) == 0
    captured = capsys.readouterr()
    assert "load failed" in captured.out


def test_save_empty_cache_does_not_create_file(tmp_path):
    """If schedule_cache is empty (nothing to save yet), the save
    helper is a no-op -- no empty file clutter."""
    from egg_toolbox.backends.tinygrad import _save_schedule_cache_from_tinygrad
    _with_clean_cache()  # empty
    path = str(tmp_path / "never-created.pkl")
    _save_schedule_cache_from_tinygrad(path)
    assert not os.path.exists(path)
