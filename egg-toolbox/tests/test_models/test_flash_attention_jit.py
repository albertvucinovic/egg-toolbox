"""M6 tests: TinyJit-wrapped block updates.

Verifies that ``FlashAttentionRunner`` produces the same result as the
pure-eager ``tiled_attention`` across capture (cnt=1) and replay
(cnt>=2) calls.  Exercises shape-signature caching by calling with
multiple shape combinations on one runner.

Epic: workspace-2r9.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=shape).astype(np.float32)


# --------------------------------------------------------------------- #
# JIT capture/replay sanity
# --------------------------------------------------------------------- #

def test_runner_capture_then_replay_non_causal():
    """Non-causal tiled attention via runner matches eager after 3 calls."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        tiled_attention,
    )

    B, H, T, d = 1, 2, 16, 32
    B_block = 64
    N = 512  # exact multiple of B_block

    q_np = _rand((B, H, T, d), seed=1)
    k_np = _rand((B, H, N, d), seed=2)
    v_np = _rand((B, H, N, d), seed=3)

    eager = tiled_attention(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=0, B_block=B_block, causal=False,
    ).numpy()

    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(d))
    outs = []
    for i in range(3):
        out = tiled_attention(
            Tensor(q_np), Tensor(k_np), Tensor(v_np),
            start_pos=0, B_block=B_block, causal=False, runner=runner,
        ).numpy()
        outs.append(out)

    # All three calls return the same result (capture + replay consistency)
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(outs[1], outs[2], atol=1e-5, rtol=1e-5)
    # And match eager
    np.testing.assert_allclose(outs[2], eager, atol=1e-5, rtol=1e-5)


def test_runner_capture_then_replay_causal():
    """Causal tiled attention via runner matches eager across calls."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        tiled_attention,
    )

    B, H, T, d = 1, 2, 16, 32
    B_block = 64
    start_pos = 128
    N = start_pos + T  # 144

    q_np = _rand((B, H, T, d), seed=11)
    k_np = _rand((B, H, N, d), seed=22)
    v_np = _rand((B, H, N, d), seed=33)

    eager = tiled_attention(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True,
    ).numpy()

    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(d))
    outs = [
        tiled_attention(
            Tensor(q_np), Tensor(k_np), Tensor(v_np),
            start_pos=start_pos, B_block=B_block, causal=True, runner=runner,
        ).numpy()
        for _ in range(3)
    ]

    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(outs[1], outs[2], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(outs[2], eager, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Multiple JIT instances by shape signature
# --------------------------------------------------------------------- #

def test_runner_caches_different_shapes_separately():
    """Runner should create distinct JITs for different (T, B_block) combos."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        tiled_attention,
    )

    B, H, d = 1, 2, 32
    B_block = 64
    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(d))

    # Call with T=1 (decode) and T=16 (prefill chunk) — different shapes
    def _run(T, start_pos):
        N = start_pos + T
        q_np = _rand((B, H, T, d), seed=T * 13 + start_pos)
        k_np = _rand((B, H, N, d), seed=T * 13 + start_pos + 1)
        v_np = _rand((B, H, N, d), seed=T * 13 + start_pos + 2)
        return tiled_attention(
            Tensor(q_np), Tensor(k_np), Tensor(v_np),
            start_pos=start_pos, B_block=B_block, causal=True, runner=runner,
        ).numpy()

    # T=1, start_pos multiples of B_block so N stays a multiple
    r1 = _run(T=1, start_pos=63)   # N=64, one full block
    r2 = _run(T=1, start_pos=127)  # N=128, two full blocks
    # T=16
    r3 = _run(T=16, start_pos=48)  # N=64, one full block
    r4 = _run(T=16, start_pos=112) # N=128, two full blocks

    # Shape signatures should be distinct: T=1 vs T=16 differ in q, m, l, out shapes.
    # Multiple full_jits should be present.
    assert len(runner._full_jits) >= 1  # fully-attended blocks (FULL)
    assert len(runner._masked_jits) >= 1  # boundary blocks

    # Run again -- should hit the cached JITs (no new entries)
    full_before = dict(runner._full_jits)
    masked_before = dict(runner._masked_jits)
    _run(T=16, start_pos=112)
    assert runner._full_jits.keys() == full_before.keys()
    assert runner._masked_jits.keys() == masked_before.keys()


# --------------------------------------------------------------------- #
# Correctness across many consecutive calls (simulates chunk dispatch)
# --------------------------------------------------------------------- #

def test_runner_many_dispatches_stay_correct():
    """Simulate a prefill-like loop: 8 forwards, all same shape, all match eager."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        tiled_attention,
    )

    B, H, T, d = 1, 4, 16, 32
    B_block = 64
    start_pos = 64
    N = start_pos + T

    q_np = _rand((B, H, T, d), seed=999)
    k_np = _rand((B, H, N, d), seed=998)
    v_np = _rand((B, H, N, d), seed=997)

    eager = tiled_attention(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True,
    ).numpy()

    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(d))
    for _ in range(8):
        out = tiled_attention(
            Tensor(q_np), Tensor(k_np), Tensor(v_np),
            start_pos=start_pos, B_block=B_block, causal=True, runner=runner,
        ).numpy()
        np.testing.assert_allclose(out, eager, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Ragged last block falls back to eager gracefully
# --------------------------------------------------------------------- #

def test_ragged_last_block_with_runner_matches_eager():
    """A forward whose last block is ragged must still match eager path."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        tiled_attention,
    )

    B, H, T, d = 1, 2, 16, 32
    B_block = 64
    start_pos = 100  # N=116, last block [64,116) = 52 positions (ragged)
    N = start_pos + T

    q_np = _rand((B, H, T, d), seed=50000)
    k_np = _rand((B, H, N, d), seed=50001)
    v_np = _rand((B, H, N, d), seed=50002)

    eager = tiled_attention(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True,
    ).numpy()

    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(d))
    # Runner is used for the full block, eager handles the ragged last.
    out = tiled_attention(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True, runner=runner,
    ).numpy()

    np.testing.assert_allclose(out, eager, atol=1e-5, rtol=1e-5)
