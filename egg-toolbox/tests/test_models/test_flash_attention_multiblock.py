"""M3 tests: multi-block tiled attention (non-causal path).

Exercises ``tiled_attention`` when ``N > B_block`` so the Python loop
iterates multiple blocks and the online-softmax accumulator must
correctly combine results across them.  Non-causal only -- causal
masking comes in M4.

Epic: workspace-2r9.
"""
from __future__ import annotations

import numpy as np
import pytest


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=shape).astype(np.float32)


# --------------------------------------------------------------------- #
# Against numpy reference (M1) at every B_block that actually tiles
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [65, 128, 129, 256, 513, 1024])
@pytest.mark.parametrize("B_block", [16, 32, 64, 128])
def test_multiblock_non_causal_matches_numpy(N, B_block):
    """Multi-block tinygrad ≡ multi-block numpy ref (1e-5 fp32)."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    # Skip degenerate where only one block fires (covered by M2 tests)
    if N <= B_block:
        pytest.skip("single-block case")

    T, B, H, d = 8, 1, 2, 32
    q_np = _rand((B, H, T, d), seed=N * 17 + B_block)
    k_np = _rand((B, H, N, d), seed=N * 17 + B_block + 1)
    v_np = _rand((B, H, N, d), seed=N * 17 + B_block + 2)

    ref = np_tiled(q_np, k_np, v_np, mask_fn=None, B_block=B_block)

    out = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=0, B_block=B_block, causal=False,
    ).numpy()

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Invariance: multi-block result should match single-block result
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("N,B_block_small", [
    (256, 32), (256, 64), (256, 128),
    (513, 64), (513, 256),
    (1024, 128), (1024, 256),
])
def test_multiblock_matches_single_block(N, B_block_small):
    """tiled(..., B_block=small) == tiled(..., B_block=N) -- block size is an
    optimisation knob; the result must not depend on it."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    T, B, H, d = 16, 1, 3, 32
    q_np = _rand((B, H, T, d), seed=N + B_block_small)
    k_np = _rand((B, H, N, d), seed=N + B_block_small + 1000)
    v_np = _rand((B, H, N, d), seed=N + B_block_small + 2000)

    q_tg, k_tg, v_tg = Tensor(q_np), Tensor(k_np), Tensor(v_np)

    single = tg_tiled(q_tg, k_tg, v_tg, start_pos=0, B_block=N, causal=False).numpy()
    multi = tg_tiled(q_tg, k_tg, v_tg, start_pos=0, B_block=B_block_small, causal=False).numpy()

    np.testing.assert_allclose(multi, single, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Against upstream SDPA for real-sized cases
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("B_block", [64, 128, 256])
def test_multiblock_matches_upstream_sdpa_non_causal(N, B_block):
    """Multi-block non-causal ≈ upstream SDPA with no mask."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    if N <= B_block:
        pytest.skip("single-block case")

    T, B, H, d = 4, 1, 2, 32
    q_np = _rand((B, H, T, d), seed=N * 3 + B_block * 5)
    k_np = _rand((B, H, N, d), seed=N * 3 + B_block * 5 + 1)
    v_np = _rand((B, H, N, d), seed=N * 3 + B_block * 5 + 2)

    q_tg = Tensor(q_np)
    k_tg = Tensor(k_np)
    v_tg = Tensor(v_np)

    upstream = q_tg.scaled_dot_product_attention(k_tg, v_tg).numpy()
    ours = tg_tiled(q_tg, k_tg, v_tg, start_pos=0, B_block=B_block, causal=False).numpy()

    np.testing.assert_allclose(ours, upstream, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------- #
# Block-alignment edge cases
# --------------------------------------------------------------------- #

def test_exact_multiple_vs_ragged_last_block():
    """N divisible by B_block vs N = B_block*k + r should both be correct."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    B, H, T, d = 1, 2, 4, 16
    B_block = 32

    for N in (32, 33, 63, 64, 65, 95, 96, 97):
        q_np = _rand((B, H, T, d), seed=N + 500)
        k_np = _rand((B, H, N, d), seed=N + 600)
        v_np = _rand((B, H, N, d), seed=N + 700)

        ref = np_tiled(q_np, k_np, v_np, mask_fn=None, B_block=B_block)
        ours = tg_tiled(
            Tensor(q_np), Tensor(k_np), Tensor(v_np),
            start_pos=0, B_block=B_block, causal=False,
        ).numpy()

        np.testing.assert_allclose(
            ours, ref, atol=1e-5, rtol=1e-5,
            err_msg=f"N={N}, B_block={B_block}",
        )


def test_large_batch_and_heads():
    """Batch and head dims broadcast correctly across the block loop."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    B, H, T, N, d = 4, 8, 16, 512, 32
    q_np = _rand((B, H, T, d), seed=9000)
    k_np = _rand((B, H, N, d), seed=9001)
    v_np = _rand((B, H, N, d), seed=9002)

    ref = np_tiled(q_np, k_np, v_np, mask_fn=None, B_block=128)
    ours = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=0, B_block=128, causal=False,
    ).numpy()

    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)
