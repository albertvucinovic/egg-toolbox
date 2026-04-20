"""M4 tests: multi-block tiled attention with causal masking.

Exercises the combination that matters for chunked prefill: multiple
K/V blocks per forward, where some are fully-attended (before
``start_pos``), at least one is a boundary block (contains the chunk's
own query positions), and the last block may be ragged.

Validates against both:
- numpy reference (``_flash_attention_numpy.tiled_attention``), M1-proven
- upstream tinygrad ``scaled_dot_product_attention`` with the full triu
  causal mask baked into an ``attn_mask`` tensor

Epic: workspace-2r9.
"""
from __future__ import annotations

import numpy as np
import pytest


def _rand(shape, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=shape).astype(dtype)


def _causal_mask_np(T, start_pos, N, dtype=np.float32):
    rows = np.arange(T).reshape(T, 1)
    cols = np.arange(N).reshape(1, N)
    allowed = cols <= (start_pos + rows)
    return np.where(allowed, 0.0, -np.inf).astype(dtype)


# --------------------------------------------------------------------- #
# Multi-block + causal vs numpy reference
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize("start_pos", [33, 256, 1024, 4095])
@pytest.mark.parametrize("B_block", [64, 128, 256])
def test_multiblock_causal_matches_numpy(T, start_pos, B_block):
    """Multi-block causal attention matches the numpy reference."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        make_causal_mask_fn,
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    N = start_pos + T
    B, H, d = 1, 2, 32
    q_np = _rand((B, H, T, d), seed=start_pos * 11 + T + B_block)
    k_np = _rand((B, H, N, d), seed=start_pos * 11 + T + B_block + 1)
    v_np = _rand((B, H, N, d), seed=start_pos * 11 + T + B_block + 2)

    ref = np_tiled(
        q_np, k_np, v_np,
        mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=B_block,
    )
    ours = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True,
    ).numpy()

    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Multi-block + causal vs upstream SDPA
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize("start_pos", [0, 33, 256, 1024])
@pytest.mark.parametrize("B_block", [64, 128, 256])
def test_multiblock_causal_matches_upstream_sdpa(T, start_pos, B_block):
    """Tiled causal matches tinygrad's SDPA with an explicit causal mask."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    N = start_pos + T
    B, H, d = 1, 4, 32
    q_np = _rand((B, H, T, d), seed=start_pos * 7 + T + B_block + 50000)
    k_np = _rand((B, H, N, d), seed=start_pos * 7 + T + B_block + 50001)
    v_np = _rand((B, H, N, d), seed=start_pos * 7 + T + B_block + 50002)

    q_tg = Tensor(q_np)
    k_tg = Tensor(k_np)
    v_tg = Tensor(v_np)

    mask_tg = Tensor(_causal_mask_np(T, start_pos, N))
    upstream = q_tg.scaled_dot_product_attention(k_tg, v_tg, attn_mask=mask_tg).numpy()
    ours = tg_tiled(q_tg, k_tg, v_tg, start_pos=start_pos, B_block=B_block, causal=True).numpy()

    np.testing.assert_allclose(ours, upstream, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------- #
# Block-size invariance for the causal path
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("start_pos", [0, 33, 512, 1023])
def test_causal_block_size_invariance(start_pos):
    """Changing B_block must not change the causal result."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    T = 16
    N = start_pos + T
    B, H, d = 1, 3, 32
    q_np = _rand((B, H, T, d), seed=start_pos + 70000)
    k_np = _rand((B, H, N, d), seed=start_pos + 70001)
    v_np = _rand((B, H, N, d), seed=start_pos + 70002)

    q_tg, k_tg, v_tg = Tensor(q_np), Tensor(k_np), Tensor(v_np)

    results = {}
    for B_block in (32, 64, 128, 256, N):
        out = tg_tiled(q_tg, k_tg, v_tg, start_pos=start_pos, B_block=B_block, causal=True).numpy()
        results[B_block] = out

    # Pick any one as reference and compare the rest
    ref_key = N
    for B_block, out in results.items():
        np.testing.assert_allclose(
            out, results[ref_key], atol=1e-5, rtol=1e-5,
            err_msg=f"B_block={B_block} vs B_block={ref_key}",
        )


# --------------------------------------------------------------------- #
# FULL + BOUNDARY + RAGGED mix (the real chunked-prefill pattern)
# --------------------------------------------------------------------- #

def test_realistic_chunked_prefill_shapes():
    """Qwen3-8B-like shapes: d=128, n_heads=32.  Chunk T=128 at start_pos=3968 -> N=4096."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        make_causal_mask_fn,
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    T, start_pos = 128, 3968
    N = start_pos + T  # 4096, divisible by 256
    B, H, d = 1, 4, 128  # small H for test speed
    q_np = _rand((B, H, T, d), seed=424242)
    k_np = _rand((B, H, N, d), seed=424243)
    v_np = _rand((B, H, N, d), seed=424244)

    ref = np_tiled(
        q_np, k_np, v_np,
        mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=256,
    )
    ours = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=256, causal=True,
    ).numpy()

    np.testing.assert_allclose(ours, ref, atol=5e-5, rtol=5e-5)


def test_ragged_last_block_causal():
    """start_pos + T not divisible by B_block: ragged last block gets masked."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        make_causal_mask_fn,
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    # start_pos=200, T=16 -> N=216.  B_block=64 -> 4 blocks: [0,64) FULL,
    # [64,128) FULL, [128,192) FULL, [192,216) BOUNDARY+RAGGED (24/64 valid).
    start_pos, T, B_block = 200, 16, 64
    N = start_pos + T

    B, H, d = 1, 2, 32
    q_np = _rand((B, H, T, d), seed=654321)
    k_np = _rand((B, H, N, d), seed=654322)
    v_np = _rand((B, H, N, d), seed=654323)

    ref = np_tiled(
        q_np, k_np, v_np,
        mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=B_block,
    )
    ours = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=B_block, causal=True,
    ).numpy()

    np.testing.assert_allclose(ours, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# fp16 tolerance (what real inference uses)
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("start_pos", [33, 256, 1024])
def test_causal_fp16(start_pos):
    """fp16 inputs: tiled attention matches upstream within 1e-2."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    T, B_block = 16, 64
    N = start_pos + T
    B, H, d = 1, 2, 32
    q_np = _rand((B, H, T, d), seed=start_pos + 80000, dtype=np.float16)
    k_np = _rand((B, H, N, d), seed=start_pos + 80001, dtype=np.float16)
    v_np = _rand((B, H, N, d), seed=start_pos + 80002, dtype=np.float16)

    q_tg = Tensor(q_np).cast("float16")
    k_tg = Tensor(k_np).cast("float16")
    v_tg = Tensor(v_np).cast("float16")

    mask_tg = Tensor(_causal_mask_np(T, start_pos, N).astype(np.float16)).cast("float16")
    upstream = q_tg.scaled_dot_product_attention(k_tg, v_tg, attn_mask=mask_tg).numpy()
    ours = tg_tiled(q_tg, k_tg, v_tg, start_pos=start_pos, B_block=B_block, causal=True).numpy()

    # fp16 SDPA has significant rounding differences since our internal
    # math is in fp32 while upstream stays in fp16.  Use 1e-2 absolute
    # tolerance (a few ULPs of fp16).
    np.testing.assert_allclose(ours, upstream, atol=5e-3, rtol=5e-3)
