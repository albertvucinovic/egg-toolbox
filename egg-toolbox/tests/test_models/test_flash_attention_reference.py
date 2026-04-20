"""Tests for the numpy reference in ``_flash_attention_numpy``.

M1 of workspace-2r9 (block-tiled FlashAttention epic).  These tests
exist to prove the online-softmax math is correct at fp32 before we
port to tinygrad -- the numerical subtleties would be painful to debug
through the JIT layer.
"""
from __future__ import annotations

import numpy as np
import pytest

from egg_toolbox.models._flash_attention_numpy import (
    make_boolean_causal_mask,
    make_causal_mask_fn,
    naive_attention,
    tiled_attention,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

def _random_qkv(B, H, T, N, d, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1, size=(B, H, T, d)).astype(np.float64)
    k = rng.normal(0, 1, size=(B, H, N, d)).astype(np.float64)
    v = rng.normal(0, 1, size=(B, H, N, d)).astype(np.float64)
    return q, k, v


# --------------------------------------------------------------------- #
# Non-causal (no mask) -- the baseline sanity case
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize("N", [1, 8, 64, 131, 512])
@pytest.mark.parametrize("B_block", [16, 64, 128])
def test_tiled_matches_naive_non_causal(T, N, B_block):
    """Tiled attention without a mask == naive ``softmax(QK^T) V``."""
    q, k, v = _random_qkv(B=2, H=3, T=T, N=N, d=32, seed=42)
    ref = naive_attention(q, k, v, mask=None)
    out = tiled_attention(q, k, v, mask_fn=None, B_block=B_block)
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------- #
# Causal attention -- the path egg-toolbox actually uses
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize("start_pos", [0, 1, 33, 256, 1024])
@pytest.mark.parametrize("B_block", [16, 64, 128, 256])
def test_tiled_causal_matches_naive(T, start_pos, B_block):
    """Tiled + causal mask_fn matches naive with full boolean mask."""
    N = start_pos + T
    q, k, v = _random_qkv(B=1, H=4, T=T, N=N, d=32, seed=7)

    # Naive: pass a full (T, N) causal mask
    mask_full = make_boolean_causal_mask(T=T, start_pos=start_pos, N=N)
    ref = naive_attention(q, k, v, mask=mask_full)

    # Tiled: use the per-block mask_fn
    mask_fn = make_causal_mask_fn(T=T, start_pos=start_pos, N=N)
    out = tiled_attention(q, k, v, mask_fn=mask_fn, B_block=B_block)

    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_tiled_causal_t1_decode_step():
    """T=1 "decode-after-long-prefill" hits the boundary block path."""
    start_pos = 4095
    T = 1
    N = start_pos + T  # 4096
    q, k, v = _random_qkv(B=1, H=2, T=T, N=N, d=64, seed=11)

    ref = naive_attention(
        q, k, v, mask=make_boolean_causal_mask(T=T, start_pos=start_pos, N=N)
    )
    out = tiled_attention(
        q, k, v, mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=256,
    )
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_tiled_causal_chunked_prefill_pattern():
    """T=128 chunk at start_pos=3968 (typical mid-prefill chunk for Qwen3-8B)."""
    start_pos = 3968
    T = 128
    N = start_pos + T
    q, k, v = _random_qkv(B=1, H=8, T=T, N=N, d=128, seed=23)

    ref = naive_attention(
        q, k, v, mask=make_boolean_causal_mask(T=T, start_pos=start_pos, N=N)
    )
    for B_block in (64, 128, 256, 512):
        out = tiled_attention(
            q, k, v,
            mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
            B_block=B_block,
        )
        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10,
                                   err_msg=f"B_block={B_block}")


# --------------------------------------------------------------------- #
# Block alignment edge cases
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [1, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_non_multiple_of_block_size(N):
    """Sequence length not divisible by B_block still matches naive."""
    q, k, v = _random_qkv(B=1, H=2, T=1, N=N, d=16, seed=51)
    ref = naive_attention(q, k, v, mask=None)
    out = tiled_attention(q, k, v, mask_fn=None, B_block=64)
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_start_pos_zero_first_chunk():
    """start_pos=0 chunk -- the edge case that used to exclude JIT."""
    start_pos = 0
    T = 128
    N = T
    q, k, v = _random_qkv(B=1, H=4, T=T, N=N, d=32, seed=77)
    ref = naive_attention(
        q, k, v, mask=make_boolean_causal_mask(T=T, start_pos=start_pos, N=N)
    )
    out = tiled_attention(
        q, k, v, mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=64,
    )
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------- #
# Mask classifier sanity: FULL vs BOUNDARY vs RAGGED blocks
# --------------------------------------------------------------------- #

def test_mask_fn_returns_none_for_fully_attended_blocks():
    """Blocks entirely below start_pos return None (FULL block signal)."""
    T = 64
    start_pos = 512
    N = start_pos + T
    mask_fn = make_causal_mask_fn(T=T, start_pos=start_pos, N=N)

    # Block [0, 128) -- all positions <= start_pos-1, fully attended
    assert mask_fn(0, 128) is None
    # Block [384, 512) -- block_end - 1 = 511 == start_pos - 1 -> FULL
    assert mask_fn(384, 512) is None
    # Block [512, 576) -- start_pos is in this block, boundary
    assert mask_fn(512, 576) is not None


def test_mask_fn_boundary_block_shape_and_content():
    """Boundary block mask must have the right shape and correct values."""
    T = 4
    start_pos = 8  # queries at absolute 8,9,10,11
    N = start_pos + T  # 12
    mask_fn = make_causal_mask_fn(T=T, start_pos=start_pos, N=N)

    # Block [0, 12) covers everything -- boundary
    m = mask_fn(0, 12)
    assert m.shape == (T, 12)

    # Query 0 (abs pos 8) attends to keys 0..8 inclusive, blocks 9..11
    assert np.all(m[0, :9] == 0.0)
    assert np.all(m[0, 9:] == -np.inf)
    # Query 3 (abs pos 11) attends to all 12 keys
    assert np.all(m[3, :] == 0.0)
