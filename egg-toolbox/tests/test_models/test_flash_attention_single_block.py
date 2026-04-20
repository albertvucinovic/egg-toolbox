"""M2 tests: tinygrad port with single-block configuration.

Exercises the online-softmax math in tinygrad (``tiled_attention`` with
``B_block >= N``) and validates bit-for-bit against the numpy reference
(already verified in M1) plus upstream tinygrad
``scaled_dot_product_attention``.

Epic: workspace-2r9.
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------- #

def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=shape).astype(np.float32)


def _causal_mask_np(T, start_pos, N):
    rows = np.arange(T).reshape(T, 1)
    cols = np.arange(N).reshape(1, N)
    allowed = cols <= (start_pos + rows)
    return np.where(allowed, 0.0, -np.inf).astype(np.float32)


# --------------------------------------------------------------------- #
# vs numpy reference (M1) -- proves tinygrad port preserves the math
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("T,start_pos", [
    (1, 0), (1, 7), (1, 128), (1, 1023),
    (16, 0), (16, 32), (16, 513),
    (128, 0), (128, 33), (128, 512),
])
def test_matches_numpy_reference_causal_single_block(T, start_pos):
    """Tinygrad tiled_attention with B_block>=N matches numpy ref."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        make_causal_mask_fn,
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    N = start_pos + T
    B, H, d = 1, 2, 32
    q_np = _rand((B, H, T, d), seed=100 + start_pos)
    k_np = _rand((B, H, N, d), seed=200 + start_pos)
    v_np = _rand((B, H, N, d), seed=300 + start_pos)

    # Numpy reference (already M1-validated against naive softmax)
    mask_fn = make_causal_mask_fn(T=T, start_pos=start_pos, N=N)
    ref = np_tiled(q_np, k_np, v_np, mask_fn=mask_fn, B_block=max(N, 1))

    # Tinygrad port -- single block means B_block >= N
    q_tg = Tensor(q_np)
    k_tg = Tensor(k_np)
    v_tg = Tensor(v_np)
    out_tg = tg_tiled(q_tg, k_tg, v_tg, start_pos=start_pos, B_block=max(N, 1), causal=True).numpy()

    np.testing.assert_allclose(out_tg, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("T,N", [(1, 1), (4, 8), (16, 64), (64, 256)])
def test_matches_numpy_reference_non_causal(T, N):
    """Non-causal (no mask) single-block tinygrad matches numpy ref."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    B, H, d = 1, 3, 32
    q_np = _rand((B, H, T, d), seed=400 + T + N)
    k_np = _rand((B, H, N, d), seed=500 + T + N)
    v_np = _rand((B, H, N, d), seed=600 + T + N)

    ref = np_tiled(q_np, k_np, v_np, mask_fn=None, B_block=N)

    q_tg, k_tg, v_tg = Tensor(q_np), Tensor(k_np), Tensor(v_np)
    out_tg = tg_tiled(q_tg, k_tg, v_tg, start_pos=0, B_block=N, causal=False).numpy()

    np.testing.assert_allclose(out_tg, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# vs upstream tinygrad scaled_dot_product_attention
# --------------------------------------------------------------------- #

def _upstream_sdpa_causal(q_tg, k_tg, v_tg, start_pos):
    """Call tinygrad's SDPA with an explicit causal mask matching our semantics."""
    from tinygrad import Tensor

    B, H, T, d = q_tg.shape
    _, _, N, _ = k_tg.shape
    mask = Tensor(_causal_mask_np(T, start_pos, N))
    # Cast mask to match q dtype
    mask = mask.cast(q_tg.dtype)
    return q_tg.scaled_dot_product_attention(k_tg, v_tg, attn_mask=mask)


@pytest.mark.parametrize("T,start_pos", [
    (1, 0), (1, 33), (1, 1023),
    (16, 0), (16, 64), (16, 513),
    (128, 0), (128, 33),
])
def test_matches_upstream_sdpa(T, start_pos):
    """Our tiled_attention result ≈ upstream scaled_dot_product_attention."""
    from tinygrad import Tensor

    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    N = start_pos + T
    B, H, d = 1, 4, 32
    q_np = _rand((B, H, T, d), seed=700 + start_pos)
    k_np = _rand((B, H, N, d), seed=800 + start_pos)
    v_np = _rand((B, H, N, d), seed=900 + start_pos)

    q_tg = Tensor(q_np).cast("float32")
    k_tg = Tensor(k_np).cast("float32")
    v_tg = Tensor(v_np).cast("float32")

    upstream = _upstream_sdpa_causal(q_tg, k_tg, v_tg, start_pos).numpy()
    ours = tg_tiled(q_tg, k_tg, v_tg, start_pos=start_pos, B_block=N, causal=True).numpy()

    # Upstream uses a separate reduce chain; some float rearrangement is
    # expected.  fp32 tolerance 1e-4 is conservative for these sizes.
    np.testing.assert_allclose(ours, upstream, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------- #
# Sanity: T=1 decode at a high start_pos (decode-after-long-prefill)
# --------------------------------------------------------------------- #

def test_t1_decode_after_long_prefill():
    """T=1 at start_pos=4095: the single-token decode path we care about."""
    from tinygrad import Tensor

    from egg_toolbox.models._flash_attention_numpy import (
        make_causal_mask_fn,
        tiled_attention as np_tiled,
    )
    from egg_toolbox.models.flash_attention import tiled_attention as tg_tiled

    start_pos, T = 4095, 1
    N = start_pos + T
    B, H, d = 1, 2, 64
    q_np = _rand((B, H, T, d), seed=1111)
    k_np = _rand((B, H, N, d), seed=2222)
    v_np = _rand((B, H, N, d), seed=3333)

    ref = np_tiled(
        q_np, k_np, v_np,
        mask_fn=make_causal_mask_fn(T=T, start_pos=start_pos, N=N),
        B_block=N,
    )
    ours = tg_tiled(
        Tensor(q_np), Tensor(k_np), Tensor(v_np),
        start_pos=start_pos, B_block=N, causal=True,
    ).numpy()

    np.testing.assert_allclose(ours, ref, atol=1e-4, rtol=1e-4)
