"""Tests for the symbolic-mask attention replacement.

Exhaustively verifies that ``symbolic_causal_mask`` produces bit-
identical output to tinygrad's ``triu``-based construction for the
INT-start_pos path (which is the path we can actually materialize
without a full JIT trace).

The UOp-start_pos path produces a symbolic-shape tensor that can't
be realized standalone -- it only takes concrete form inside a full
transformer forward where every op is symbolic-aware.  For that
path we test end-to-end via the real model when available; here we
just verify the mask construction doesn't raise and has the expected
symbolic shape.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("T,start_pos,max_ctx", [
    (4, 0, 16),
    (4, 3, 16),
    (4, 10, 32),
    (8, 0, 64),
    (8, 16, 64),
    (16, 0, 128),
    (16, 7, 128),
    (32, 96, 256),
    (128, 0, 512),
    (128, 128, 512),
])
def test_int_start_pos_matches_triu(T, start_pos, max_ctx):
    """For any int start_pos with T>1, our mask must be bit-identical
    to tinygrad's ``Tensor.full(...).triu(start_pos+1)``.
    (T=1 is tested separately since upstream uses ``mask=None``
    rather than constructing a triu.)"""
    from tinygrad import Tensor
    from egg_toolbox.models.symbolic_attention import symbolic_causal_mask

    ours = symbolic_causal_mask(
        T=T, start_pos=start_pos, max_context=max_ctx,
        dtype="float32", device="CPU",
    )
    upstream = Tensor.full(
        (1, 1, T, start_pos + T), float("-inf"),
        dtype="float32", device="CPU",
    ).triu(start_pos + 1)

    assert ours.shape == upstream.shape, f"shape {ours.shape} vs {upstream.shape}"
    assert ours.tolist() == upstream.tolist(), (
        f"value mismatch at T={T} start_pos={start_pos}"
    )


def test_uop_start_pos_produces_symbolic_shape():
    """UOp start_pos should produce a tensor with a symbolic last
    dim -- we can't materialize it standalone, but it must at least
    construct without raising, so it's usable inside a JIT trace."""
    from tinygrad import Tensor, UOp
    from egg_toolbox.models.symbolic_attention import symbolic_causal_mask

    v_sp = UOp.variable("start_pos", 1, 127)
    sp = v_sp.bind(5)

    mask = symbolic_causal_mask(
        T=8, start_pos=sp, max_context=128,
        dtype="float32", device="CPU",
    )

    # Shape should be (1, 1, 8, UOp-expr).  First three dims are int;
    # last dim is symbolic.
    assert mask.shape[:3] == (1, 1, 8)
    # Last dim is not an int -- it's the symbolic start_pos + T.
    assert not isinstance(mask.shape[3], int), (
        f"expected symbolic last dim, got {mask.shape[3]!r}"
    )


def test_t_equals_one_no_mask_needed_upstream():
    """Upstream tinygrad skips mask construction entirely for T=1.
    Our symbolic path produces a valid mask anyway (no short-circuit),
    but callers should gate on ``if T > 1`` just like upstream does."""
    from tinygrad import Tensor
    from egg_toolbox.models.symbolic_attention import symbolic_causal_mask

    # T=1 with start_pos=5: mask should be (1,1,1,6) filled with 0.
    # (Every column j=0..5 is <= start_pos+0=5 so all attend.)
    ours = symbolic_causal_mask(
        T=1, start_pos=5, max_context=16, dtype="float32", device="CPU",
    )
    expected = [[[[0.0] * 6]]]
    assert ours.tolist() == expected, f"T=1 mask wrong: {ours.tolist()}"
