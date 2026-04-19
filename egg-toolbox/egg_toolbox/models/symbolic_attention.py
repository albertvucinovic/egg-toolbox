"""Symbolic-start_pos causal attention for T>1 prefill chunks.

tinygrad's ``TransformerBlock._attention`` constructs the causal mask
via ``Tensor.full((1, 1, T, start_pos+T), -inf).triu(start_pos+1)``.
``Tensor.full`` accepts the symbolic ``start_pos+T`` shape dim -- that
part works with a UOp-bound ``start_pos``.  But ``.triu(...)`` calls
``Tensor._tri`` which asserts both shape dims are plain ints:

    assert isinstance(r, int) and isinstance(c, int), \\
        f"does not support symbolic, getting {r=}, {c=}"

So T>1 with UOp start_pos fails at mask construction.  Today we work
around this by passing int start_pos for chunks -- but then every
chunk position is a distinct kernel shape signature, 300+ unique
kernels needing ``cuModuleLoadData`` on first process use (~27s
startup) and 680+ per-kernel Python dispatches per chunk (~8s each
AFTER startup).

This module replaces the triu step with an equivalent ``arange + >=
+ where`` construction that uses only ops that accept symbolic
shapes, which tinygrad's cache_kv slice already exercises (see
``cache_kv[:, :, :, start_pos:start_pos+T, :]``).  With this patched
in, T>1 chunks become JIT-eligible under UOp start_pos, collapsing N
per-position kernel variants into one symbolic graph and letting
TinyJit submit the whole forward as a single GPU-side launch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinygrad import Tensor, UOp


def symbolic_causal_mask(
    T: int,
    start_pos: "int | UOp",
    max_context: int,
    dtype: Any,
    device: Any,
) -> "Tensor":
    """Build the ``(1, 1, T, start_pos+T)`` causal mask without triu.

    For each (i, j) position in the mask:
      mask[0, 0, i, j] = 0       if j <= start_pos + i  (attend)
                       = -inf    if j >  start_pos + i  (block)

    This matches the semantic of
    ``Tensor.full((1,1,T,start_pos+T), -inf).triu(start_pos+1)``
    used in tinygrad's ``TransformerBlock._attention``, but every
    op (``arange``, symbolic slice, comparison, ``where``, ``reshape``)
    supports symbolic shape dims.  tinygrad's ``cache_kv[...,
    start_pos:start_pos+T, ...]`` slice already exercises the same
    pattern so this is not relying on novel symbolic machinery.
    """
    from tinygrad import Tensor

    # Build the mask DIRECTLY at the target symbolic shape
    # ``(1, 1, T, start_pos+T)``.  An earlier version of this
    # function worked by constructing at fixed ``(T, max_context)``
    # and slicing the last dim down at the end, but the scheduler
    # had to chew through a (T, 8192) graph on every prefill chunk
    # -- ~30x slower cold-compile than this direct path on a
    # measured benchmark (68ms vs 2036ms for the mask alone).
    #
    # Pattern: a symbolic-sized column index from slicing a fixed
    # arange, broadcast-subtracted against a fixed row index, then
    # compared + ``where``.  The one tricky move is that you can't
    # call ``.unsqueeze()``/``.reshape(..., -1)`` on a
    # symbolic-shape tensor (those query ``.ndim`` / ``.shape``
    # which assert int dims), but reshape to an explicit tuple that
    # includes the symbolic dim DOES work -- same shape signature
    # that tinygrad's own ``Tensor.full((1,1,T,start_pos+T), -inf)``
    # uses internally.

    # Column indices at symbolic extent: slice a fixed arange.
    # ``cache_kv[..., 0:start_pos+T, ...]`` uses the same pattern.
    cols_fixed = Tensor.arange(max_context, device=device)        # (max_ctx,)
    cols_sym = cols_fixed[:start_pos + T]                         # (start_pos+T,) symbolic

    # Row indices at fixed extent.
    rows = Tensor.arange(T, device=device)                        # (T,)

    # Broadcast ``(T, 1)`` minus ``(start_pos+T,)`` gives
    # ``(T, start_pos+T)`` with numpy-style right-align broadcasting,
    # no explicit unsqueeze of the symbolic side needed.
    # ``j <= start_pos + i`` iff ``(i - j + start_pos) >= 0``.
    offset = (rows.reshape(T, 1) - cols_sym) + start_pos          # (T, start_pos+T)
    allowed = offset >= 0                                         # (T, start_pos+T) bool

    minus_inf = Tensor.full((1,), float("-inf"), dtype=dtype, device=device)
    zero = Tensor.full((1,), 0.0, dtype=dtype, device=device)
    mask = allowed.where(zero, minus_inf)                         # (T, start_pos+T)

    # Reshape with explicit shape tuple (including the symbolic
    # dim) is supported -- it's the same tuple ``Tensor.full`` would
    # accept.  Avoid ``.unsqueeze(...)`` which walks ``.shape`` and
    # trips the int-assertion on symbolic-shape tensors.
    return mask.reshape(1, 1, T, start_pos + T)                   # (1, 1, T, start_pos+T)


def patch_block_attention(block: Any, max_context: int) -> None:
    """Rebind ``block._attention`` to the symbolic-mask variant.

    We patch in-place so tinygrad's scheduler sees one contiguous
    forward path -- simpler than subclassing and rewiring model.blk.
    Called once per TransformerBlock at model-load time.

    Idempotent: if a block already has our patched attention, leaves
    it alone.
    """
    from tinygrad import Tensor

    if getattr(block, "_egg_symbolic_attention_patched", False):
        return

    # Close over the block so we can read its attributes the same way
    # tinygrad's method does.
    def _egg_attention(self: Any, x: "Tensor", start_pos: "int | UOp") -> "Tensor":
        from tinygrad.apps.llm import apply_rope, precompute_freqs_cis

        x_norm = self.attn_norm(x)
        q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
        if self.qk_norm and self.qk_norm != self.head_dim:
            q, k = self.attn_q_norm(q), self.attn_k_norm(k)

        B, T, _ = x.shape
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm == self.head_dim:
            q, k = self.attn_q_norm(q), self.attn_k_norm(k)

        freqs_cis = precompute_freqs_cis(
            self.head_dim, self.max_context, self.rope_theta,
        )[start_pos:start_pos + T]
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if not hasattr(self, "cache_kv"):
            self.cache_kv = Tensor.zeros(
                2, B, self.n_kv_heads, self.max_context, self.head_dim,
                dtype=k.dtype, device=k.device,
            ).contiguous().realize()
        self.cache_kv[:, :, :, start_pos:start_pos + T, :].assign(
            Tensor.stack(k, v),
        ).realize()
        k = self.cache_kv[0, :, :, 0:start_pos + T, :]
        v = self.cache_kv[1, :, :, 0:start_pos + T, :]

        # THE ONE CHANGE vs tinygrad upstream: mask via arange+where
        # instead of triu.  Symbolic-friendly, so T>1 with UOp
        # start_pos doesn't hit the int-shape assertion in _tri.
        if T > 1:
            mask = symbolic_causal_mask(
                T=T, start_pos=start_pos,
                max_context=self.max_context,
                dtype=x.dtype, device=x.device,
            )
        else:
            mask = None

        attn = q.scaled_dot_product_attention(
            k, v, attn_mask=mask, enable_gqa=True,
        )
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        attn = self.attn_output(attn)
        return x + attn

    # Bind the method to this specific instance.
    import types
    block._attention = types.MethodType(_egg_attention, block)
    block._egg_symbolic_attention_patched = True
