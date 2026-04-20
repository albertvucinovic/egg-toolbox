"""Block-tiled FlashAttention in tinygrad.

Mirrors the numpy reference in ``_flash_attention_numpy``.  The inner
block kernel has only int-shape tensors, which makes it BEAM-tuneable
and JIT-friendly (unlike the upstream ``softmax(Q K^T + mask) V`` path
whose mask axis is symbolic under UOp ``start_pos``).

Two APIs:

- ``tiled_attention(q, k, v, start_pos, B_block)`` — pure eager path;
  each block update is scheduled and executed separately.  Good for
  validation and the ragged-last-block case.
- ``FlashAttentionRunner`` — holds per-shape ``TinyJit`` instances for
  the two block-update variants (``_block_update_full`` and
  ``_block_update_masked``).  Fixed-size blocks dispatch through the
  cached JIT graph; the runner is passed into ``tiled_attention`` via
  the ``runner`` keyword.

Epic: workspace-2r9.  See ``docs/flash-attention-design.md``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinygrad import Tensor


# --------------------------------------------------------------------- #
# Per-block online-softmax update
# --------------------------------------------------------------------- #

def _block_update_full(
    q: "Tensor",
    k_block: "Tensor",
    v_block: "Tensor",
    m_prev: "Tensor",
    l_prev: "Tensor",
    out_prev: "Tensor",
    inv_sqrt_d: float,
):
    """Online-softmax update for a fully-attended block (no mask).

    Shapes::

        q         : (B, H, T, d)       [fp32 internal]
        k_block   : (B, H, block_len, d)
        v_block   : (B, H, block_len, d)
        m_prev    : (B, H, T, 1)
        l_prev    : (B, H, T, 1)
        out_prev  : (B, H, T, d)

    Returns updated ``(m, l, out)`` with the same shapes.
    """
    s = q.matmul(k_block.transpose(-2, -1)) * inv_sqrt_d             # (B, H, T, block_len)
    m_block = s.max(axis=-1, keepdim=True)                            # (B, H, T, 1)
    m_new = m_prev.maximum(m_block)                                   # (B, H, T, 1)
    alpha = (m_prev - m_new).exp()                                    # (B, H, T, 1)
    p_tilde = (s - m_new).exp()                                       # (B, H, T, block_len)
    l_new = l_prev * alpha + p_tilde.sum(axis=-1, keepdim=True)
    out_new = out_prev * alpha + p_tilde.matmul(v_block)
    return m_new, l_new, out_new


def _block_update_masked(
    q: "Tensor",
    k_block: "Tensor",
    v_block: "Tensor",
    mask: "Tensor",
    m_prev: "Tensor",
    l_prev: "Tensor",
    out_prev: "Tensor",
    inv_sqrt_d: float,
):
    """Same as ``_block_update_full`` plus a ``(T, block_len)`` mask
    added element-wise before the online-softmax normalise.

    The mask encodes -inf for blocked positions, 0.0 for attended; it
    must match the block length.  Broadcasting handles batch/head dims.
    """
    s = q.matmul(k_block.transpose(-2, -1)) * inv_sqrt_d + mask
    m_block = s.max(axis=-1, keepdim=True)
    m_new = m_prev.maximum(m_block)
    alpha = (m_prev - m_new).exp()
    p_tilde = (s - m_new).exp()
    l_new = l_prev * alpha + p_tilde.sum(axis=-1, keepdim=True)
    out_new = out_prev * alpha + p_tilde.matmul(v_block)
    return m_new, l_new, out_new


# --------------------------------------------------------------------- #
# Boundary-block mask construction
# --------------------------------------------------------------------- #

def _build_boundary_mask(
    T: int,
    block_start: int,
    block_end: int,
    start_pos: int,
    dtype: Any,
    device: Any,
) -> "Tensor":
    """Build a ``(T, block_end - block_start)`` causal mask.

    Attend iff ``key_abs_pos <= query_abs_pos``, i.e.,
    ``block_start + col <= start_pos + row`` → ``col <= (start_pos - block_start) + row``.
    All inputs are Python ints; the resulting tensor has fully int-shape.
    """
    from tinygrad import Tensor
    from tinygrad.dtype import dtypes

    block_len = block_end - block_start
    rows = Tensor.arange(T, dtype=dtypes.int32, device=device).reshape(T, 1)
    cols = Tensor.arange(block_len, dtype=dtypes.int32, device=device).reshape(1, block_len)
    threshold = rows + (start_pos - block_start)
    allowed = cols <= threshold
    # ``allowed.where(0, -inf)`` yields a (T, block_len) float mask.
    minus_inf = Tensor.full((1,), float("-inf"), dtype=dtype, device=device)
    zero = Tensor.full((1,), 0.0, dtype=dtype, device=device)
    return allowed.where(zero, minus_inf)


# --------------------------------------------------------------------- #
# High-level tiled attention
# --------------------------------------------------------------------- #

def tiled_attention(
    q: "Tensor",
    k: "Tensor",
    v: "Tensor",
    start_pos: int,
    B_block: int = 128,
    causal: bool = True,
    runner: "FlashAttentionRunner | None" = None,
) -> "Tensor":
    """Block-tiled FlashAttention over (k, v).

    Shapes::

        q      : (B, H, T, d)
        k, v   : (B, H, N, d)

    For chunked prefill, ``N = start_pos + T``.  For ``causal=True``,
    the query at chunk-relative position ``i`` attends to key
    positions ``[0, start_pos + i]``.

    If ``runner`` is provided, full-size blocks (block_len == B_block)
    dispatch through JIT'd graphs; the ragged last block (if any) falls
    back to eager.  Without a runner, every block is eager (correct,
    but slow — useful for validation).

    Returns: (B, H, T, d) in the same dtype as ``q``.  Internal math in
    fp32 for numerical stability of online softmax.
    """
    from tinygrad import Tensor
    from tinygrad.dtype import dtypes

    _, _, T, d = q.shape
    _, _, N, _ = k.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)

    internal_dtype = dtypes.float32
    q32 = q.cast(internal_dtype)
    k32 = k.cast(internal_dtype)
    v32 = v.cast(internal_dtype)

    accum_shape = (*q.shape[:-1], 1)  # (B, H, T, 1)
    out_shape = q.shape                # (B, H, T, d)

    # -inf initial m lets exp(m - m_new) = 0 when the first block's
    # m_block is finite; numpy reference uses the same.  TinyJit
    # requires buffer (not CONST) inputs, so realize the init tensors
    # before any runner.full / runner.masked call.
    m = Tensor.full(accum_shape, float("-inf"), dtype=internal_dtype, device=q.device).contiguous().realize()
    l = Tensor.zeros(*accum_shape, dtype=internal_dtype, device=q.device).contiguous().realize()
    out = Tensor.zeros(*out_shape, dtype=internal_dtype, device=q.device).contiguous().realize()

    for block_start in range(0, N, B_block):
        block_end = min(block_start + B_block, N)
        block_len = block_end - block_start

        k_block = k32[:, :, block_start:block_end, :]
        v_block = v32[:, :, block_start:block_end, :]

        # Queries abs positions [start_pos, start_pos+T).  Block covers
        # key abs positions [block_start, block_end).  Mask needed iff
        # any query attends to a subset of keys (causal and block
        # extends past start_pos).
        use_mask = causal and (block_end - 1 > start_pos)
        full_size = (block_len == B_block)
        use_jit = runner is not None and full_size

        if use_mask:
            mask = _build_boundary_mask(
                T=T, block_start=block_start, block_end=block_end,
                start_pos=start_pos, dtype=internal_dtype, device=q.device,
            )
            if use_jit:
                # In-place: runner mutates m, l, out.
                runner.masked(q32, k_block, v_block, mask, m, l, out)
            else:
                m, l, out = _block_update_masked(
                    q32, k_block, v_block, mask, m, l, out, inv_sqrt_d,
                )
        else:
            if use_jit:
                runner.full(q32, k_block, v_block, m, l, out)
            else:
                m, l, out = _block_update_full(
                    q32, k_block, v_block, m, l, out, inv_sqrt_d,
                )

    out_final = out / l
    return out_final.cast(q.dtype)


# --------------------------------------------------------------------- #
# TinyJit wrapper for the per-block updates
# --------------------------------------------------------------------- #

class FlashAttentionRunner:
    """Holds per-shape-signature ``TinyJit`` instances for the two block
    updates.  One runner per model (``inv_sqrt_d = 1/sqrt(head_dim)`` is
    fixed per model and is baked into every JIT at capture).

    ``.full(q, k, v, m, l, out)`` and ``.masked(q, k, v, mask, m, l, out)``
    look up (or lazily create) a ``TinyJit`` keyed by the tuple of input
    shapes and return the captured-then-replayed outputs.  Shapes must
    be fully int (no symbolic variables); this is the whole point of
    block tiling.

    TinyJit counter semantics (tinygrad engine/jit.py): cnt=0 pure
    eager, cnt=1 capture (eager + record), cnt>=2 replay.  The runner
    doesn't warm up automatically; first call is eager, second is
    capture, third+ are replay.
    """

    def __init__(self, inv_sqrt_d: float, beam_override: int | None = None):
        """``beam_override``: if set, wrap each JIT call in
        ``Context(BEAM=beam_override)`` so flash-attention inner kernels
        use a different BEAM level than the global ``JITBEAM``.  Useful
        when BEAM on one specific kernel hangs (see bd memory
        ``flash-attention-beam-action-58-hang-2026-04-20``): set to 0 to
        skip BEAM on FA kernels while keeping BEAM=2 on the rest of the
        model."""
        self._inv_sqrt_d = float(inv_sqrt_d)
        self._full_jits: dict = {}
        self._masked_jits: dict = {}
        self._beam_override = beam_override

    def _full_jit_for(self, key):
        if key in self._full_jits:
            return self._full_jits[key]
        from tinygrad import TinyJit

        inv = self._inv_sqrt_d

        # In-place assign pattern: m_io, l_io, out_io are running
        # accumulator buffers owned by the caller.  Compute all three
        # new values and realize them BEFORE any assign -- l_new and
        # out_new depend on m_prev (= m_io); if we assigned m_io first,
        # their later computation would re-read the already-overwritten
        # buffer.
        def _fn(q, k_block, v_block, m_io, l_io, out_io):
            from tinygrad import Tensor

            m_new, l_new, out_new = _block_update_full(
                q, k_block, v_block, m_io, l_io, out_io, inv,
            )
            Tensor.realize(m_new, l_new, out_new)
            m_io.assign(m_new).realize()
            l_io.assign(l_new).realize()
            out_io.assign(out_new).realize()

        jit = TinyJit(_fn)
        self._full_jits[key] = jit
        return jit

    def _masked_jit_for(self, key):
        if key in self._masked_jits:
            return self._masked_jits[key]
        from tinygrad import TinyJit

        inv = self._inv_sqrt_d

        def _fn(q, k_block, v_block, mask, m_io, l_io, out_io):
            from tinygrad import Tensor

            m_new, l_new, out_new = _block_update_masked(
                q, k_block, v_block, mask, m_io, l_io, out_io, inv,
            )
            Tensor.realize(m_new, l_new, out_new)
            m_io.assign(m_new).realize()
            l_io.assign(l_new).realize()
            out_io.assign(out_new).realize()

        jit = TinyJit(_fn)
        self._masked_jits[key] = jit
        return jit

    def _run(self, jit_fn, *args):
        if self._beam_override is None:
            jit_fn(*args)
            return
        from tinygrad.helpers import Context
        with Context(BEAM=self._beam_override):
            jit_fn(*args)

    def full(self, q, k_block, v_block, m_io, l_io, out_io):
        """Apply one fully-attended block update, mutating ``m_io``,
        ``l_io``, ``out_io`` in place."""
        # TinyJit rejects CONST-UOp inputs; contiguous() materializes
        # slices/constants into real buffers.
        q = q.contiguous()
        k_block = k_block.contiguous()
        v_block = v_block.contiguous()
        key = (q.shape, k_block.shape, m_io.shape, l_io.shape, out_io.shape)
        self._run(self._full_jit_for(key), q, k_block, v_block, m_io, l_io, out_io)

    def masked(self, q, k_block, v_block, mask, m_io, l_io, out_io):
        """Apply one boundary-block update (with mask), mutating
        ``m_io``, ``l_io``, ``out_io`` in place."""
        q = q.contiguous()
        k_block = k_block.contiguous()
        v_block = v_block.contiguous()
        mask = mask.contiguous()
        key = (q.shape, k_block.shape, mask.shape, m_io.shape, l_io.shape, out_io.shape)
        self._run(self._masked_jit_for(key), q, k_block, v_block, mask, m_io, l_io, out_io)


# --------------------------------------------------------------------- #
# Integration: monkey-patch tinygrad.apps.llm.TransformerBlock._attention
# --------------------------------------------------------------------- #

def patch_block_with_flash_attention(block: Any, runner: "FlashAttentionRunner", B_block: int = 256) -> None:
    """Rebind ``block._attention`` to route through tiled FlashAttention.

    Upstream ``TransformerBlock._attention`` computes attention via
    ``softmax(q @ k^T / sqrt(d) + triu_mask(start_pos+1)) @ v`` over
    the full K/V cache -- the triu mask axis is symbolic, which defeats
    BEAM kernel search.  This patched variant splits the attention into
    fixed-size K/V blocks so each inner kernel is fully int-shape.

    ``runner`` holds the JITs that dispatch each block update.  One
    runner per model is enough (all blocks share ``head_dim``).

    Idempotent: if already patched, no-op.
    """
    import types as _types

    if getattr(block, "_egg_flash_attention_patched", False):
        return

    def _flash_attention(self: Any, x: "Tensor", start_pos: "int") -> "Tensor":
        """Replacement for upstream ``TransformerBlock._attention``.

        Requires integer ``start_pos`` (not a UOp); tiled FlashAttention's
        block-count depends on a concrete value at dispatch time.  With
        block-fixed kernels, JIT reuses across all positions, so we don't
        need UOp start_pos for kernel-cache reuse anyway.
        """
        from tinygrad import Tensor
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

        # start_pos is an int here, so N is a Python int and the block
        # loop has a concrete count.
        N = int(start_pos) + T
        k_full = self.cache_kv[0, :, :, 0:N, :]
        v_full = self.cache_kv[1, :, :, 0:N, :]

        # GQA: expand k/v once to n_heads.  M5 will replace this with
        # in-kernel broadcast; for now, simple repeat_interleave matches
        # tinygrad upstream's enable_gqa=True behaviour.
        if self.n_heads != self.n_kv_heads:
            repeat = self.n_heads // self.n_kv_heads
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        attn = tiled_attention(
            q, k_full, v_full,
            start_pos=int(start_pos),
            B_block=B_block,
            causal=True,
            runner=runner,
        )
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        attn = self.attn_output(attn)
        return x + attn

    block._attention = _types.MethodType(_flash_attention, block)
    block._egg_flash_attention_patched = True
