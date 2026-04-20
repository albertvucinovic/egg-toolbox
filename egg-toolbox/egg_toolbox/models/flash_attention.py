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

    ``.contiguous()`` barriers split the accumulator updates into
    distinct kernels -- the fused ``(out_prev * alpha) + matmul``
    kernel signature reproducibly hangs tinygrad's BEAM search on the
    specific action it picks, so we avoid generating it.  Splitting
    costs ~1 extra kernel launch per block but unblocks BEAM tuning.
    """
    s = q.matmul(k_block.transpose(-2, -1)) * inv_sqrt_d             # (B, H, T, block_len)
    m_block = s.max(axis=-1, keepdim=True)                            # (B, H, T, 1)
    m_new = m_prev.maximum(m_block)                                   # (B, H, T, 1)
    alpha = (m_prev - m_new).exp()                                    # (B, H, T, 1)
    p_tilde = (s - m_new).exp()                                       # (B, H, T, block_len)
    l_new = l_prev * alpha + p_tilde.sum(axis=-1, keepdim=True)

    # Split fused mul+matmul+add so tinygrad can't produce the BEAM-
    # hanging kernel pattern.  ``.contiguous()`` forces a realize barrier.
    scaled_prev = (out_prev * alpha).contiguous()
    new_contrib = p_tilde.matmul(v_block)
    out_new = scaled_prev + new_contrib
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

    scaled_prev = (out_prev * alpha).contiguous()
    new_contrib = p_tilde.matmul(v_block)
    out_new = scaled_prev + new_contrib
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
        # TinyJit wraps each capture in ``Context(BEAM=getenv("JITBEAM",
        # BEAM.value))`` (tinygrad/engine/jit.py:300), which blows away
        # any outer ``Context(BEAM=...)`` we set.  And ``getenv`` is
        # ``@functools.cache``d, so just mutating os.environ doesn't
        # take effect either.  So: mutate env, invalidate the cache,
        # then restore.
        import os as _os
        from tinygrad.helpers import getenv

        old_jitbeam = _os.environ.get("JITBEAM")
        _os.environ["JITBEAM"] = str(self._beam_override)
        getenv.cache_clear()
        try:
            jit_fn(*args)
        finally:
            if old_jitbeam is None:
                _os.environ.pop("JITBEAM", None)
            else:
                _os.environ["JITBEAM"] = old_jitbeam
            getenv.cache_clear()

    def full(self, q, k_block, v_block, m_io, l_io, out_io):
        """Apply one fully-attended block update, mutating ``m_io``,
        ``l_io``, ``out_io`` in place."""
        # ``.contiguous().realize()`` forces each input into a fresh
        # standalone buffer (no slice/padded-view metadata).  tinygrad
        # BEAM search otherwise mishandles padded tensors from slices
        # of a larger backing (per upstream dev guidance).
        q = q.contiguous().realize()
        k_block = k_block.contiguous().realize()
        v_block = v_block.contiguous().realize()
        key = (q.shape, k_block.shape, m_io.shape, l_io.shape, out_io.shape)
        self._run(self._full_jit_for(key), q, k_block, v_block, m_io, l_io, out_io)

    def masked(self, q, k_block, v_block, mask, m_io, l_io, out_io):
        """Apply one boundary-block update (with mask), mutating
        ``m_io``, ``l_io``, ``out_io`` in place."""
        q = q.contiguous().realize()
        k_block = k_block.contiguous().realize()
        v_block = v_block.contiguous().realize()
        mask = mask.contiguous().realize()
        key = (q.shape, k_block.shape, mask.shape, m_io.shape, l_io.shape, out_io.shape)
        self._run(self._masked_jit_for(key), q, k_block, v_block, mask, m_io, l_io, out_io)


# --------------------------------------------------------------------- #
# Integration: monkey-patch tinygrad.apps.llm.TransformerBlock._attention
# --------------------------------------------------------------------- #

def patch_block_with_flash_attention(block: Any, runner: "FlashAttentionRunner | None" = None, B_block: int = 256) -> None:
    """Rebind ``block._attention`` to route through padded-full-context
    attention.

    Compared to upstream (``q @ k[:N]^T + triu_mask(start_pos+1)`` which
    varies the K/V axis with ``start_pos``), this variant keeps every
    attention kernel's K/V axis fixed at ``max_context``.  The mask is
    built from ``arange(max_context) <= start_pos + arange(T)`` so the
    "future + beyond-valid-positions" contribution decays to zero in
    softmax while the compute graph stays shape-stable.

    Rationale: upstream tinygrad BEAM mis-searches tensors that are
    sliced views over a larger backing (the padded-tensor bug the
    tinygrad dev acknowledged).  Skipping the slice eliminates the
    padded-view entirely -- ``cache_kv`` IS the buffer, used at full
    shape.  Every forward produces the same kernel signatures, so BEAM
    tunes once and every subsequent call is a cache hit.

    Compute cost: at start_pos << max_context, we do extra work on
    positions that contribute zero to the output.  This is amortised
    because (a) GPU matmul throughput is high relative to Python
    dispatch overhead, (b) block-tiled FA was ~5x worse due to
    per-block dispatch cost, (c) once ``start_pos`` is reasonable
    (say > max_context/4), waste is < 4x.

    ``runner`` and ``B_block`` are kept for signature compatibility
    with the earlier block-tiled implementation; both are unused.

    Idempotent: if already patched, no-op.
    """
    import types as _types

    if getattr(block, "_egg_flash_attention_patched", False):
        return

    def _padded_attention(self: Any, x: "Tensor", start_pos) -> "Tensor":
        """Replacement for upstream ``TransformerBlock._attention``.

        Uses the full ``cache_kv[:, :, :, :, :]`` (max_context-wide)
        every call and masks out positions beyond ``start_pos + T``.
        Kernel shapes never vary with ``start_pos`` -- one BEAM search
        per signature serves every position.

        Accepts int OR symbolic UOp ``start_pos``.  When the caller
        passes a UOp, the mask's threshold (``start_pos + rows``) is
        part of the graph as a symbolic value -- same graph topology
        for every call, only the bound value varies.  tinygrad reuses
        the compiled kernels across positions without recompile.
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

        # Use the FULL max_context cache_kv.  No slice, no padded view.
        # Every forward sees K, V with shape (B, n_kv, max_context, d).
        k_full = self.cache_kv[0]
        v_full = self.cache_kv[1]

        # Build a causal + future-mask at fixed (T, max_context) shape.
        # Attend iff ``col <= start_pos + row``: this hides future
        # positions (j > start_pos + i) AND any cache_kv entries at
        # indices >= start_pos + T (either zeros from init or stale).
        mask = _build_padded_mask(
            T=T, start_pos=start_pos, max_context=self.max_context,
            dtype=x.dtype, device=x.device,
        )

        attn = q.scaled_dot_product_attention(
            k_full, v_full, attn_mask=mask, enable_gqa=True,
        )
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        attn = self.attn_output(attn)
        return x + attn

    block._attention = _types.MethodType(_padded_attention, block)
    block._egg_flash_attention_patched = True


def _build_padded_mask(T: int, start_pos, max_context: int, dtype: Any, device: Any) -> "Tensor":
    """Build the (T, max_context) attention mask at fixed shape.

    Attend (mask = 0) iff ``col_abs <= start_pos + row``.  All other
    positions get ``-inf`` so softmax assigns them zero weight.

    ``start_pos`` may be a Python int or a tinygrad UOp (e.g. bound
    symbolic variable).  When UOp, the threshold enters the graph as
    a symbolic value; same graph topology across all calls -- only
    the bound value varies.  Shape stays int-constant either way, so
    BEAM compiles this path once per ``(T, max_context, dtype)``
    signature.
    """
    from tinygrad import Tensor
    from tinygrad.dtype import dtypes

    rows = Tensor.arange(T, dtype=dtypes.int32, device=device).reshape(T, 1)
    cols = Tensor.arange(max_context, dtype=dtypes.int32, device=device).reshape(1, max_context)
    allowed = cols <= (rows + start_pos)
    zero = Tensor.full((1,), 0.0, dtype=dtype, device=device)
    minus_inf = Tensor.full((1,), float("-inf"), dtype=dtype, device=device)
    return allowed.where(zero, minus_inf)
