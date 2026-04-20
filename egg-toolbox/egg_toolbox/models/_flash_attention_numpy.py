"""Numpy reference implementation of block-tiled online-softmax attention.

This is the mathematical reference that the tinygrad port (M2+) will
mirror.  It exists to (a) prove the online-softmax algorithm is correct
vs. a naive ``softmax(Q K^T + mask) V``, and (b) give us a fixed-seed
oracle that later milestones can diff against bit-exactly in fp32.

Algorithm: FlashAttention-1 online softmax, tiled over the key/value
sequence dimension.  For each block of ``B_block`` key/value positions,
maintain a running ``(m, l, out)`` where:

  m : running row-wise maximum of the pre-softmax scores
  l : running row-wise sum of ``exp(scores - m)`` (normalizer)
  out : running numerator of ``softmax(scores) @ V``

Per block:

  s       = (q @ k_block^T) / sqrt(d)              # (..., T, block_len)
  s      += mask                                    # if any
  m_block = max(s, axis=-1, keepdims=True)          # (..., T, 1)
  m_new   = max(m_running, m_block)
  alpha   = exp(m_running - m_new)                  # correction factor
  p_tilde = exp(s - m_new)                          # un-normalized probs
  l_new   = l_running * alpha + p_tilde.sum(-1, keepdims=True)
  out_new = out_running * alpha + p_tilde @ v_block

At end: ``out_final = out_running / l_running``.

See ``docs/flash-attention-design.md`` for the full design.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------- #
# Naive reference (used only to validate the tiled version)
# --------------------------------------------------------------------- #

def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def naive_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Textbook attention: ``softmax(q @ k^T / sqrt(d) + mask) @ v``.

    Shapes:
      q: (..., T, d)
      k, v: (..., N, d)
      mask: (T, N) or None.  Added element-wise; use ``-inf`` to block.

    Returns: (..., T, d).
    """
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -2, -1) / np.sqrt(d)
    if mask is not None:
        scores = scores + mask
    probs = _softmax(scores, axis=-1)
    return probs @ v


# --------------------------------------------------------------------- #
# Tiled online-softmax reference
# --------------------------------------------------------------------- #

MaskFn = "callable[[int, int], np.ndarray | None]"


def tiled_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask_fn=None,
    B_block: int = 128,
) -> np.ndarray:
    """Block-tiled attention with online-softmax accumulation.

    Shapes same as ``naive_attention``.  ``mask_fn(block_start, block_end)``
    returns a ``(T, block_end - block_start)`` mask or ``None`` for fully
    attended blocks.  When ``mask_fn`` returns ``None``, no mask is
    applied to that block (faster path that the tinygrad port will also
    take via a separate JIT).

    Blocks that are entirely masked (all ``-inf`` after ``mask_fn``)
    should be skipped by the caller via ``mask_fn`` returning a sentinel
    ``"skip"`` -- but our simple reference treats them numerically: the
    contribution is zero because ``exp(-inf - m_new) = 0``.  The one
    failure mode is an entire forward where every single block is
    masked (``l`` stays zero, divide-by-zero at finalize); that doesn't
    arise in causal attention, which always has at least one attended
    position per query.
    """
    # Use fp64 internally for the online softmax math to match naive at
    # ~1e-12 tolerance.  Inputs stay whatever dtype the caller passes.
    work_dtype = np.float64

    q64 = q.astype(work_dtype)
    k64 = k.astype(work_dtype)
    v64 = v.astype(work_dtype)

    T = q.shape[-2]
    d = q.shape[-1]
    N = k.shape[-2]
    batch_shape = q.shape[:-2]
    inv_sqrt_d = 1.0 / np.sqrt(d)

    m_running = np.full(batch_shape + (T, 1), -np.inf, dtype=work_dtype)
    l_running = np.zeros(batch_shape + (T, 1), dtype=work_dtype)
    out_running = np.zeros(batch_shape + (T, d), dtype=work_dtype)

    for block_start in range(0, N, B_block):
        block_end = min(block_start + B_block, N)
        k_block = k64[..., block_start:block_end, :]
        v_block = v64[..., block_start:block_end, :]

        s = q64 @ np.swapaxes(k_block, -2, -1) * inv_sqrt_d

        if mask_fn is not None:
            mask = mask_fn(block_start, block_end)
            if mask is not None:
                # If block is entirely masked (-inf everywhere), skip to
                # avoid nan propagation from (-inf) - (-inf).  Caller
                # should prefer signalling this via skip but defence in
                # depth is cheap.
                if np.all(np.isneginf(mask)):
                    continue
                s = s + mask

        m_block = s.max(axis=-1, keepdims=True)
        m_new = np.maximum(m_running, m_block)

        # alpha handles the m_running = -inf init on the first useful
        # block: exp(-inf - finite) = 0, so the pre-existing (zero)
        # accumulator contributes zero -- correct.
        alpha = np.exp(m_running - m_new)
        p_tilde = np.exp(s - m_new)

        l_running = l_running * alpha + p_tilde.sum(axis=-1, keepdims=True)
        out_running = out_running * alpha + p_tilde @ v_block
        m_running = m_new

    return (out_running / l_running).astype(q.dtype)


# --------------------------------------------------------------------- #
# Causal-mask helpers (the shape egg-toolbox actually calls with)
# --------------------------------------------------------------------- #

def make_causal_mask_fn(T: int, start_pos: int, N: int):
    """Build a ``mask_fn`` for causal attention as used in chunked prefill.

    Query positions are absolute ``[start_pos, start_pos + T)``.
    Key positions are absolute ``[0, N)`` (note: for prefill chunks
    ``N == start_pos + T``).

    Attend iff ``key_abs_pos <= query_abs_pos``.
    """
    assert N >= start_pos + T, f"N={N} < start_pos+T={start_pos + T}; query positions out of range of keys"

    def mask_fn(block_start: int, block_end: int):
        block_len = block_end - block_start

        # Fully attended: last query (absolute start_pos + T - 1) attends
        # to everything, so if last key in block (absolute block_end - 1)
        # <= start_pos (first query), the whole block is unrestricted.
        if block_end - 1 <= start_pos:
            return None  # signal: no mask needed

        # Construct the mask tensor.
        rows = np.arange(T, dtype=np.int64).reshape(T, 1)            # query-local row idx
        cols = np.arange(block_len, dtype=np.int64).reshape(1, block_len)  # key-local col idx

        # key_abs = block_start + cols, query_abs = start_pos + rows
        # attend iff key_abs <= query_abs
        #         <=> cols <= (start_pos - block_start) + rows
        threshold = (start_pos - block_start) + rows  # (T, 1)
        allowed = cols <= threshold
        mask = np.where(allowed, 0.0, -np.inf).astype(np.float64)
        return mask

    return mask_fn


def make_boolean_causal_mask(T: int, start_pos: int, N: int) -> np.ndarray:
    """Full ``(T, N)`` causal mask matrix.  For the naive reference."""
    rows = np.arange(T, dtype=np.int64).reshape(T, 1)
    cols = np.arange(N, dtype=np.int64).reshape(1, N)
    threshold = start_pos + rows
    allowed = cols <= threshold
    return np.where(allowed, 0.0, -np.inf).astype(np.float64)
