"""Token sampling: temperature, top_p, top_k from logits.

tinygrads ``Transformer.forward`` hardcodes ``argmax`` at the tail,
which ignores every sampling parameter.  This module replaces that
tail: given a logits array and a ``SamplingParams``, produce the
next token id.

Pure numpy.  The logits tensor from tinygrad is copied to numpy via
``.numpy()`` once per decode step; cost is negligible next to the
forward pass even at large vocab sizes.

Per-request determinism: construct a fresh ``numpy.random.Generator``
seeded from ``SamplingParams.seed`` so concurrent requests don't leak
into one another (process-global ``np.random.seed`` would).
"""
from __future__ import annotations

import numpy as np

from .types import SamplingParams


_TINY = 1e-20  # floor for numerical safety


def sample_next_token(
    logits: np.ndarray,
    sampling: SamplingParams,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a token id from ``logits`` under ``sampling``.

    ``logits`` is a 1-D float array of length ``vocab_size`` (pre-softmax).
    If ``rng`` is None one is built from ``sampling.seed`` (None seed ->
    OS entropy).  Callers that generate many tokens in one request should
    pass a reused rng so the seeding cost is paid once.
    """
    if rng is None:
        rng = _rng_for(sampling.seed)

    # Greedy shortcut: temperature=0 or top_k=1 both collapse to argmax.
    if sampling.temperature <= 0.0 or sampling.top_k == 1:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float32, copy=False) / sampling.temperature

    # Top-k: drop every logit below the k-th largest.
    if sampling.top_k > 0 and sampling.top_k < scaled.shape[0]:
        kth = np.partition(scaled, -sampling.top_k)[-sampling.top_k]
        scaled = np.where(scaled >= kth, scaled, -np.inf)

    # Stable softmax.
    finite_max = scaled[np.isfinite(scaled)].max() if np.isfinite(scaled).any() else 0.0
    probs = np.exp(scaled - finite_max)
    probs_sum = probs.sum()
    if probs_sum <= 0.0:
        # All -inf after filtering -- degenerate case, fall back to greedy.
        return int(np.argmax(logits))
    probs = probs / probs_sum

    # Top-p (nucleus): drop the long tail whose cumulative mass exceeds p.
    if 0.0 < sampling.top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        # Keep the smallest prefix whose sum reaches top_p, plus the token
        # that crossed the threshold (so the nucleus is never empty).
        cutoff = int(np.searchsorted(cumulative, sampling.top_p) + 1)
        cutoff = max(cutoff, 1)
        keep = np.zeros_like(probs, dtype=bool)
        keep[order[:cutoff]] = True
        probs = np.where(keep, probs, 0.0)
        norm = probs.sum()
        if norm <= 0.0:
            return int(np.argmax(logits))
        probs = probs / norm

    # Inverse-CDF sample using the per-request rng.
    u = rng.random()
    cumulative = np.cumsum(probs)
    return int(np.searchsorted(cumulative, u).clip(0, probs.shape[0] - 1))


def _rng_for(seed: int | None) -> np.random.Generator:
    """Build a numpy Generator.  None seed -> OS entropy (non-reproducible).

    We deliberately use Generator rather than ``np.random.seed`` because
    the latter touches a process-global state and would leak randomness
    between concurrent requests.
    """
    return np.random.default_rng(seed)
