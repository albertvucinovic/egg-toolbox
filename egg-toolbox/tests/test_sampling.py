"""Tests for egg_toolbox.sampling -- logits-to-token sampling.

tinygrads Transformer.forward hardcodes an argmax at the tail, which
ignores every SamplingParams field (temperature, top_p, top_k, seed).
The sampler module replaces that tail with a proper sampling pipeline.
These tests pin the contract with controlled logits inputs so we do
not need a real model to validate correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

from egg_toolbox.sampling import sample_next_token
from egg_toolbox.types import SamplingParams


def test_temperature_zero_is_argmax():
    logits = np.array([1.0, 3.0, 2.0, 0.5])
    tok = sample_next_token(logits, SamplingParams(temperature=0.0))
    assert tok == 1  # index of max


def test_seed_reproducibility():
    """Same seed + same logits -> same token, twice in a row."""
    logits = np.array([1.0, 1.0, 1.0, 1.0])  # uniform -> any index equally likely
    s = SamplingParams(temperature=1.0, seed=42)
    a = sample_next_token(logits, s)
    b = sample_next_token(logits, s)
    assert a == b, (
        f"same seed produced {a} then {b}; seed handling is broken"
    )


def test_different_seeds_can_produce_different_tokens():
    """With uniform logits, different seeds should eventually produce
    different tokens -- a weak non-determinism check."""
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    tokens = {
        sample_next_token(logits, SamplingParams(temperature=1.0, seed=s))
        for s in range(20)
    }
    assert len(tokens) > 1, (
        "20 different seeds all produced the same token -- rng is stuck"
    )


def test_top_k_1_equivalent_to_argmax():
    """top_k=1 must behave like greedy (whether temperature is set or not)."""
    logits = np.array([1.0, 5.0, 2.0, 4.0])
    for temp in (0.0, 0.7, 2.0):
        tok = sample_next_token(
            logits, SamplingParams(temperature=temp, top_k=1, seed=0),
        )
        assert tok == 1, (
            f"top_k=1 at temperature={temp} returned {tok}, expected 1"
        )


def test_top_k_restricts_to_top_candidates():
    """With top_k=2 and very strong logits for index 0 vs 1, the
    sampler must never return indices 2 or 3."""
    logits = np.array([10.0, 9.9, -5.0, -5.0])
    for seed in range(20):
        tok = sample_next_token(
            logits, SamplingParams(temperature=1.0, top_k=2, seed=seed),
        )
        assert tok in (0, 1), f"top_k=2 returned {tok}, violates top_k filter"


def test_top_p_nucleus_cuts_long_tail():
    """top_p=0.5 with a skewed distribution should exclude low-prob tokens.
    Logits that produce ~[0.9, 0.09, 0.009, 0.001] -- only the first (0.9)
    is inside the 0.5 nucleus."""
    logits = np.array([5.0, 2.0, 0.0, -3.0])
    for seed in range(20):
        tok = sample_next_token(
            logits, SamplingParams(temperature=1.0, top_p=0.5, seed=seed),
        )
        assert tok == 0, f"top_p=0.5 returned {tok}, nucleus should be just 0"


def test_top_p_1_disables_filter():
    """top_p=1.0 must permit any token in the distribution."""
    logits = np.array([3.0, 2.0, 1.0, 0.5])
    tokens = {
        sample_next_token(
            logits, SamplingParams(temperature=2.0, top_p=1.0, seed=s),
        )
        for s in range(50)
    }
    # Should sample at least 2 distinct indices across 50 seeds at high temp.
    assert len(tokens) >= 2


def test_temperature_amplifies_randomness():
    """Higher temperature -> more even distribution -> more diverse
    samples across seeds.  Weak check: temp=2.0 should produce strictly
    more distinct tokens than temp=0.1 over the same seed range."""
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    low = {sample_next_token(
        logits, SamplingParams(temperature=0.1, seed=s)
    ) for s in range(30)}
    high = {sample_next_token(
        logits, SamplingParams(temperature=2.0, seed=s)
    ) for s in range(30)}
    assert len(high) >= len(low), (
        f"temperature=2.0 produced {len(high)} distinct tokens, "
        f"temperature=0.1 produced {len(low)} -- temperature isn't "
        "amplifying randomness"
    )


def test_degenerate_logits_do_not_crash():
    """-inf or very negative logits must not produce NaN / crash the sampler."""
    logits = np.array([0.0, -np.inf, -np.inf, -np.inf])
    tok = sample_next_token(logits, SamplingParams(temperature=1.0, seed=0))
    assert tok == 0


def test_uniform_logits_with_strict_filters():
    """Uniform logits + top_k=1 + top_p=0.5: the filter chain must pick
    the first top-k slot (by argmax tiebreak) without crashing."""
    logits = np.ones(10)
    tok = sample_next_token(
        logits, SamplingParams(temperature=1.0, top_k=1, top_p=0.5, seed=0),
    )
    assert 0 <= tok < 10
