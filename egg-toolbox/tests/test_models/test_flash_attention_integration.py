"""M7 tests: LlamaArchitecture integration via EGG_FLASH_ATTENTION=1.

Constructs a small tinygrad TransformerBlock, patches it with
``patch_block_with_flash_attention``, and verifies the patched block's
output matches an unpatched equivalent (upstream attention) within
fp16 tolerance.

Epic: workspace-2r9.
"""
from __future__ import annotations

import math

import numpy as np


def _realize_block_params(block):
    """Realize all block parameters so they're stable buffers."""
    from tinygrad import nn
    for p in nn.state.get_parameters(block):
        p.replace(p.contiguous()).realize()


def test_patched_block_matches_upstream_decoder_step():
    """Patched block's _attention matches upstream block's _attention.

    Build two identical blocks (same random seed), patch one with flash
    attention, feed the same input, compare outputs.
    """
    from tinygrad import Tensor
    from tinygrad.apps.llm import TransformerBlock

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        patch_block_with_flash_attention,
    )

    dim = 64
    hidden_dim = 128
    n_heads = 4
    n_kv_heads = 2
    head_dim = dim // n_heads
    max_context = 256

    # Two blocks with IDENTICAL weights: create both, then copy weights.
    block_upstream = TransformerBlock(
        dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        norm_eps=1e-5, head_dim=head_dim, rope_theta=10000.0,
        max_context=max_context, qk_norm=0,
    )
    block_flash = TransformerBlock(
        dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        norm_eps=1e-5, head_dim=head_dim, rope_theta=10000.0,
        max_context=max_context, qk_norm=0,
    )
    _realize_block_params(block_upstream)
    _realize_block_params(block_flash)

    # Copy weights from upstream to flash block so outputs are comparable.
    from tinygrad import nn
    state_upstream = nn.state.get_state_dict(block_upstream)
    nn.state.load_state_dict(block_flash, state_upstream, verbose=False, realize=True)

    # Patch the second block.
    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(head_dim))
    patch_block_with_flash_attention(block_flash, runner=runner, B_block=32)

    # Feed a few consecutive decode steps: start_pos=0, then 1, 2, ...
    B, T = 1, 1
    rng = np.random.default_rng(42)

    start_positions = [0, 1, 8, 31, 32, 33, 64]
    for start_pos in start_positions:
        x_np = rng.normal(0, 1, size=(B, T, dim)).astype(np.float32)
        x_u = Tensor(x_np.copy())
        x_f = Tensor(x_np.copy())

        out_u = block_upstream(x_u, start_pos).realize().numpy()
        out_f = block_flash(x_f, start_pos).realize().numpy()

        # fp32 model + fp32 inputs → tolerance well within float32
        np.testing.assert_allclose(
            out_f, out_u, atol=1e-3, rtol=1e-3,
            err_msg=f"mismatch at start_pos={start_pos}",
        )


def test_patched_block_prefill_chunk():
    """Patched block handles T>1 prefill chunks correctly."""
    from tinygrad import Tensor, nn
    from tinygrad.apps.llm import TransformerBlock

    from egg_toolbox.models.flash_attention import (
        FlashAttentionRunner,
        patch_block_with_flash_attention,
    )

    dim, hidden_dim = 64, 128
    n_heads, n_kv_heads = 4, 2
    head_dim = dim // n_heads
    max_context = 512

    block_u = TransformerBlock(
        dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        norm_eps=1e-5, head_dim=head_dim, rope_theta=10000.0,
        max_context=max_context, qk_norm=0,
    )
    block_f = TransformerBlock(
        dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        norm_eps=1e-5, head_dim=head_dim, rope_theta=10000.0,
        max_context=max_context, qk_norm=0,
    )
    _realize_block_params(block_u)
    _realize_block_params(block_f)
    nn.state.load_state_dict(
        block_f, nn.state.get_state_dict(block_u), verbose=False, realize=True,
    )

    runner = FlashAttentionRunner(inv_sqrt_d=1.0 / math.sqrt(head_dim))
    patch_block_with_flash_attention(block_f, runner=runner, B_block=64)

    # Chunked prefill pattern: chunk 0 at start_pos=0 T=32,
    # chunk 1 at start_pos=32 T=32, chunk 2 at start_pos=64 T=32.
    rng = np.random.default_rng(111)
    for start_pos in (0, 32, 64, 96):
        x_np = rng.normal(0, 1, size=(1, 32, dim)).astype(np.float32)
        out_u = block_u(Tensor(x_np.copy()), start_pos).realize().numpy()
        out_f = block_f(Tensor(x_np.copy()), start_pos).realize().numpy()
        np.testing.assert_allclose(
            out_f, out_u, atol=1e-3, rtol=1e-3,
            err_msg=f"mismatch at start_pos={start_pos}",
        )


def test_llama_architecture_opt_in_via_env(monkeypatch):
    """EGG_FLASH_ATTENTION=1 triggers patching at LlamaArchitecture init."""
    from tinygrad.apps.llm import Transformer

    monkeypatch.setenv("EGG_FLASH_ATTENTION", "1")
    monkeypatch.setenv("EGG_FLASH_BLOCK_SIZE", "128")

    t = Transformer(
        num_blocks=2, dim=32, hidden_dim=64,
        n_heads=4, n_kv_heads=2, norm_eps=1e-5,
        vocab_size=100, head_dim=8, rope_theta=10000.0,
        max_context=128, qk_norm=0,
    )
    # Realize its params
    from tinygrad import nn
    for p in nn.state.get_parameters(t):
        p.replace(p.contiguous()).realize()

    from egg_toolbox.models.llama import LlamaArchitecture
    arch = LlamaArchitecture(t)

    assert arch._flash_attention is True
    for block in arch.blk:
        assert getattr(block, "_egg_flash_attention_patched", False)


def test_llama_architecture_default_off(monkeypatch):
    """Without EGG_FLASH_ATTENTION, blocks are not patched."""
    from tinygrad.apps.llm import Transformer

    monkeypatch.delenv("EGG_FLASH_ATTENTION", raising=False)

    t = Transformer(
        num_blocks=2, dim=32, hidden_dim=64,
        n_heads=4, n_kv_heads=2, norm_eps=1e-5,
        vocab_size=100, head_dim=8, rope_theta=10000.0,
        max_context=128, qk_norm=0,
    )
    from tinygrad import nn
    for p in nn.state.get_parameters(t):
        p.replace(p.contiguous()).realize()

    from egg_toolbox.models.llama import LlamaArchitecture
    arch = LlamaArchitecture(t)

    assert arch._flash_attention is False
    for block in arch.blk:
        assert not getattr(block, "_egg_flash_attention_patched", False)
