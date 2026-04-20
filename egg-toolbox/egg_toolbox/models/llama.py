"""Llama-family architecture: dense GQA, RMSNorm, SwiGLU, rotary embeddings.

Covers the core tinygrad-compatible architectures:

- ``llama`` (Llama 2/3/3.1/3.2/3.3): Q/K weight rearrangement required.
- ``qwen2`` (Qwen 2/2.5): attention Q/K/V bias tensors required -- tinygrad's
  upstream TransformerBlock builds bias-free Linears and silently drops
  those weights, so we attach zero-init biases before load_state_dict so
  the GGUF values land in the model.
- ``qwen3``: qk_norm variant; tinygrad handles it already.
- MoE variants routed through this class work as long as tinygrad's
  ExpertWeights primitive is the right sparse-routing shape.

Registered with ``fallback=True`` so unfamiliar architectures try this
class first instead of erroring out at load time.  Architectures with
genuinely different attention (MLA, partial-RoPE) or post-norm layouts
(Gemma soft-capping) get their own module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Architecture, register

if TYPE_CHECKING:
    from tinygrad import Tensor, UOp


def _unbind_uop_to_int(sp: "UOp") -> int:
    """Extract the bound int value from a ``UOp.variable(...).bind(n)``.

    Flash-attention's Python-side block loop needs an int ``start_pos``.
    The backend sometimes passes a bound symbolic UOp (for the T=1
    decode JIT codepath); we unpack it here.

    Tinygrad shape: ``UOp(Ops.BIND, src=(DEFINE_VAR(...), CONST(n)))``.
    """
    # Ops.BIND source tuple is (DEFINE_VAR, CONST); the CONST's arg is
    # the bound value.
    if getattr(sp, "src", None) and len(sp.src) >= 2 and sp.src[1].arg is not None:
        return int(sp.src[1].arg)
    # Fallback: some UOps expose the int value via ``.arg`` directly.
    if getattr(sp, "arg", None) is not None and isinstance(sp.arg, int):
        return int(sp.arg)
    raise TypeError(f"cannot unbind UOp to int: {sp}")


@register("llama", "qwen2", "qwen3", fallback=True)
class LlamaArchitecture(Architecture):
    """Thin shell around ``tinygrad.apps.llm.Transformer`` that:

    1. Returns raw logits for the last position (not argmax); sampling
       happens in Python so SamplingParams take effect.
    2. Attaches zero-init Q/K/V bias tensors before load_state_dict so
       Qwen2/Qwen2.5 weights land in the model.
    3. Permutes Q/K weights from interleaved to half-split RoPE layout
       for ``arch == "llama"`` (matches upstream ``from_gguf`` logic).

    We compose rather than subclass Transformer because we want to
    override ``forward`` without touching upstream code, and keeping
    the wrapper separate makes the delegation explicit.
    """

    def __init__(self, transformer: Any):
        self._t = transformer
        # Expose tinygrad attributes so the backend and external callers
        # can walk ``blk``, ``max_context``, ``token_embd`` etc. exactly
        # as before.
        self.blk = transformer.blk
        self.token_embd = transformer.token_embd
        self.output_norm = transformer.output_norm
        self.output = transformer.output
        self.max_context = transformer.max_context

        # EGG_JIT_CHUNKS=1: route T>1 prefill chunks through TinyJit
        # with a UOp start_pos, so every chunk position collapses into
        # one captured graph and TinyJit replays it.  Requires the
        # upstream ``Tensor._tri`` int-shape assertion removed.
        #
        # EGG_FLASH_ATTENTION=1: swap each block's ``_attention`` for
        # a block-tiled FlashAttention variant that keeps all kernel
        # shapes int.  Every per-block kernel reuses the same JIT
        # across all positions, layers, and forwards.  This supersedes
        # the symbolic-mask approach (which depended on BEAM being able
        # to tune symbolic-axis kernels -- it can't).  See
        # ``docs/flash-attention-design.md``.
        import os as _os
        self._jit_chunks = _os.environ.get("EGG_JIT_CHUNKS", "0") != "0"
        self._flash_attention = _os.environ.get("EGG_FLASH_ATTENTION", "0") != "0"

        if self._flash_attention:
            import math as _math
            from .flash_attention import (
                FlashAttentionRunner, patch_block_with_flash_attention,
            )

            B_block = int(_os.environ.get("EGG_FLASH_BLOCK_SIZE", "256"))
            head_dim = transformer.blk[0].head_dim
            inv_sqrt_d = 1.0 / _math.sqrt(head_dim)

            # Optional BEAM override for just the FA inner kernels.
            # Useful when JITBEAM=2 causes tinygrad to hang on a
            # specific BEAM action for one of our flash kernels
            # (observed 2026-04-20 on the out-update kernel).  Set
            # EGG_FLASH_BEAM=0 to skip BEAM on FA kernels while
            # keeping BEAM=2 on the rest of the model.
            _beam_ov = _os.environ.get("EGG_FLASH_BEAM") or None
            beam_override = int(_beam_ov) if _beam_ov else None

            # One runner shared across all blocks: head_dim (and thus
            # inv_sqrt_d) is the same for every block in a Llama-family
            # model.  Each shape signature gets its own cached JIT.
            self._flash_runner = FlashAttentionRunner(
                inv_sqrt_d=inv_sqrt_d, beam_override=beam_override,
            )
            for block in self.blk:
                patch_block_with_flash_attention(
                    block, runner=self._flash_runner, B_block=B_block,
                )

        # Build JITs once at construction time.  tinygrad's TinyJit
        # captures ONE trace per instance (shape mismatches raise
        # ``JitError``), so we keep a SEPARATE TinyJit per (T value)
        # we need to accelerate: one for T=1 decode, one for T=chunk
        # prefill.  Both wrap the same ``_forward_logits`` but each
        # caches its own compiled graph.
        from tinygrad import TinyJit

        self.forward_jit = TinyJit(self._forward_logits)        # T=1 decode
        self.forward_jit_chunk = TinyJit(self._forward_logits)  # T=chunk

    def _forward_logits(self, tokens: "Tensor", start_pos: "int | UOp") -> "Tensor":
        x = self.token_embd(tokens)
        for block in self.blk:
            x = block(x, start_pos)
        return self.output(self.output_norm(x))[:, -1, :]

    def forward(self, tokens: "Tensor", start_pos: "int | UOp") -> "Tensor":
        """Prefill (T>1, int start_pos) or re-entrant call: eager path."""
        return self._forward_logits(tokens, start_pos)

    def __call__(self, tokens: "Tensor", start_pos: "int | UOp" = 0) -> "Tensor":
        """JIT when start_pos is a UOp, eager otherwise.

        Upstream tinygrad additionally gates on ``tokens.shape[1] == 1``
        in its own ``__call__`` because its ``_attention`` used to
        trip on ``triu``'s int-shape assertion.  With the upstream
        ``Tensor._tri`` assertion removed (its body is already
        symbolic-compatible), T>1 + UOp start_pos works natively, so
        ``EGG_JIT_CHUNKS=1`` just adds the JIT routing that upstream's
        gate would otherwise skip.

        Two JITs because TinyJit captures ONE trace per instance and
        raises ``JitError`` on shape mismatch: ``forward_jit`` for
        T=1 decode, ``forward_jit_chunk`` for T>1 chunks.
        """
        import os as _os
        from tinygrad import UOp, getenv

        debug = _os.environ.get("EGG_DEBUG_JIT", "0") != "0"
        path = "eager"
        try:
            # With flash attention on, the block-tiled path needs an
            # int ``start_pos`` (its Python loop count depends on it),
            # and it already provides per-block JIT reuse internally --
            # we get kernel-cache hits across positions without the
            # outer UOp-JIT.  Unbind if the caller passed a UOp.
            if self._flash_attention:
                if isinstance(start_pos, UOp):
                    start_pos = _unbind_uop_to_int(start_pos)
                path = f"flash-eager-T{tokens.shape[1]}"
                return self._forward_logits(tokens, start_pos)
            if getenv("JIT", 1) and isinstance(start_pos, UOp):
                if tokens.shape[1] == 1:
                    path = "jit-decode"
                    return self.forward_jit(tokens, start_pos)
                elif self._jit_chunks:
                    path = f"jit-chunk-T{tokens.shape[1]}"
                    return self.forward_jit_chunk(tokens, start_pos)
            return self._forward_logits(tokens, start_pos)
        finally:
            if debug:
                print(
                    f"[egg jit] path={path} T={tokens.shape[1]} "
                    f"start_pos_is_uop={isinstance(start_pos, UOp)}",
                    flush=True,
                )

    @classmethod
    def from_gguf_kv(
        cls,
        kv: dict,
        state_dict: dict,
        max_context: int | None = None,
        realize: bool = True,
        keep_packed: bool = False,
    ) -> tuple["LlamaArchitecture", dict]:
        """Construct an instance from pre-parsed GGUF kv and state_dict.

        Matches the signature of ``tinygrad.apps.llm.Transformer.from_gguf``
        except that the (kv, state_dict) have already been read off disk
        by the dispatcher, and the returned instance is a logits-returning
        wrapper with Q/K/V bias tensors attached where needed.

        Memory/speed tradeoff via ``keep_packed``:

        - ``keep_packed=False`` (default, matches upstream): call
          ``.contiguous() + realize()`` on every parameter.  Weights are
          dequantized to fp16 dense tensors -> fast matmul, but an 8B
          Q4_0 GGUF occupies ~16 GB of VRAM regardless of the on-disk
          quantization.
        - ``keep_packed=True``: skip both calls.  Parameters remain lazy,
          referring to the packed GGUF-layout tensors on-device.  tinygrad's
          scheduler fuses the dequantize ops into each matmul kernel, so
          the packed weights stay put.  ~4x lower weight-memory on Q4_0;
          generation is slower because dequantize runs per-matmul.  The
          upstream loader has a comment "without this contiguous, it
          unpacks the weights from the model every time" explaining
          exactly this.
        """
        from tinygrad import Tensor, nn
        from tinygrad.apps.llm import Transformer
        from tinygrad.helpers import getenv

        state_dict = {
            k: (v.cast("float16") if getenv("HALF", 1) else v)
            for k, v in state_dict.items()
        }
        if "output.weight" not in state_dict:
            state_dict["output.weight"] = state_dict["token_embd.weight"]

        arch = kv["general.architecture"]
        ctx_from_gguf = kv[f"{arch}.context_length"]
        max_context = (
            min(max_context, ctx_from_gguf) if max_context is not None else ctx_from_gguf
        )
        n_heads = kv[f"{arch}.attention.head_count"]
        n_kv_heads = kv[f"{arch}.attention.head_count_kv"]

        if arch == "llama":
            for name in state_dict:
                if "attn_q.weight" in name:
                    state_dict[name] = state_dict[name].rearrange(
                        "(n h two) d -> (n two h) d", n=n_heads, two=2
                    )
                if "attn_k.weight" in name:
                    state_dict[name] = state_dict[name].rearrange(
                        "(n h two) d -> (n two h) d", n=n_kv_heads, two=2
                    )

        transformer = Transformer(
            num_blocks=kv[f"{arch}.block_count"],
            dim=kv[f"{arch}.embedding_length"],
            hidden_dim=kv.get(
                f"{arch}.expert_feed_forward_length",
                kv[f"{arch}.feed_forward_length"],
            ),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=kv[f"{arch}.attention.layer_norm_rms_epsilon"],
            vocab_size=len(kv["tokenizer.ggml.tokens"]),
            head_dim=kv.get(
                f"{arch}.attention.key_length",
                kv[f"{arch}.embedding_length"] // n_heads,
            ),
            rope_theta=kv[f"{arch}.rope.freq_base"],
            max_context=max_context,
            qk_norm=int(state_dict["blk.0.attn_q_norm.weight"].shape[0])
            if "blk.0.attn_q_norm.weight" in state_dict
            else 0,
            num_experts=kv.get(f"{arch}.expert_count", 0),
            num_experts_per_tok=kv.get(f"{arch}.expert_used_count", 0),
        )

        # Attach zero-init Q/K/V bias tensors so load_state_dict can
        # populate them.  tinygrad's TransformerBlock builds Q/K/V
        # Linear with bias=False, so biases from the GGUF would
        # otherwise be silently dropped (Qwen2 / Qwen2.5 symptom:
        # random script-switching tokens).
        if "blk.0.attn_q.bias" in state_dict:
            half = getenv("HALF", 1)
            bias_dtype = (
                "float16" if half else state_dict["blk.0.attn_q.bias"].dtype
            )
            for i, block in enumerate(transformer.blk):
                for proj in ("attn_q", "attn_k", "attn_v"):
                    key = f"blk.{i}.{proj}.bias"
                    if key in state_dict:
                        out_features = state_dict[key].shape[0]
                        getattr(block, proj).bias = Tensor.zeros(
                            out_features, dtype=bias_dtype
                        )

        nn.state.load_state_dict(
            transformer, state_dict, verbose=False, consume=True, realize=False
        )
        if not keep_packed:
            params = nn.state.get_parameters(transformer)
            for s in params:
                s.replace(s.contiguous())
            if realize:
                Tensor.realize(*params)

        return cls(transformer), kv
