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

        # Opt-in: patch each block's ``_attention`` with a symbolic-
        # friendly causal mask so T>1 forwards with UOp start_pos
        # become JIT-eligible (tinygrad upstream gates T>1 on int
        # start_pos because ``triu`` rejects symbolic shapes).  With
        # EGG_JIT_CHUNKS=1, chunked-prefill submits every chunk via
        # the JITted graph: ~680 per-kernel Python dispatches per
        # chunk collapse into one graph submission, ~8s -> ~400ms.
        # Off by default while we measure stability; see
        # egg_toolbox/models/symbolic_attention.py for the risks.
        import os as _os
        self._jit_chunks = _os.environ.get("EGG_JIT_CHUNKS", "0") != "0"
        if self._jit_chunks:
            from .symbolic_attention import patch_block_attention
            for block in self.blk:
                patch_block_attention(block, max_context=self.max_context)

        # Build the JIT once at construction time with our logits-returning
        # forward.  tinygrad's Transformer builds its own ``forward_jit``
        # over the argmax-folded forward; we replace it so the backend's
        # decode-path kernel compiles against ``_forward_logits``.
        from tinygrad import TinyJit

        self.forward_jit = TinyJit(self._forward_logits)

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
        because its ``_attention`` uses ``triu`` for the causal mask,
        which rejects symbolic shape dims.  When ``EGG_JIT_CHUNKS=1``
        we've patched each block's ``_attention`` to use an arange-
        based mask instead, so T>1 with UOp start_pos also works --
        one symbolic JIT trace covers every chunk position.
        """
        from tinygrad import UOp, getenv

        if getenv("JIT", 1) and isinstance(start_pos, UOp):
            if tokens.shape[1] == 1 or self._jit_chunks:
                return self.forward_jit(tokens, start_pos)
        return self._forward_logits(tokens, start_pos)

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
