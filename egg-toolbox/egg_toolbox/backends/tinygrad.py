from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .base import StepBackend, Tokenizer
from ..types import CompiledRequest


class TinygradTokenizer(Tokenizer):
    """Wraps tinygrad's tokenizer to our interface."""

    def __init__(self, inner: Any, eos_id: int, bos_id: int | None = None):
        self._inner = inner
        self._eos_id = eos_id
        self._bos_id = bos_id

    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._inner.decode(token_ids)

    def decode_single(self, token_id: int) -> str:
        return self._inner.decode([token_id])

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    @property
    def vocab_size(self) -> int:
        if hasattr(self._inner, "_normal_tokens") and hasattr(self._inner, "_special_tokens"):
            return len(self._inner._normal_tokens) + len(self._inner._special_tokens)
        return 0


class TinygradBackend(StepBackend):
    """Integration with tinygrad's Transformer.generate().

    This wraps tinygrad's existing token generation loop. The key integration
    point is that we yield token IDs one at a time from the generation.
    """

    def __init__(self):
        self._model: Any = None
        self._tokenizer: TinygradTokenizer | None = None
        self._model_name_str: str = ""
        self._chat_template: str = ""

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        from tinygrad import Tensor
        from tinygrad.apps.llm import SimpleTokenizer

        model_path_obj = Path(model_path)
        self._model_name_str = model_path_obj.stem

        # Build model from GGUF -- returns (model, kv_metadata).
        # We use _from_gguf_with_qkv_bias instead of Transformer.from_gguf
        # because tinygrad's default ignores attn_{q,k,v}.bias tensors,
        # which Qwen2/Qwen2.5 and other archs require for correct output.
        self._model, kv = _from_gguf_with_qkv_bias(Tensor(model_path_obj), **kwargs)

        # Extract chat template from GGUF metadata
        from ..template import ChatTemplate
        ct = ChatTemplate.from_gguf(model_path)
        self._chat_template = ct.source

        # Create tokenizer from GGUF kv metadata
        tokenizer = SimpleTokenizer.from_gguf_kv(kv)

        # Get token IDs from kv metadata
        eos_id = int(kv.get("tokenizer.ggml.eos_token_id", 0))
        bos_id_raw = kv.get("tokenizer.ggml.bos_token_id")
        bos_id = int(bos_id_raw) if bos_id_raw is not None else None

        self._tokenizer = TinygradTokenizer(tokenizer, eos_id=eos_id, bos_id=bos_id)

    def tokenizer(self) -> Tokenizer:
        assert self._tokenizer is not None, "Model not loaded"
        return self._tokenizer

    def chat_template_source(self) -> str:
        return self._chat_template

    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        """Wrap Transformer.generate() which already yields token IDs."""
        assert self._model is not None, "Model not loaded"
        for token_id in self._model.generate(list(request.prompt_tokens)):
            yield token_id

    def model_name(self) -> str:
        return self._model_name_str


def _from_gguf_with_qkv_bias(
    gguf,
    max_context: int | None = None,
    realize: bool = True,
    keep_packed: bool = False,
):
    """Load a GGUF into a tinygrad Transformer, preserving attn_{q,k,v}.bias.

    tinygrad's Transformer.from_gguf constructs bias-free Linear modules for
    the attention projections, so biases present in the GGUF (Qwen2, Qwen2.5,
    some others) are silently dropped and generation produces garbage.
    This loader mirrors tinygrad's from_gguf but attaches zero-initialised
    bias tensors before load_state_dict so those weights land in the model.

    Memory/speed tradeoff via ``keep_packed``:

    - ``keep_packed=False`` (default, matches upstream): call ``.contiguous()
      + realize()`` on every parameter.  Weights are dequantized to fp16
      dense tensors -> fast matmul, but an 8B Q4_0 GGUF occupies ~16 GB
      of VRAM regardless of the on-disk quantization.
    - ``keep_packed=True``: skip both calls.  Parameters remain lazy,
      referring to the packed GGUF-layout tensors on-device.  tinygrad's
      scheduler fuses the dequantize ops into each matmul kernel, so
      the packed weights stay put.  ~4x lower weight-memory on Q4_0;
      generation is slower because dequantize runs per-matmul.  The
      upstream loader has a comment "without this contiguous, it unpacks
      the weights from the model every time" explaining exactly this.
    """
    from tinygrad import Tensor, nn
    from tinygrad.apps.llm import Transformer
    from tinygrad.helpers import getenv

    kv, state_dict = nn.state.gguf_load(gguf.to(None))

    state_dict = {k: (v.cast("float16") if getenv("HALF", 1) else v) for k, v in state_dict.items()}
    if "output.weight" not in state_dict:
        state_dict["output.weight"] = state_dict["token_embd.weight"]

    arch = kv["general.architecture"]
    max_context = (
        min(max_context, kv[f"{arch}.context_length"])
        if max_context is not None
        else kv[f"{arch}.context_length"]
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

    model = Transformer(
        num_blocks=kv[f"{arch}.block_count"],
        dim=kv[f"{arch}.embedding_length"],
        hidden_dim=kv.get(
            f"{arch}.expert_feed_forward_length", kv[f"{arch}.feed_forward_length"]
        ),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        norm_eps=kv[f"{arch}.attention.layer_norm_rms_epsilon"],
        vocab_size=len(kv["tokenizer.ggml.tokens"]),
        head_dim=kv.get(
            f"{arch}.attention.key_length", kv[f"{arch}.embedding_length"] // n_heads
        ),
        rope_theta=kv[f"{arch}.rope.freq_base"],
        max_context=max_context,
        qk_norm=int(state_dict["blk.0.attn_q_norm.weight"].shape[0])
        if "blk.0.attn_q_norm.weight" in state_dict
        else 0,
        num_experts=kv.get(f"{arch}.expert_count", 0),
        num_experts_per_tok=kv.get(f"{arch}.expert_used_count", 0),
    )

    # Attach zero-init bias tensors so load_state_dict can populate them.
    # tinygrad's TransformerBlock builds Q/K/V Linear with bias=False.
    if "blk.0.attn_q.bias" in state_dict:
        half = getenv("HALF", 1)
        dtype = "float16" if half else state_dict["blk.0.attn_q.bias"].dtype
        for i, block in enumerate(model.blk):
            for proj in ("attn_q", "attn_k", "attn_v"):
                key = f"blk.{i}.{proj}.bias"
                if key in state_dict:
                    out_features = state_dict[key].shape[0]
                    getattr(block, proj).bias = Tensor.zeros(out_features, dtype=dtype)

    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
    if keep_packed:
        # Memory-efficient path: do NOT materialize the dequantized fp16
        # weights.  tinygrad's scheduler will fuse dequant ops into each
        # matmul, so packed GGUF weights stay on device.  Generation is
        # slower but weight memory drops to roughly the on-disk size.
        return model, kv
    for s in (params := nn.state.get_parameters(model)):
        s.replace(s.contiguous())
    if realize:
        Tensor.realize(*params)
    return model, kv
