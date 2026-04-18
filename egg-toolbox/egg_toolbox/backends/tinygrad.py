from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from .base import StepBackend, Tokenizer
from ..types import CompiledRequest


# All tinygrad calls (model load + every generate_tokens invocation)
# must happen on the same OS thread.  tinygrad caches its
# kernel-compile SQLite connection in a process global, but sqlite3
# rejects cross-thread use.  We pin everything to a single dedicated
# worker thread here.
_TINYGRAD_THREAD_NAME = "egg-tinygrad"


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

        # Dedicated worker thread for ALL tinygrad operations.  See
        # module-level comment above -- tinygrad's sqlite kernel cache
        # is thread-bound, so load_model AND every generate_tokens
        # iteration must run on the same OS thread.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=_TINYGRAD_THREAD_NAME,
        )

        # Cancellation flag for the currently-active generation.
        # set() from any thread by cancel_generation(); the worker
        # _run loop checks it after every yielded token.  We cannot
        # use generator.close() for this because close() is a
        # cross-thread operation on a generator that might be
        # currently executing on another thread ("generator already
        # executing" error), and silently swallowing that error
        # left cancellation broken.
        self._active_cancel: threading.Event | None = None

        # Prompt-prefix KV cache bookkeeping.  tinygrad's
        # TransformerBlock keeps a persistent ``cache_kv`` tensor
        # keyed by (start_pos, T); upstream Transformer.generate
        # always passes start_pos=0 so the cache is overwritten from
        # scratch every request.  We drive the forward path ourselves
        # and keep _cache_tokens = the ordered tokens whose K/V
        # currently sit in every block's cache_kv[0:len).  On a new
        # request we feed only the suffix past the longest common
        # prefix; on exception/cancel we reset to [] to stay
        # consistent with whatever the cache now contains.  Setting
        # EGG_PREFIX_CACHE=0 disables the whole mechanism.
        self._cache_tokens: list[int] = []

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        # Hop onto the dedicated tinygrad thread so the SQLite kernel
        # cache connection (opened lazily inside from_gguf's realize())
        # is created there.
        self._executor.submit(self._do_load_model, model_path, **kwargs).result()

    def _do_load_model(self, model_path: str, **kwargs: Any) -> None:
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
        """Yield token IDs one at a time.

        The actual call to ``Transformer.generate()`` runs on the
        dedicated tinygrad worker thread (where the sqlite kernel
        cache connection lives).  Tokens are ferried back to the
        caller's thread through a queue; the caller iterates as if
        the generator were native.

        **Cancellation**: when the caller stops iterating (breaks
        out of their for-loop -- stop string matched, EOS, tool
        call complete, max_tokens reached), Python's generator
        close() fires the ``finally`` below, which sets
        ``cancelled``.  The worker's inner loop checks that flag
        after every yielded token and bails out, so the GPU stops
        doing forward passes for tokens nobody will read.  Without
        this, tinygrad kept computing tokens until ``max_context``
        every request, locking the backend's single-thread executor
        for minutes and producing the "stuck after tool_call" loop
        users hit.

        The queue is unbounded because the consumer may be many
        tokens ahead-of-producer when cancellation arrives; we
        don't want the worker to block on put() before it can
        check the flag.
        """
        assert self._model is not None, "Model not loaded"

        token_q: queue.Queue = queue.Queue()
        SENTINEL = object()
        cancelled = threading.Event()
        self._active_cancel = cancelled

        def _run() -> None:
            try:
                for tid in self._iter_model_tokens(request, cancelled):
                    if cancelled.is_set():
                        break
                    token_q.put(tid)
            except BaseException as exc:  # noqa: BLE001
                # Cache may be partially written; next request must
                # prefill from 0 to avoid attending to stale K/V.
                self._cache_tokens = []
                token_q.put(("__error__", exc))
            finally:
                token_q.put(SENTINEL)

        self._executor.submit(_run)

        try:
            while True:
                item = token_q.get()
                if item is SENTINEL:
                    return
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    raise item[1]
                yield item
        finally:
            cancelled.set()
            if self._active_cancel is cancelled:
                self._active_cancel = None

    def cancel_generation(self) -> None:
        """Signal the currently-active generate_tokens call to stop.

        Safe to call from any thread.  A no-op if nothing is
        generating.  The worker loop checks the flag after each
        yielded token from tinygrad, so the GPU stops within one
        more forward pass at most.
        """
        ev = self._active_cancel
        if ev is not None:
            ev.set()

    def _iter_model_tokens(
        self,
        request: CompiledRequest,
        cancelled: threading.Event,
    ) -> Iterator[int]:
        """Drive tinygrad's Transformer forward path while reusing the
        per-block cache_kv for the longest token prefix shared with
        the previous request.

        We bypass ``Transformer.generate`` because its line
        ``start_pos = 0`` discards the start-pos argument and its
        ``forward_jit.reset()`` drops the T=1 decode kernel between
        requests.  Both defeat prefix caching.

        Invariant maintained: ``self._cache_tokens`` equals the
        ordered tokens currently resident in each block's
        ``cache_kv[0:len(self._cache_tokens))``.  The invariant is
        preserved across cancel and across the producer's normal
        exit.  On exception the outer ``_run`` clears it.
        """
        from tinygrad import Tensor, UOp, getenv

        model = self._model
        assert model is not None

        prompt_list = list(request.prompt_tokens)
        assert prompt_list, "empty prompt_tokens"

        enable_cache = os.environ.get("EGG_PREFIX_CACHE", "1") != "0"

        # --- Find longest common prefix with the resident cache. ----
        if enable_cache:
            cached = self._cache_tokens
            cp = 0
            cap = min(len(cached), len(prompt_list))
            while cp < cap and cached[cp] == prompt_list[cp]:
                cp += 1
        else:
            cp = 0

        # If the new prompt is fully contained in the cache we still
        # need to run forward on at least one input token to produce
        # the next-token logits.  Roll back by 1 and re-feed that
        # token at its position -- the cache_kv slot will be rewritten
        # identically, so correctness is preserved.
        if cp == len(prompt_list):
            cp -= 1
        assert cp >= 0

        new_suffix = prompt_list[cp:]

        # --- Prefill the diverging suffix. --------------------------
        # T > 1 -> non-JIT forward path (matches upstream generate).
        t = Tensor([new_suffix], dtype="int32")
        t = model(t, cp)
        # cache_kv[0:len(prompt_list)) now reflects prompt_list exactly.
        self._cache_tokens = list(prompt_list)
        next_id = int(t.item())
        yield next_id

        # --- Decode loop. -------------------------------------------
        # T = 1 with UOp-bound start_pos: hits the JITted kernel, which
        # persists across requests because start_pos is symbolic.  We
        # intentionally do NOT call forward_jit.reset().
        v_start_pos = UOp.variable("start_pos", 1, model.max_context - 1)
        pos = len(prompt_list)
        max_ctx = model.max_context
        use_sym = bool(getenv("SYM", 1))
        while pos < max_ctx:
            if cancelled.is_set():
                break
            t = Tensor([[next_id]], dtype="int32")
            sp = v_start_pos.bind(pos) if use_sym else pos
            t = model(t, sp)
            # We've just fed `next_id` at cache position `pos`.  The
            # assign in _attention writes cache_kv[pos:pos+1), and the
            # resident-tokens list must grow in lockstep so that if a
            # cancel arrives right now, the invariant still holds.
            self._cache_tokens.append(next_id)
            pos += 1
            next_id = int(t.item())
            yield next_id

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
