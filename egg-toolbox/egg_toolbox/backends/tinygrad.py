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
        from tinygrad.apps.llm import SimpleTokenizer

        from ..models import load_from_gguf

        model_path_obj = Path(model_path)
        self._model_name_str = model_path_obj.stem

        # Dispatch to the right Architecture class based on GGUF
        # ``general.architecture``.  The loaded instance already returns
        # logits (not argmax) from its forward and has any arch-specific
        # weight fixups applied (e.g. Qwen2 Q/K/V bias tensors, llama
        # Q/K half-split RoPE rearrangement).
        self._model, kv = load_from_gguf(str(model_path_obj), **kwargs)

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
        import numpy as np
        from tinygrad import Tensor, UOp, getenv

        from ..sampling import sample_next_token, _rng_for

        model = self._model
        assert model is not None

        prompt_list = list(request.prompt_tokens)
        assert prompt_list, "empty prompt_tokens"

        # Per-request rng so concurrent requests (serialised on the
        # backend thread, but still semantically per-request) do not
        # share entropy.  Built once, reused for every decode step.
        rng = _rng_for(request.sampling.seed)

        # Rolling window of the last N tokens seen by the model
        # (prompt plus generated).  Used to apply repetition,
        # frequency, and presence penalties before sampling.  Matches
        # llama.cpps penalty_last_n default of 64.
        _PENALTY_WINDOW = 64

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

        # Rolling penalty window.  Seeded with prompt tail.
        recent = list(prompt_list[-_PENALTY_WINDOW:])

        def _record_seen(tid: int) -> None:
            recent.append(tid)
            if len(recent) > _PENALTY_WINDOW:
                del recent[0]

        # --- Prefill the diverging suffix. --------------------------
        # T > 1 -> non-JIT forward path (matches upstream generate).
        # The forward has been patched (see _do_load_model) to return
        # logits shape (1, vocab_size) instead of argmax; we sample in
        # Python via egg_toolbox.sampling.
        t = Tensor([new_suffix], dtype="int32")
        logits = model(t, cp)
        # cache_kv[0:len(prompt_list)) now reflects prompt_list exactly.
        self._cache_tokens = list(prompt_list)
        next_id = sample_next_token(
            np.asarray(logits.tolist()[0], dtype=np.float32),
            request.sampling,
            rng,
            recent_tokens=recent,
        )
        _record_seen(next_id)
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
            logits = model(t, sp)
            # We've just fed `next_id` at cache position `pos`.  The
            # assign in _attention writes cache_kv[pos:pos+1), and the
            # resident-tokens list must grow in lockstep so that if a
            # cancel arrives right now, the invariant still holds.
            self._cache_tokens.append(next_id)
            pos += 1
            next_id = sample_next_token(
                np.asarray(logits.tolist()[0], dtype=np.float32),
                request.sampling,
                rng,
                recent_tokens=recent,
            )
            _record_seen(next_id)
            yield next_id

    def model_name(self) -> str:
        return self._model_name_str
