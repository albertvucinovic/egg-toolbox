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
            debug_prefix = os.environ.get("EGG_DEBUG_PREFIX", "0") != "0"
            try:
                for tid in self._iter_model_tokens(request, cancelled):
                    if cancelled.is_set():
                        break
                    token_q.put(tid)
            except BaseException as exc:  # noqa: BLE001
                # Cache may be partially written; next request must
                # prefill from 0 to avoid attending to stale K/V.
                if debug_prefix:
                    print(
                        f"[egg prefix-cache] EXCEPTION in _iter_model_tokens "
                        f"({type(exc).__name__}: {exc}) -- clearing _cache_tokens "
                        f"(was {len(self._cache_tokens)} tokens)",
                        flush=True,
                    )
                self._cache_tokens = []
                token_q.put(("__error__", exc))
            finally:
                if debug_prefix:
                    print(
                        f"[egg prefix-cache] request done -- _cache_tokens "
                        f"now has {len(self._cache_tokens)} tokens",
                        flush=True,
                    )
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
        debug_prefix = os.environ.get("EGG_DEBUG_PREFIX", "0") != "0"

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

        if debug_prefix:
            # EGG_DEBUG_PREFIX=1: print prefix-cache diagnostics so the
            # user can see whether the cache is hitting across requests.
            # "cp" is how many tokens of the new prompt were found in
            # cache_kv from the prior request.  If cp==0 every request
            # despite identical conversation history, either (a) the
            # client is resending different tokens each turn (template
            # rendering issue) or (b) _cache_tokens got reset.
            # When the match breaks partway we also print tokens around
            # the mismatch point so you can see exactly what diverges.
            cache_head = list(self._cache_tokens[:8])
            cache_tail = list(self._cache_tokens[-8:]) if len(self._cache_tokens) > 8 else []
            prompt_head = list(prompt_list[:8])
            prompt_tail = list(prompt_list[-8:]) if len(prompt_list) > 8 else []
            print(
                f"[egg prefix-cache] enable={enable_cache} "
                f"prompt_len={len(prompt_list)} "
                f"cache_len={len(self._cache_tokens)} "
                f"common_prefix={cp} "
                f"prefill_tokens={len(new_suffix)} "
                f"cache_head={cache_head} cache_tail={cache_tail} "
                f"prompt_head={prompt_head} prompt_tail={prompt_tail}",
                flush=True,
            )
            # If the match broke partway (cp > 0 but cp < min(len cache,
            # len prompt)), show an 8-token window straddling the
            # mismatch point.  This is what you need to diagnose BPE
            # boundary drift or template rendering differences.
            cap_for_window = min(len(self._cache_tokens), len(prompt_list))
            if 0 < cp < cap_for_window:
                lo = max(0, cp - 3)
                hi = min(cap_for_window, cp + 5)
                window_cache = list(self._cache_tokens[lo:hi])
                window_prompt = list(prompt_list[lo:hi])
                marker = ["<<<" if lo + i == cp else "" for i in range(hi - lo)]
                print(
                    f"[egg prefix-cache] mismatch window @ cp={cp} "
                    f"(indices {lo}..{hi - 1}):\n"
                    f"  cache:  {window_cache}\n"
                    f"  prompt: {window_prompt}\n"
                    f"  marker: {marker}",
                    flush=True,
                )
                # Also try decoding the tokens if a tokenizer is attached;
                # the text form usually makes the template drift obvious.
                if self._tokenizer is not None:
                    try:
                        cache_text = self._tokenizer.decode(window_cache)
                        prompt_text = self._tokenizer.decode(window_prompt)
                        print(
                            f"[egg prefix-cache]   cache decoded:  {cache_text!r}\n"
                            f"[egg prefix-cache]   prompt decoded: {prompt_text!r}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[egg prefix-cache]   (decode failed: {exc})",
                            flush=True,
                        )

        # Rolling penalty window.  Seeded with prompt tail.
        recent = list(prompt_list[-_PENALTY_WINDOW:])

        def _record_seen(tid: int) -> None:
            recent.append(tid)
            if len(recent) > _PENALTY_WINDOW:
                del recent[0]

        # --- Chunked prefill of the diverging suffix. ---------------
        # Rather than feed ``new_suffix`` as one T=len(new_suffix) call
        # (which compiles a fresh kernel for every unique prompt length
        # -- a huge cost under JITBEAM>=1), we feed fixed-size chunks
        # of length ``EGG_PREFILL_CHUNK`` through a SINGLE shape signature.
        # After the first request has paid the BEAM compile cost for
        # T=CHUNK kernels, every future prefill of any length reuses
        # those cached kernels from tinygrad's on-disk cache.db.  The
        # residual <CHUNK_T tokens at the tail feed one-at-a-time
        # through the already-JITted T=1 decode path.
        #
        # EGG_PREFILL_CHUNK=0 disables chunking and restores the old
        # single-shot prefill (useful for A/B debugging).
        chunk_size_env = os.environ.get("EGG_PREFILL_CHUNK", "128")
        try:
            chunk_size = int(chunk_size_env)
        except ValueError:
            chunk_size = 128

        v_start_pos = UOp.variable("start_pos", 1, model.max_context - 1)
        use_sym = bool(getenv("SYM", 1))

        # Helper: run one T=1 forward via the JITted decode kernel.
        # UOp-bound start_pos (for sp_int>=1) lets tinygrad reuse the
        # same compiled kernel across every decode position.
        # Falls back to an int for pos=0 because the UOp's declared
        # range is [1, max_context-1).
        def _forward_t1(tok: int, sp_int: int):
            t = Tensor([[tok]], dtype="int32")
            if use_sym and sp_int >= 1:
                sp = v_start_pos.bind(sp_int)
            else:
                sp = sp_int
            return model(t, sp)

        # Helper: run one forward with T>1 (chunk prefill).
        # MUST use an int start_pos because tinygrad's TransformerBlock
        # constructs the causal mask via ``triu(start_pos+1)`` over a
        # shape of ``(1, 1, T, start_pos+T)``, and ``triu`` asserts
        # that the shape dims are plain ints (see tensor.py::_tri).
        # A UOp start_pos here raises "does not support symbolic".
        # We accept the tradeoff: each unique chunk start position
        # compiles its own kernel (up to max_context/chunk_size
        # distinct kernels, ~64 for default settings), all disk-cached
        # after first compile -- still a big win over the pre-chunking
        # behaviour where every unique prompt length T triggered a
        # fresh BEAM compile.
        def _forward_chunk(chunk_tokens: list[int], sp_int: int):
            t = Tensor([chunk_tokens], dtype="int32")
            return model(t, sp_int)

        feed_pos = cp
        end_pos = len(prompt_list)
        logits = None

        if chunk_size > 0:
            while feed_pos + chunk_size <= end_pos:
                chunk = list(prompt_list[feed_pos:feed_pos + chunk_size])
                logits = _forward_chunk(chunk, feed_pos)
                feed_pos += chunk_size

            # Residual (< chunk_size tokens) via T=1 each.  This path
            # hits the T=1 decode kernel, which is also reused across
            # all requests.
            while feed_pos < end_pos:
                logits = _forward_t1(prompt_list[feed_pos], feed_pos)
                feed_pos += 1
        else:
            # Unchunked fallback: single large prefill at int start_pos.
            logits = _forward_chunk(list(prompt_list[cp:]), cp)
            feed_pos = end_pos

        assert logits is not None, "prefill produced no logits -- suffix length was 0?"
        # cache_kv[0:end_pos) now reflects prompt_list exactly.
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
        pos = len(prompt_list)
        max_ctx = model.max_context
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
