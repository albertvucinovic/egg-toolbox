from __future__ import annotations

import json
import os
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from .base import StepBackend, Tokenizer
from ..types import CompiledRequest


def _default_schedule_cache_file() -> str:
    base = os.environ.get(
        "XDG_CACHE_HOME",
        os.path.expanduser("~/.cache"),
    )
    return os.path.join(base, "egg-toolbox", "schedule-cache.pkl")


def _load_schedule_cache_into_tinygrad(path: str) -> int:
    """Read a previously saved ``schedule_cache`` snapshot from disk
    and merge it into tinygrad's module-level dict so subsequent
    forwards hit those cached schedules instead of re-running the
    ~125ms/layer create_schedule() pass.

    Returns the number of entries restored.  Tolerant of missing/
    malformed files -- those return 0 and leave the cache alone.

    Buffer portability across processes is handled by tinygrad itself:
    ``pm_pre_sched_cache`` rewrites UNIQUE -> LUNIQUE before hashing
    the key, and ``pm_post_sched_cache`` re-binds LUNIQUE -> current
    process's input_buffers at each retrieval, so serialized entries
    don't hold stale GPU memory handles.
    """
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "rb") as f:
            restored: dict = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
        print(
            f"[egg schedule-cache] load failed ({type(exc).__name__}: "
            f"{exc}); discarding.  Remove the file and retry if "
            f"you want to rebuild from scratch: {path}",
            flush=True,
        )
        return 0
    if not isinstance(restored, dict):
        return 0
    try:
        from tinygrad.engine.schedule import schedule_cache
        before = len(schedule_cache)
        schedule_cache.update(restored)
        after = len(schedule_cache)
        return after - before
    except ImportError:
        return 0


def _save_schedule_cache_from_tinygrad(path: str) -> None:
    """Snapshot tinygrad's module-level ``schedule_cache`` to disk so
    the next process restart can skip the expensive scheduling pass.
    Atomic write via tmp + replace so a crashed save doesn't corrupt
    the file readers of a prior good save would pick up.

    Silent on failure -- a missed save just means next restart has
    to rebuild from positions.
    """
    try:
        from tinygrad.engine.schedule import schedule_cache
    except ImportError:
        return
    if not schedule_cache:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        # Snapshot dict contents before pickling so concurrent writes
        # from the backend thread (if any) don't see a half-iterated
        # dict -- Python dict copy is atomic from Python-level view.
        snapshot = dict(schedule_cache)
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except (OSError, pickle.PickleError):
        pass


# Where we persist the set of chunk positions we've actually seen in
# real requests.  The ``persisted`` warmup mode reads this file at
# load time and warms those positions before any user request lands,
# trading a bounded one-time load cost for fast steady-state response.
# Keyed by (chunk_size, max_context) so stale entries don't apply when
# the user changes those.
def _default_positions_file() -> str:
    base = os.environ.get(
        "XDG_CACHE_HOME",
        os.path.expanduser("~/.cache"),
    )
    return os.path.join(base, "egg-toolbox", "warmup-positions.json")


def _load_warmup_positions(
    path: str, chunk_size: int, max_context: int,
) -> list[int]:
    """Return the persisted list of chunk-aligned positions seen in
    previous runs, filtered to match the current (chunk_size,
    max_context).  Entries that don't match are silently dropped so
    upgrades / config changes don't crash the server."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    if data.get("chunk_size") != chunk_size:
        return []
    if data.get("max_context") != max_context:
        return []
    positions = data.get("positions", [])
    if not isinstance(positions, list):
        return []
    return sorted({
        int(p) for p in positions
        if isinstance(p, int) and 0 <= p < max_context - chunk_size
    })


def _save_warmup_positions(
    path: str, chunk_size: int, max_context: int, positions: set[int],
) -> None:
    """Atomically write the observed positions to disk.  Tolerant of
    write failures -- a missed save just means the next restart
    doesn't get the benefit from this session's discoveries."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "chunk_size": chunk_size,
                "max_context": max_context,
                "positions": sorted(positions),
            }, f)
        os.replace(tmp, path)
    except OSError:
        pass


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

        # Chunk-positions observed in real requests, used by the
        # ``persisted`` warmup mode.  tinygrad's schedule_cache is
        # in-memory only (engine/schedule.py:130) so every process
        # restart re-pays ~12s per chunk position on first touch.
        # By recording which positions this workload actually uses
        # and replaying the warmup on next startup, we bound the
        # steady-state cost to one session of discovery.  Saved
        # after each request to disk at
        # ``$XDG_CACHE_HOME/egg-toolbox/warmup-positions.json`` (or
        # EGG_WARMUP_POSITIONS_FILE if set).
        self._used_chunk_positions: set[int] = set()
        self._positions_file = os.environ.get(
            "EGG_WARMUP_POSITIONS_FILE", _default_positions_file(),
        )

        # Experimental: persist tinygrad's module-level
        # ``schedule_cache`` (engine/schedule.py:130) across process
        # restarts by pickling it to disk.  Unlike persisted-positions
        # (which replays a list to rebuild schedule_cache via dummy
        # forwards, paying ~12s per position at each startup), this
        # stores the actual scheduled UOp trees + ExecItems.  On next
        # startup we unpickle them straight into the cache dict so the
        # schedule pass is skipped entirely -- seconds of load instead
        # of minutes.  Tinygrad's own pm_pre_sched_cache /
        # pm_post_sched_cache rewrites make entries portable: buffer
        # references are replaced with LUNIQUE placeholders before
        # hashing and re-bound to the current process's buffers on
        # retrieval, so stale GPU handles from the previous process
        # never get executed.  Opt-in because this depends on
        # tinygrad-internals stability (a version bump could break
        # the pickle format).
        self._schedule_cache_enabled = (
            os.environ.get("EGG_SCHEDULE_CACHE", "0") != "0"
        )
        self._schedule_cache_file = os.environ.get(
            "EGG_SCHEDULE_CACHE_FILE", _default_schedule_cache_file(),
        )

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

        # Restore tinygrad's schedule_cache from disk before warmup so
        # positions we've seen in prior runs don't re-run the schedule
        # pass.  Under EGG_SCHEDULE_CACHE=1 this is a seconds-scale
        # load vs minutes for position-replay warmup.
        if self._schedule_cache_enabled:
            n = _load_schedule_cache_into_tinygrad(self._schedule_cache_file)
            if n > 0:
                print(
                    f"[egg schedule-cache] loaded {n} entries from "
                    f"{self._schedule_cache_file}",
                    flush=True,
                )

        # Kernel warmup.  Each unique (shape, start_pos) forward triggers
        # a fresh schedule pass the first time it's seen in a Python
        # process (tinygrad's schedule_cache is in-memory only and not
        # persisted to cache.db).  At chunk_size=128 on Qwen3-8B this
        # costs ~4s of scheduling plus ~8s of GPU work per chunk
        # position, paid at the time the user's request hits.  We move
        # that cost to load time by pre-running dummy forwards here.
        #
        # EGG_WARMUP modes:
        #   off   -- skip warmup entirely (fastest load, slow first
        #            request)
        #   first -- warm one chunk @ 0 + one T=1 decode (default;
        #            ~12s added to load, first chunk position is
        #            pre-cached but other positions still pay)
        #   full  -- warm every chunk-aligned position 0, CHUNK,
        #            2*CHUNK, ..., up to max_context.  Longest load
        #            (~minutes) but every real request is fast.
        self._warmup(
            mode=os.environ.get("EGG_WARMUP", "first"),
            chunk_size=int(os.environ.get("EGG_PREFILL_CHUNK", "128")),
        )

    def _warmup(self, mode: str, chunk_size: int) -> None:
        """Populate tinygrad's in-memory schedule_cache so the first
        real request doesn't pay per-chunk-position scheduling cost."""
        import time as _time
        from tinygrad import Tensor, UOp

        if mode == "off":
            return

        if self._model is None:  # defensive -- load_model() sets it
            return

        max_ctx = self._model.max_context
        if mode == "first":
            positions = [0]
        elif mode == "full":
            # Stop before max_context so we never bind a cache position
            # out of range.  Chunk kernels are keyed on int start_pos,
            # so each position is a distinct schedule.
            positions = list(range(0, max_ctx - chunk_size, chunk_size))
        elif mode == "persisted":
            # Replay chunk positions this user's workload actually hit
            # in prior runs.  Falls back to ``first`` if the persisted
            # file is missing or doesn't match our current (chunk_size,
            # max_context) -- fresh installs transparently get the
            # quick warmup until the workload teaches us what to warm.
            persisted = _load_warmup_positions(
                self._positions_file, chunk_size, max_ctx,
            )
            if persisted:
                positions = persisted
                print(
                    f"[egg warmup] replaying {len(persisted)} "
                    f"positions from {self._positions_file}",
                    flush=True,
                )
            else:
                print(
                    f"[egg warmup] no persisted positions at "
                    f"{self._positions_file} (or mismatched chunk/ctx); "
                    f"falling back to 'first' mode",
                    flush=True,
                )
                positions = [0]
        else:
            print(
                f"[egg warmup] unknown EGG_WARMUP={mode!r}, accepted: "
                f"off, first, full, persisted.  Skipping.",
                flush=True,
            )
            return

        print(
            f"[egg warmup] mode={mode} chunk_size={chunk_size} "
            f"positions={len(positions)}; "
            f"pre-running dummy forwards to populate schedule_cache.  "
            f"Cold cache.db = many minutes; warm cache.db = seconds "
            f"per position.",
            flush=True,
        )

        t_total = _time.monotonic()
        dummy_chunk = [0] * chunk_size
        dummy_decode = 0

        # Chunk kernels (T = chunk_size, int start_pos).
        for i, pos in enumerate(positions):
            t = Tensor([dummy_chunk], dtype="int32")
            t0 = _time.monotonic()
            try:
                _ = self._model(t, pos)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[egg warmup] chunk @ {pos} FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            elapsed = _time.monotonic() - t0
            print(
                f"[egg warmup]   [{i + 1:3d}/{len(positions)}] "
                f"chunk @ {pos:5d} T={chunk_size}  {elapsed:6.2f}s",
                flush=True,
            )

        # T=1 decode kernel (UOp-bound start_pos).  Only needs one
        # warmup because the same kernel serves every decode step.
        v_sp = UOp.variable("start_pos", 1, max_ctx - 1)
        t = Tensor([[dummy_decode]], dtype="int32")
        t0 = _time.monotonic()
        try:
            _ = self._model(t, v_sp.bind(1))
            elapsed = _time.monotonic() - t0
            print(
                f"[egg warmup]   T=1 decode @ 1  {elapsed:6.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[egg warmup] T=1 decode FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        # Reset _cache_tokens -- dummy forwards above wrote garbage
        # K/V into cache_kv[0:chunk_size * N], but that doesn't matter
        # because the first real request matches against an empty
        # _cache_tokens (cp=0) and rewrites those positions anyway.
        # Being explicit about it keeps the prefix-cache invariant
        # (``_cache_tokens reflects the ordered tokens currently
        # resident in cache_kv[0:len(_cache_tokens))``) trivially true:
        # we assert nothing is resident post-warmup.
        self._cache_tokens = []
        print(
            f"[egg warmup] done in {_time.monotonic() - t_total:.1f}s",
            flush=True,
        )

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

        # Annotated per-forward log.  ``EGG_LOG_FORWARD=1`` prints one
        # line per model() call with purpose + shape + elapsed time,
        # so you can correlate tinygrad's ``CACHE MISS <hash>`` /
        # ``CACHE HIT <hash>`` lines (which don't say what they're
        # caching) with the egg-toolbox operation that triggered them.
        # Also prints a one-line summary at the end of each request
        # with aggregate counts and timings.
        log_forward = os.environ.get("EGG_LOG_FORWARD", "0") != "0"
        # Per-request counters, populated if log_forward is on.
        fwd_stats = {
            "chunks": 0, "residuals": 0, "decodes": 0,
            "chunk_ms": 0.0, "residual_ms": 0.0, "decode_ms": 0.0,
            "slowest": (0.0, ""),  # (ms, label)
        }

        def _call(label: str, t, sp):
            """Call model() with optional annotated logging.  ``label``
            is a short tag like ``chunk @ 128`` for the log line.
            Captures wall-clock so you see which calls pay a compile
            (the first of a given shape will be much slower than the
            rest)."""
            import time as _time
            if not log_forward:
                return model(t, sp)
            sp_disp = (
                sp if isinstance(sp, int)
                else getattr(sp, "_bound_value", "UOp(?)")
            )
            T = t.shape[1]
            t0 = _time.monotonic()
            out = model(t, sp)
            elapsed_ms = (_time.monotonic() - t0) * 1000.0
            print(
                f"[egg forward] {label} T={T} start_pos={sp_disp} "
                f"{elapsed_ms:7.1f}ms",
                flush=True,
            )
            # Track slowest for the summary line.
            if elapsed_ms > fwd_stats["slowest"][0]:
                fwd_stats["slowest"] = (elapsed_ms, f"{label} T={T} sp={sp_disp}")
            if label.startswith("chunk"):
                fwd_stats["chunks"] += 1
                fwd_stats["chunk_ms"] += elapsed_ms
            elif label.startswith("residual"):
                fwd_stats["residuals"] += 1
                fwd_stats["residual_ms"] += elapsed_ms
            elif label.startswith("decode"):
                fwd_stats["decodes"] += 1
                fwd_stats["decode_ms"] += elapsed_ms
            return out

        # Helper: run one T=1 forward via the JITted decode kernel.
        # UOp-bound start_pos (for sp_int>=1) lets tinygrad reuse the
        # same compiled kernel across every decode position.
        # Falls back to an int for pos=0 because the UOp's declared
        # range is [1, max_context-1).
        def _forward_t1(tok: int, sp_int: int, label: str):
            t = Tensor([[tok]], dtype="int32")
            if use_sym and sp_int >= 1:
                sp = v_start_pos.bind(sp_int)
            else:
                sp = sp_int
            return _call(label, t, sp)

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
        def _forward_chunk(chunk_tokens: list[int], sp_int: int, label: str):
            # Record each position we actually use so the ``persisted``
            # warmup mode can replay exactly these on next startup.
            self._used_chunk_positions.add(sp_int)
            t = Tensor([chunk_tokens], dtype="int32")
            return _call(label, t, sp_int)

        feed_pos = cp
        end_pos = len(prompt_list)
        logits = None

        # Wrap the prefill + decode in try/finally so the one-line
        # per-request summary fires on any exit -- normal completion,
        # caller break (GeneratorExit at yield), or exception.
        # The summary reports aggregate counts and timings so the user
        # doesn't have to scroll through the per-call log to see what
        # the request did overall.
        try:
            if chunk_size > 0:
                while feed_pos + chunk_size <= end_pos:
                    chunk = list(prompt_list[feed_pos:feed_pos + chunk_size])
                    logits = _forward_chunk(chunk, feed_pos, f"chunk @ {feed_pos}")
                    feed_pos += chunk_size

                # Residual (< chunk_size tokens) via T=1 each.  This
                # path hits the T=1 decode kernel, which is also
                # reused across all requests.
                while feed_pos < end_pos:
                    logits = _forward_t1(
                        prompt_list[feed_pos], feed_pos,
                        f"residual @ {feed_pos}",
                    )
                    feed_pos += 1
            else:
                # Unchunked fallback: single large prefill at int
                # start_pos.
                logits = _forward_chunk(
                    list(prompt_list[cp:]), cp, f"full prefill @ {cp}",
                )
                feed_pos = end_pos

            assert logits is not None, (
                "prefill produced no logits -- suffix length was 0?"
            )
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

            # --- Decode loop. ---------------------------------------
            # T = 1 with UOp-bound start_pos: hits the JITted kernel,
            # which persists across requests because start_pos is
            # symbolic.  We intentionally do NOT call
            # forward_jit.reset().
            pos = len(prompt_list)
            max_ctx = model.max_context
            while pos < max_ctx:
                if cancelled.is_set():
                    break
                t = Tensor([[next_id]], dtype="int32")
                sp = v_start_pos.bind(pos) if use_sym else pos
                logits = (
                    _call(f"decode @ {pos}", t, sp)
                    if log_forward else model(t, sp)
                )
                # We've just fed `next_id` at cache position `pos`.
                # The assign in _attention writes cache_kv[pos:pos+1),
                # and the resident-tokens list must grow in lockstep
                # so that if a cancel arrives right now, the invariant
                # still holds.
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
        finally:
            if log_forward:
                s = fwd_stats
                total_fwd = s["chunks"] + s["residuals"] + s["decodes"]
                total_ms = s["chunk_ms"] + s["residual_ms"] + s["decode_ms"]
                avg_decode = (
                    s["decode_ms"] / s["decodes"]
                    if s["decodes"] else 0.0
                )
                slow_ms, slow_label = s["slowest"]
                print(
                    f"[egg forward] request summary: "
                    f"chunks={s['chunks']}({s['chunk_ms']:.0f}ms) "
                    f"residuals={s['residuals']}({s['residual_ms']:.0f}ms) "
                    f"decode={s['decodes']}tok({s['decode_ms']:.0f}ms, "
                    f"avg {avg_decode:.1f}ms/tok) "
                    f"total={total_fwd}calls/{total_ms:.0f}ms; "
                    f"slowest={slow_ms:.0f}ms ({slow_label})",
                    flush=True,
                )
            # Persist the chunk positions we've touched this session so
            # the ``persisted`` warmup mode can replay them on next
            # startup.  Cheap (small JSON, atomic replace); failures
            # are silent.  We save after EVERY request so the set is
            # up-to-date even if the server exits ungracefully (crash,
            # SIGKILL, power loss).
            if self._used_chunk_positions:
                _save_warmup_positions(
                    self._positions_file,
                    chunk_size,
                    model.max_context,
                    self._used_chunk_positions,
                )
            # And, under the experimental flag, snapshot the actual
            # tinygrad schedule_cache to disk so next startup can skip
            # the schedule pass entirely.  The file can get sizeable
            # (~MB for a full transformer's worth of schedules) so
            # saving after each request is a tradeoff; could be
            # debounced in the future but disk write of a few MB is
            # trivial next to the 12s-per-position we're avoiding.
            if self._schedule_cache_enabled:
                _save_schedule_cache_from_tinygrad(self._schedule_cache_file)

    def model_name(self) -> str:
        return self._model_name_str
