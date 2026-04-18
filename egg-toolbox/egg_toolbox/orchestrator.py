from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from .types import (
    ChatMessage, CompiledRequest, SamplingParams, SemanticEvent,
    EventKind, StopReason, Tool, FormatAnalysis,
)
from .template import ChatTemplate
from .detector import detect_format
from .parser import StreamingParser
from .formats import get_handler_for_format
from .backends.base import Backend, StepBackend, ConstraintBackend


class StopStringMatcher:
    """Efficient partial-match detector for multiple stop strings.

    Used by the orchestrator's step-backend loop to detect stop
    conditions without false negatives from partial token matches.
    """

    def __init__(self, stop_strings: tuple[str, ...]):
        self._stops = stop_strings
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str | None]:
        """Feed new text. Returns (safe_text, matched_stop_or_none).

        safe_text: text that can be emitted (definitely not part of a stop string)
        matched_stop: the stop string that was matched, or None
        """
        self._buffer += text

        # Check for complete matches
        for stop in self._stops:
            idx = self._buffer.find(stop)
            if idx >= 0:
                safe = self._buffer[:idx]
                self._buffer = ""
                return safe, stop

        # Check for partial matches (suffix of buffer = prefix of any stop string)
        max_partial = 0
        for stop in self._stops:
            for i in range(1, min(len(self._buffer), len(stop)) + 1):
                if self._buffer[-i:] == stop[:i]:
                    max_partial = max(max_partial, i)

        if max_partial > 0:
            safe = self._buffer[:-max_partial]
            self._buffer = self._buffer[-max_partial:]
            return safe, None
        else:
            safe = self._buffer
            self._buffer = ""
            return safe, None

    def flush(self) -> str:
        """Flush remaining buffer (at end of generation)."""
        remaining = self._buffer
        self._buffer = ""
        return remaining


class Orchestrator:
    """Central coordinator for the tool-calling middleware.

    One Orchestrator instance per loaded model. Constructed at model load time.
    """

    def __init__(self, backend: Backend, *, enable_thinking: bool | None = None):
        self._backend = backend
        self._tokenizer = backend.tokenizer()
        # If the chat template takes an ``enable_thinking`` variable
        # (Qwen3-style), this flag controls whether thinking is on
        # for every request.  None = template default (usually True
        # for Qwen3); False explicitly disables; True explicitly
        # enables.  Exposed through __main__ via --disable-thinking.
        self._enable_thinking = enable_thinking

        # Dedicated single-thread executor for all backend generation
        # calls.  Must be max_workers=1 so every request's producer
        # runs on the *same* OS thread -- tinygrad caches its
        # kernel-compile SQLite connection globally but sqlite3
        # rejects cross-thread use ("SQLite objects created in a
        # thread can only be used in that same thread").  Also:
        # serialising requests is the right thing anyway because
        # the underlying GPU is a single resource.
        self._backend_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="egg-backend"
        )

        # Load and analyze the chat template
        template_source = backend.chat_template_source()
        bos = ""
        if self._tokenizer.bos_token_id is not None:
            bos = self._tokenizer.decode_single(self._tokenizer.bos_token_id)
        eos = self._tokenizer.decode_single(self._tokenizer.eos_token_id)

        self._template = ChatTemplate(
            template_str=template_source,
            bos_token=bos,
            eos_token=eos,
        )

        # Detect tool calling format
        self._format_analysis: FormatAnalysis = detect_format(self._template)
        self._handler = get_handler_for_format(self._format_analysis)

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        sampling: SamplingParams = SamplingParams(),
        stream: bool = True,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute a chat completion request and yield semantic events."""

        # 1. Render the prompt using the model's chat template
        prompt_str = self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        prompt_tokens = self._tokenizer.encode(prompt_str)

        # 2. Compute stop conditions from format handler
        stop_strings = self._handler.stop_strings() + sampling.stop
        stop_token_ids = self._handler.stop_token_ids(self._tokenizer) + (self._tokenizer.eos_token_id,)
        if sampling.stop_token_ids:
            stop_token_ids = stop_token_ids + sampling.stop_token_ids

        # 3. Optionally generate grammar constraints
        grammar = None
        json_schema = None
        if tools:
            grammar = self._handler.generate_grammar(tools)
            json_schema = self._handler.generate_json_schema(tools)

        # 4. Build compiled request
        request = CompiledRequest(
            prompt_tokens=prompt_tokens,
            sampling=sampling,
            stop_strings=stop_strings,
            stop_token_ids=stop_token_ids,
            grammar=grammar,
            json_schema=json_schema,
            format_handler_name=type(self._handler).__name__,
        )

        # 5. Create parser
        parser = StreamingParser(
            handler=self._handler,
            tools=tools,
        )

        # 6. Generate and parse based on backend type
        if isinstance(self._backend, StepBackend):
            async for event in self._run_step_backend(request, parser):
                yield event
        elif isinstance(self._backend, ConstraintBackend):
            async for event in self._run_constraint_backend(request, parser):
                yield event

    async def _run_step_backend(
        self,
        request: CompiledRequest,
        parser: StreamingParser,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute generation on a step backend (tinygrad, llama-cpp-python).

        The backend's ``generate_tokens`` is a synchronous iterator whose
        ``next()`` can block for tens of milliseconds to several seconds
        per token (kernel compile, GPU decode).  Running it directly in
        an ``async def`` would block the asyncio event loop and prevent
        uvicorn from flushing SSE chunks between tokens -- the client
        would see bursts rather than a live stream.

        Instead we spawn a worker thread that pulls tokens from the
        backend and pushes them onto an asyncio queue via
        ``call_soon_threadsafe``.  The async iterator awaits the queue,
        so the event loop stays responsive between tokens and each
        yielded event flushes to the client immediately.
        """
        assert isinstance(self._backend, StepBackend)

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        # Hold a reference to the backend generator so we can call
        # close() on early exit -- that triggers its finally block,
        # which sets its cancellation flag and stops the GPU from
        # producing tokens that nobody will read.
        backend_gen = self._backend.generate_tokens(request)

        def producer() -> None:
            def _push(item: object) -> None:
                # Schedule on the event loop, swallowing "loop closed"
                # if the async side has already torn down (client
                # disconnect, early break).  Losing tokens after that
                # point is fine; the consumer has stopped reading.
                try:
                    loop.call_soon_threadsafe(token_queue.put_nowait, item)
                except RuntimeError:
                    pass
            try:
                for tid in backend_gen:
                    _push(tid)
            except BaseException as exc:  # noqa: BLE001
                _push(("__error__", exc))
            finally:
                _push(_SENTINEL)

        # Submit onto the orchestrator's dedicated single-thread
        # executor so every request lands on the *same* OS thread
        # (see note in __init__ re: tinygrad's thread-bound SQLite
        # kernel cache).
        self._backend_executor.submit(producer)

        token_count = 0
        max_tokens = request.sampling.max_tokens
        stop_matcher = StopStringMatcher(request.stop_strings)

        # Optional raw-generation logger.  Set EGG_LOG_GEN=<path>
        # (or EGG_LOG_GEN=1 for default /tmp/egg-gen.log) to dump
        # every decoded token verbatim, plus the rendered prompt at
        # the head of the file.  Only enable for debugging; writes
        # happen inline in the decode loop.
        gen_log_path: str | None = None
        env_val = os.environ.get("EGG_LOG_GEN", "").strip()
        if env_val:
            gen_log_path = "/tmp/egg-gen.log" if env_val == "1" else env_val
            try:
                prompt_text = self._tokenizer.decode(list(request.prompt_tokens))
            except Exception:
                prompt_text = f"<{len(request.prompt_tokens)} tokens, decode failed>"
            with open(gen_log_path, "a", encoding="utf-8") as _f:
                _f.write("\n========== REQUEST @ {} ==========\n".format(
                    time.strftime("%Y-%m-%d %H:%M:%S")))
                _f.write(f"prompt ({len(request.prompt_tokens)} tokens):\n")
                _f.write(prompt_text)
                _f.write("\n--- generated tokens (one per line) ---\n")

        try:
            while True:
                item = await token_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    raise item[1]
                token_id = item

                if token_id in request.stop_token_ids:
                    break

                token_text = self._tokenizer.decode_single(token_id)
                if gen_log_path:
                    try:
                        with open(gen_log_path, "a", encoding="utf-8") as _f:
                            _f.write(f"[{token_id}] {token_text!r}\n")
                    except OSError:
                        pass
                safe_text, matched_stop = stop_matcher.feed(token_text)

                if matched_stop:
                    if safe_text:
                        for event in parser.feed_token(token_id, safe_text):
                            yield event
                    break

                if safe_text:
                    for event in parser.feed_token(token_id, safe_text):
                        yield event

                token_count += 1
                if max_tokens and token_count >= max_tokens:
                    break
        finally:
            # Close the backend generator immediately.  Python's
            # GC would call close() eventually, but we want
            # deterministic cancellation so the GPU stops spending
            # cycles on tokens we've already decided to drop
            # (stop string matched, EOS reached, max_tokens,
            # tool_call completed, or client disconnected).  The
            # backend's generate_tokens() has a finally that sets
            # a threading.Event the worker checks after each
            # yielded token; it bails out within one more forward
            # pass at most.
            try:
                backend_gen.close()
            except Exception:
                pass

        # Flush any remaining text in stop matcher
        remaining = stop_matcher.flush()
        if remaining:
            for event in parser.feed_token(-1, remaining):
                yield event

        num_prompt = len(request.prompt_tokens)
        for event in parser.finish():
            if event.kind == EventKind.DONE:
                event = dataclasses.replace(
                    event,
                    prompt_tokens=num_prompt,
                    completion_tokens=token_count,
                )
            yield event

    async def _run_constraint_backend(
        self,
        request: CompiledRequest,
        parser: StreamingParser,
    ) -> AsyncIterator[SemanticEvent]:
        """Execute generation on a constraint backend (vLLM, SGLang)."""
        assert isinstance(self._backend, ConstraintBackend)

        accumulated_text: list[str] = []
        async for text_chunk in self._backend.generate_stream(request):
            accumulated_text.append(text_chunk)
            events = parser.feed_text(text_chunk)
            for event in events:
                yield event

        num_prompt = len(request.prompt_tokens)
        completion_tokens = len(self._tokenizer.encode("".join(accumulated_text)))
        for event in parser.finish():
            if event.kind == EventKind.DONE:
                event = dataclasses.replace(
                    event,
                    prompt_tokens=num_prompt,
                    completion_tokens=completion_tokens,
                )
            yield event
