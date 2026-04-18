from __future__ import annotations

import dataclasses
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

    def __init__(self, backend: Backend):
        self._backend = backend
        self._tokenizer = backend.tokenizer()

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
        """Execute generation on a step backend (tinygrad, llama-cpp-python)."""
        assert isinstance(self._backend, StepBackend)

        token_count = 0
        max_tokens = request.sampling.max_tokens
        stop_matcher = StopStringMatcher(request.stop_strings)

        for token_id in self._backend.generate_tokens(request):
            # Check stop token
            if token_id in request.stop_token_ids:
                break

            token_text = self._tokenizer.decode_single(token_id)

            # Check stop strings with partial matching
            safe_text, matched_stop = stop_matcher.feed(token_text)

            if matched_stop:
                # Feed any safe text before the stop string
                if safe_text:
                    events = parser.feed_token(token_id, safe_text)
                    for event in events:
                        yield event
                break

            if safe_text:
                events = parser.feed_token(token_id, safe_text)
                for event in events:
                    yield event

            token_count += 1
            if max_tokens and token_count >= max_tokens:
                break

        # Flush any remaining text in stop matcher
        remaining = stop_matcher.flush()
        if remaining:
            events = parser.feed_token(-1, remaining)
            for event in events:
                yield event

        # Finalize
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
