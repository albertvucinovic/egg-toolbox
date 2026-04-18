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
        from tinygrad.nn.llm import Transformer, SimpleTokenizer

        model_path_obj = Path(model_path)
        self._model_name_str = model_path_obj.stem

        # Build model from GGUF
        self._model = Transformer.build(model_path, **kwargs)

        # Extract chat template from GGUF metadata
        from ..template import ChatTemplate
        ct = ChatTemplate.from_gguf(model_path)
        self._chat_template = ct.source

        # Create tokenizer wrapper
        tokenizer = SimpleTokenizer(model_path)
        eos_id = ct.eos_token  # We need the ID, not the token string
        # Get IDs from the GGUF metadata directly
        import struct
        bos_id = None
        eos_id_int = 0
        with open(model_path, "rb") as f:
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            from ..template import _read_gguf_string, _read_gguf_value
            for _ in range(kv_count):
                key = _read_gguf_string(f, version)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = _read_gguf_value(f, value_type, version)
                if key == "tokenizer.ggml.bos_token_id":
                    bos_id = value
                elif key == "tokenizer.ggml.eos_token_id":
                    eos_id_int = value

        self._tokenizer = TinygradTokenizer(tokenizer, eos_id=eos_id_int, bos_id=bos_id)

    def tokenizer(self) -> Tokenizer:
        assert self._tokenizer is not None, "Model not loaded"
        return self._tokenizer

    def chat_template_source(self) -> str:
        return self._chat_template

    def generate_tokens(self, request: CompiledRequest) -> Iterator[int]:
        """Wrap Transformer.generate() which already yields token IDs."""
        assert self._model is not None, "Model not loaded"
        for token_id in self._model.generate(
            list(request.prompt_tokens),
            temperature=request.sampling.temperature,
        ):
            yield token_id

    def model_name(self) -> str:
        return self._model_name_str
