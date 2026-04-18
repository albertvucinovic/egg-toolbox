"""Architecture ABC and dispatch registry for the tinygrad backend.

Conventions a subclass must honour so the existing prefix-cache path
in ``backends/tinygrad.py`` works unchanged:

- ``forward(tokens, start_pos) -> Tensor`` returns the logits of the
  LAST input position only, shape ``(B, vocab_size)``.  Sampling (not
  argmax) happens in Python via :mod:`egg_toolbox.sampling` so the
  SamplingParams temperature / top_p / top_k / penalties take effect.
- ``__call__(tokens, start_pos)`` chooses between a JITted T=1 kernel
  and an eager path for T>1.  Decode calls the backend makes are T=1
  with a ``UOp``-bound ``start_pos``; prefills are T>1 with an int
  ``start_pos``.
- ``max_context: int`` is the usable KV-cache length.  It must match
  whatever the blocks' ``cache_kv`` tensors are sized for, otherwise
  the symbolic decode JIT blows past the slot.
- Per-block KV cache: each entry in ``self.blk`` owns an in-place KV
  cache keyed by position.  After a forward at ``start_pos=P`` with
  ``T`` tokens, positions ``[P, P+T)`` reflect those tokens; earlier
  positions are preserved.  The backend relies on this to reuse the
  longest common token prefix across sequential requests.
- ``token_embd``, ``output``, ``output_norm``: by convention match the
  tinygrad Transformer names so existing code that walks the model
  (state-dict dumps, parameter counts) keeps working.  A custom arch
  can deviate as long as its own ``forward`` produces logits of the
  right shape.

Dispatch: each module registers itself with :func:`register` for one
or more GGUF ``general.architecture`` strings.  Unknown architectures
fall back to the class registered with ``fallback=True``.

Factory protocol: each registered class provides a
``from_gguf_kv(kv, state_dict, **kwargs) -> (instance, kv)`` classmethod.
The dispatcher parses the GGUF once and hands both dicts to the
factory, so a subclass never re-reads the file.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from tinygrad import Tensor, UOp


class Architecture(abc.ABC):
    """Base class every per-family architecture must extend.

    Subclasses typically inherit from ``tinygrad.apps.llm.Transformer``
    as well when the standard block shape applies.  A fully custom
    architecture (DeepSeek MLA, Gemma soft-capping, ...) overrides
    whatever it needs while keeping this contract.
    """

    # Set by subclass / loader.
    max_context: int
    blk: list[Any]

    @abc.abstractmethod
    def forward(self, tokens: "Tensor", start_pos: "int | UOp") -> "Tensor":
        """Return logits for the last position, shape ``(B, vocab_size)``."""

    def __call__(self, tokens: "Tensor", start_pos: "int | UOp" = 0) -> "Tensor":
        return self.forward(tokens, start_pos)


ArchFactory = Callable[..., tuple["Architecture", dict]]

# Registry: GGUF ``general.architecture`` string -> factory callable.
# Factory signature is ``(kv, state_dict, **kwargs) -> (arch, kv)``.
_REGISTRY: dict[str, ArchFactory] = {}

# Architecture name used when the GGUF arch is not registered.
_FALLBACK_ARCH: str | None = None


_T = TypeVar("_T", bound=type)


def register(*arch_names: str, fallback: bool = False) -> Callable[[_T], _T]:
    """Class decorator: wire ``cls.from_gguf_kv`` into the dispatch map.

    Each name is a string that appears under ``general.architecture``
    in a GGUF file (``llama``, ``qwen2``, ``qwen3``, ``gemma2``, ...).
    Pass ``fallback=True`` on exactly one class to make it the default
    for architectures no module has claimed.
    """

    def _wrap(cls: _T) -> _T:
        factory = getattr(cls, "from_gguf_kv", None)
        if not callable(factory):
            raise TypeError(
                f"{cls.__name__} must define a classmethod "
                "from_gguf_kv(kv, state_dict, **kw) -> (instance, kv) "
                "to be registered."
            )
        for name in arch_names:
            _REGISTRY[name] = factory
        if fallback:
            global _FALLBACK_ARCH
            _FALLBACK_ARCH = arch_names[0] if arch_names else cls.__name__
        return cls

    return _wrap


def registered_architectures() -> tuple[str, ...]:
    """Return the sorted tuple of GGUF arch names the registry handles."""
    return tuple(sorted(_REGISTRY))


def load_from_gguf(model_path: str, **kwargs: Any) -> tuple["Architecture", dict]:
    """Load a GGUF into the right Architecture subclass.

    Reads ``general.architecture`` from the file header.  Dispatches to
    the registered factory for that arch, or the fallback class if the
    arch is unknown.  ``**kwargs`` are forwarded to the factory
    (``max_context``, ``keep_packed``, ``realize``, ...).
    """
    from pathlib import Path

    from tinygrad import Tensor, nn

    kv, state_dict = nn.state.gguf_load(Tensor(Path(model_path)).to(None))

    arch_name = kv.get("general.architecture")
    if arch_name is None:
        raise ValueError(
            f"GGUF at {model_path} is missing the general.architecture key."
        )

    factory = _REGISTRY.get(arch_name)
    if factory is None:
        if _FALLBACK_ARCH is None:
            raise ValueError(
                f"Unknown GGUF architecture {arch_name!r} and no fallback "
                f"is registered.  Known: {registered_architectures()}"
            )
        factory = _REGISTRY[_FALLBACK_ARCH]

    return factory(kv, state_dict, **kwargs)
