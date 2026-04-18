"""Per-family model architectures for the tinygrad backend.

Each supported architecture lives in its own module (``llama.py``,
``gemma.py``, ``deepseek.py``, ...) and registers itself with the
dispatch registry via the ``@register(...)`` decorator.  The backend
calls :func:`load_from_gguf` which reads ``general.architecture`` from
the GGUF metadata, picks the right class, and returns a loaded
instance plus the kv-metadata dict.

Convention: a new architecture is one file (~300-800 lines) that
implements :class:`Architecture` and registers itself.  It never
touches the orchestrator or the backend.
"""
from .base import Architecture, load_from_gguf, register, registered_architectures

# Importing these modules triggers their @register(...) calls.  Keep
# the imports at module scope (not lazy) so dispatch works the moment
# ``egg_toolbox.models`` is imported.
from . import llama as _llama  # noqa: F401

__all__ = [
    "Architecture",
    "load_from_gguf",
    "register",
    "registered_architectures",
]
