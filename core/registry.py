"""Backend registry: maps backend names in the config to classes.

A backend registers itself once:

    @register("classifier", "modernbert_nli")
    class ModernBertClassifier(Classifier): ...

and the pipeline builds it from the config:

    classifier = create("classifier", "modernbert_nli", model_id="...")

This is how a backend is swapped without changing the core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from core.interfaces import Classifier, Extractor, Gate, Generator, Reranker, Retriever

# Every stage kind that can be swapped, and the interface its backends must subclass.
STAGE_INTERFACES: dict[str, type] = {
    "classifier": Classifier,
    "extractor": Extractor,
    "retriever": Retriever,
    "reranker": Reranker,
    "gate": Gate,
    "generator": Generator,
}

# kind -> backend name -> class
_registry: dict[str, dict[str, type]] = {kind: {} for kind in STAGE_INTERFACES}

ClassT = TypeVar("ClassT", bound=type)


def _check_kind(kind: str) -> None:
    """Fail early with a clear message when the kind is misspelled."""
    if kind not in STAGE_INTERFACES:
        known = ", ".join(sorted(STAGE_INTERFACES))
        raise ValueError(f"Unknown stage kind '{kind}'. Known kinds: {known}")


def register(kind: str, name: str) -> Callable[[ClassT], ClassT]:
    """Class decorator that adds a backend under (kind, name)."""
    _check_kind(kind)

    def decorator(cls: ClassT) -> ClassT:
        interface = STAGE_INTERFACES[kind]
        if not issubclass(cls, interface):
            raise TypeError(f"{cls.__name__} must subclass {interface.__name__} to be a {kind} backend")
        if name in _registry[kind]:
            raise ValueError(f"A {kind} backend named '{name}' is already registered")
        _registry[kind][name] = cls
        return cls

    return decorator


def unregister(kind: str, name: str) -> None:
    """Remove a backend. Mainly used by tests to clean up after themselves."""
    _check_kind(kind)
    _registry[kind].pop(name, None)


def get_backend(kind: str, name: str) -> type:
    """Look up a backend class by kind and name."""
    _check_kind(kind)
    if name not in _registry[kind]:
        known = ", ".join(sorted(_registry[kind])) or "none registered"
        raise ValueError(f"No {kind} backend named '{name}'. Available: {known}")
    return _registry[kind][name]


def create(kind: str, name: str, **settings: Any) -> Any:
    """Build a backend, passing its config settings to the constructor."""
    return get_backend(kind, name)(**settings)


def available(kind: str) -> list[str]:
    """Names of all registered backends of one kind, sorted."""
    _check_kind(kind)
    return sorted(_registry[kind])
