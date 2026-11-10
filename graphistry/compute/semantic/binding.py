"""Execution-scoped provider alias binding for semantic operators."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Iterator, Mapping, Union

from graphistry.compute.semantic.contract import DecisionProvider, GenerationProvider


SemanticProvider = Union[DecisionProvider, GenerationProvider]

_EMPTY: Mapping[str, SemanticProvider] = MappingProxyType({})
_BOUND_PROVIDERS: "ContextVar[Mapping[str, SemanticProvider]]" = ContextVar(
    "gfql_semantic_providers", default=_EMPTY
)


@contextmanager
def bind_providers(providers: Mapping[str, SemanticProvider]) -> Iterator[None]:
    """Bind provider aliases for semantic operators run inside this block.

    Inner blocks overlay outer ones alias by alias and restore them on exit. The mapping is
    copied on entry. Bindings follow ``contextvars`` rules: enter the block in the thread that
    calls ``g.gfql`` or ``g.sem_decide``.

    :param providers: alias name -> provider object
    :type providers: Mapping[str, Union[DecisionProvider, GenerationProvider]]

    **Example**
        ::

            with bind_providers({"fast": my_provider}):
                g2 = g.gfql([call("sem_decide", {"provider": "fast", ...})])
    """
    if not isinstance(providers, Mapping):
        raise ValueError(f"bind_providers expects a mapping of alias -> provider, got {type(providers).__name__}")
    for alias, provider in providers.items():
        if not isinstance(alias, str) or alias == "":
            raise ValueError(f"provider aliases must be non-empty strings, got {alias!r}")
        if not isinstance(provider, (DecisionProvider, GenerationProvider)):
            raise ValueError(f"provider bound to alias {alias!r} is not a DecisionProvider or GenerationProvider")
    merged = dict(_BOUND_PROVIDERS.get())
    merged.update(providers)
    token = _BOUND_PROVIDERS.set(MappingProxyType(merged))
    try:
        yield
    finally:
        _BOUND_PROVIDERS.reset(token)


def bound_providers() -> Mapping[str, SemanticProvider]:
    """Read-only view of the aliases bound in the current context."""
    return _BOUND_PROVIDERS.get()


def resolve_decision_provider(provider: Union[str, DecisionProvider]) -> DecisionProvider:
    resolved = _resolve(provider)
    if not isinstance(resolved, DecisionProvider):
        raise ValueError(f"semantic provider {_describe(provider)} is not a DecisionProvider")
    return resolved


def resolve_generation_provider(provider: Union[str, GenerationProvider]) -> GenerationProvider:
    resolved = _resolve(provider)
    if not isinstance(resolved, GenerationProvider):
        raise ValueError(f"semantic provider {_describe(provider)} is not a GenerationProvider")
    return resolved


def _resolve(provider: Union[str, SemanticProvider]) -> object:
    if isinstance(provider, str):
        bound = _BOUND_PROVIDERS.get()
        if provider not in bound:
            raise ValueError(f"semantic provider alias '{provider}' is not bound")
        return bound[provider]
    return provider


def _describe(provider: object) -> str:
    return f"alias '{provider}'" if isinstance(provider, str) else type(provider).__name__
