"""Execution-scoped provider alias binding for semantic operators."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Iterator, Mapping, Tuple

from graphistry.compute.semantic.contract import DecisionProvider


PROVIDER_PROTOCOLS: Tuple[type, ...] = (DecisionProvider,)
_BOUND_PROVIDERS: "ContextVar[Mapping[str, object]]" = ContextVar("gfql_semantic_providers",
                                                                  default=MappingProxyType({}))


@contextmanager
def bind_providers(providers: Mapping[str, DecisionProvider]) -> Iterator[None]:
    """Bind provider aliases for semantic operators run inside this block.

    Inner blocks overlay outer ones alias by alias and restore them on exit. The mapping is copied on entry.
    Bindings follow ``contextvars`` rules: enter the block in the thread that calls ``g.gfql`` or
    ``g.sem_decide``.

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
        if not isinstance(provider, PROVIDER_PROTOCOLS):
            raise ValueError(f"provider bound to alias {alias!r} is not a semantic provider")
    token = _BOUND_PROVIDERS.set(MappingProxyType({**_BOUND_PROVIDERS.get(), **providers}))
    try:
        yield
    finally:
        _BOUND_PROVIDERS.reset(token)


def resolve_provider(provider: object) -> object:
    """The provider object itself, or the provider bound to an alias in the current context."""
    if not isinstance(provider, str):
        return provider
    bound = _BOUND_PROVIDERS.get()
    if provider not in bound:
        raise ValueError(f"semantic provider alias '{provider}' is not bound")
    return bound[provider]
