"""Execution-scoped provider alias binding for semantic operators."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Iterator, Mapping, Optional, Tuple, Union

from graphistry.compute.semantic.cache import SemanticCache
from graphistry.compute.semantic.contract import DecisionProvider, GenerationProvider


PROVIDER_PROTOCOLS: Tuple[type, ...] = (DecisionProvider, GenerationProvider)
_BOUND_PROVIDERS: "ContextVar[Mapping[str, object]]" = ContextVar("gfql_semantic_providers",
                                                                  default=MappingProxyType({}))
_BOUND_CACHE: "ContextVar[Optional[SemanticCache]]" = ContextVar("gfql_semantic_cache", default=None)


@contextmanager
def bind_providers(providers: Mapping[str, Union[DecisionProvider, GenerationProvider]], *,
                   cache: Optional[SemanticCache] = None) -> Iterator[None]:
    """Bind provider aliases, and optionally a result cache, for semantic operators run inside this block.

    Inner blocks overlay outer ones alias by alias and restore them on exit. The mapping is copied on entry.
    ``cache`` is a :class:`~graphistry.compute.semantic.cache.SemanticCache` that every ``sem_decide`` and ``sem_map``
    run inside the block uses, for provider objects as well as aliases; an inner block without a ``cache`` keeps the
    outer one. There is no default cache: you own the instance, so a server can bind one cache per tenant.
    Bindings follow ``contextvars`` rules: enter the block in the thread that calls ``g.gfql``,
    ``g.sem_decide`` or ``g.sem_map``.

    **Example**
        ::

            with bind_providers({"fast": my_provider}, cache=MemorySemanticCache()):
                g2 = g.gfql([call("sem_decide", {"provider": "fast", ...})])
    """
    if not isinstance(providers, Mapping):
        raise ValueError(f"bind_providers expects a mapping of alias -> provider, got {type(providers).__name__}")
    for alias, provider in providers.items():
        if not isinstance(alias, str) or alias == "":
            raise ValueError(f"provider aliases must be non-empty strings, got {alias!r}")
        if not isinstance(provider, PROVIDER_PROTOCOLS):
            raise ValueError(f"provider bound to alias {alias!r} is not a semantic provider")
    if cache is not None and not isinstance(cache, SemanticCache):
        raise ValueError("cache must have get_many, put_many and clear (a SemanticCache)")
    token = _BOUND_PROVIDERS.set(MappingProxyType({**_BOUND_PROVIDERS.get(), **providers}))
    cache_token = _BOUND_CACHE.set(_BOUND_CACHE.get() if cache is None else cache)
    try:
        yield
    finally:
        _BOUND_CACHE.reset(cache_token)
        _BOUND_PROVIDERS.reset(token)


def bound_cache() -> Optional[SemanticCache]:
    """The result cache bound in the current context, or None."""
    return _BOUND_CACHE.get()


def resolve_provider(provider: object) -> object:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    """The provider object itself, or the provider bound to an alias in the current context."""
    if not isinstance(provider, str):
        return provider
    bound = _BOUND_PROVIDERS.get()
    if provider not in bound:
        raise ValueError(f"semantic provider alias '{provider}' is not bound")
    return bound[provider]
