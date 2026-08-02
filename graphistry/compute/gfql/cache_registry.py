"""Central ledger of every process-lifetime GFQL cache.

Each cache SETUP SITE registers itself here, adjacent to its own definition, as
either CLEARABLE (keyed by caller input; ``gfql_clear_caches`` empties it) or a
PROCESS SINGLETON (a function of the code, exempt, with the written reason).

Why a registry instead of a central list of names: the bug that motivated this
file was ``gfql_clear_caches`` looking a clear target up BY NAME and silently
doing nothing when the memo actually lived on a different object
(``parse_cypher`` vs ``_parse_cypher_cached``). A registration hands over the
bound ``cache_clear``/``dict.clear`` of the real object at definition time, so
there is no later lookup to get wrong. Classification and exemption reasons
live next to the cache they describe, and the coverage lock in
``graphistry/tests/compute/gfql/test_clear_caches_covers_every_cache.py``
statically discovers every cache in the tree and fails when one is not
registered -- registration is enforced, not optional.

A module that is never imported registers nothing, and that is correct: its
cache object does not exist, so there is nothing to clear.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, MutableMapping, NamedTuple, Optional


class CacheEntry(NamedTuple):
    name: str
    clear: Optional[Callable[[], None]]  # None => exempt process singleton
    reason: Optional[str]                # None => clearable


_REGISTRY: Dict[str, CacheEntry] = {}
_LOCK = threading.Lock()


def _register(entry: CacheEntry) -> None:
    with _LOCK:
        existing = _REGISTRY.get(entry.name)
        if existing is not None and existing != entry:
            raise ValueError(f"cache {entry.name!r} registered twice with different handles")
        _REGISTRY[entry.name] = entry


def register_clearable(fn: Any, name: Optional[str] = None) -> None:
    """Register an ``@lru_cache`` function whose memo gfql_clear_caches must empty."""
    clear = getattr(fn, "cache_clear", None)
    if clear is None:
        raise TypeError(f"{fn!r} has no cache_clear; register the decorated function itself")
    _register(CacheEntry(name or fn.__name__, clear, None))


def register_clearable_dict(name: str, mapping: MutableMapping[Any, Any]) -> None:
    """Register a hand-rolled dict/OrderedDict memo for clearing."""
    _register(CacheEntry(name, mapping.clear, None))


def register_clearable_callable(name: str, clear: Callable[[], None]) -> None:
    """Register a clear callable for a cache that needs its own locking discipline."""
    _register(CacheEntry(name, clear, None))


def register_process_singleton(fn: Any, reason: str) -> None:
    """Exempt a maxsize=1, function-of-the-code cache, with the reason in writing."""
    if len(reason.split()) < 6:
        raise ValueError(f"{fn.__name__}: exemption reason is too thin to be a reason")
    _register(CacheEntry(fn.__name__, None, reason))


def clear_all() -> None:
    """Empty every registered clearable cache. Raises if nothing is registered."""
    with _LOCK:
        entries = list(_REGISTRY.values())
    if not any(entry.clear is not None for entry in entries):
        raise RuntimeError(
            "no clearable GFQL cache is registered; the registry import wiring is broken"
        )
    for entry in entries:
        if entry.clear is not None:
            entry.clear()


def entries() -> Dict[str, CacheEntry]:
    """Snapshot for the coverage lock test."""
    with _LOCK:
        return dict(_REGISTRY)
