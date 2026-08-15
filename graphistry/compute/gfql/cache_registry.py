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

Threading: registration runs at import time under the import lock plus this
module's own lock; ``clear_all`` snapshots under the lock and clears outside it.
``lru_cache.cache_clear`` is internally locked; dict-style memos must make their
hit paths recompute-safe against a clear racing a read (see the single-alias
memo). ``importlib.reload`` of a cache host raises the registered-twice error on
purpose: the reloaded module would leave a zombie cache this registry can no
longer reach, and that deserves a loud failure, not a silent overwrite.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, NamedTuple, Optional, Protocol


class _SupportsCacheClear(Protocol):
    """An ``@lru_cache``-decorated function (mypy's wrapper stub carries no __name__,
    so the name is read with getattr at runtime, where functools always sets it)."""

    def cache_clear(self) -> None: ...


class _SupportsClear(Protocol):
    """A dict-like memo: anything with an argument-free ``clear``."""

    def clear(self) -> None: ...


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


def register_clearable(fn: _SupportsCacheClear, name: Optional[str] = None) -> None:
    """Register an ``@lru_cache`` function whose memo gfql_clear_caches must empty."""
    clear = getattr(fn, "cache_clear", None)  # runtime check: typing cannot stop a bare fn
    if clear is None:
        raise TypeError(f"{fn!r} has no cache_clear; register the decorated function itself")
    resolved = name if name is not None else str(getattr(fn, "__name__", repr(fn)))
    _register(CacheEntry(resolved, clear, None))


def register_clearable_dict(name: str, mapping: _SupportsClear) -> None:
    """Register a hand-rolled dict/OrderedDict memo for clearing."""
    _register(CacheEntry(name, mapping.clear, None))


def register_clearable_callable(name: str, clear: Callable[[], None]) -> None:
    """Register a clear callable for a cache that needs its own locking discipline."""
    _register(CacheEntry(name, clear, None))


def register_exempt(name: str, reason: str) -> None:
    """Exempt a discovered process-global by NAME, with the reason in writing.

    For state the coverage lock finds but ``gfql_clear_caches`` must NOT empty --
    deliberate process configuration, or this ledger itself. Named rather than
    handed a callable because the thing being exempted is often a bare container.
    """
    if len(reason.split()) < 6:
        raise ValueError(f"{name}: exemption reason is too thin to be a reason")
    _register(CacheEntry(name, None, reason))


def register_process_singleton(fn: _SupportsCacheClear, reason: str) -> None:
    """Exempt a maxsize=1, function-of-the-code cache, with the reason in writing."""
    fn_name = str(getattr(fn, "__name__", repr(fn)))
    register_exempt(fn_name, reason)


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


# This ledger is itself module-level mutable state, so the widened coverage lock
# discovers it (#1913). It is exempt by construction: clearing it would delete the
# clear handles and turn gfql_clear_caches into the silent no-op this file exists
# to prevent, and clear_all() already raises on an empty registry.
register_exempt("_REGISTRY", "the cache ledger itself; emptying it would delete every clear handle and make gfql_clear_caches a silent no-op")
