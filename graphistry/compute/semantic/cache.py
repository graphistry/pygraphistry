"""Result cache for semantic operators: the ``SemanticCache`` protocol and a bounded in-memory implementation.

A cache holds validated provider answers, keyed by a hash of the schema version, the provider fingerprint and the
input hash. Values are JSON strings, so a server can back the protocol with Redis or a database. No cache is
bound by default and this module keeps no process-wide state: the caller owns every ``MemorySemanticCache``.
"""
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable


CACHE_SCHEMA_VERSION = "1"
DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TTL_S = 3600.0


@runtime_checkable
class SemanticCache(Protocol):
    """Where semantic operators keep answers between runs.

    Keys are hex digests and values are JSON strings. ``get_many`` returns only the keys it holds. An operator treats
    an exception, or a reply that is not a mapping of strings, as a miss, and a failed ``put_many`` as a skipped
    write: a cache is an optimisation and never fails a run. Implementations must be safe to call from several threads.
    """

    def get_many(self, keys: Sequence[str]) -> Mapping[str, str]:
        ...

    def put_many(self, entries: Mapping[str, str]) -> None:
        ...

    def clear(self) -> None:
        ...


def cache_key(provider_fingerprint: str, input_hash: str, salt: str = "") -> str:
    """The cache key of one answer: a digest of the schema version, the provider fingerprint, the input hash and a
    salt for settings the input hash does not cover. It never holds evidence, keys or tenant names."""
    payload = json.dumps([CACHE_SCHEMA_VERSION, provider_fingerprint, input_hash, salt], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_fingerprint(provider: object) -> Optional[str]:
    """What a provider says its answers depend on (its ``cache_fingerprint()``), or None when it has none.
    Provider code is untrusted: a failing or non-string result is None."""
    hook = getattr(provider, "cache_fingerprint", None)
    if not callable(hook):
        return None
    try:
        value = hook()
    except Exception:
        return None
    return value if isinstance(value, str) and value != "" else None


def fingerprint_of(settings: Mapping[str, object]) -> str:
    """A digest of the settings that shape a provider's answers; callers leave out keys and headers."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _positive(value: object, name: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    if not ok:
        raise ValueError(f"{name} must be a positive number{' or None' if allow_none else ''}, got {value!r}")


class MemorySemanticCache:
    """A thread-safe LRU cache in process memory with an entry cap, a byte cap and a time to live.

    :param max_entries: most entries kept; the least recently used go first
    :param max_bytes: most bytes of keys and values kept; one entry larger than this is not stored
    :param ttl_s: seconds an entry lives after it was written; None keeps it until it is evicted
    :param clock: monotonic clock in seconds, for tests

    **Example**
        ::

            cache = MemorySemanticCache(max_entries=50_000, ttl_s=900)
            with bind_providers({"fast": provider}, cache=cache):
                out = g.sem_decide(questions, "fast", task_id="t", evidence_cols=["name"])
            cache.stats()
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, max_bytes: int = DEFAULT_MAX_BYTES,
                 ttl_s: Optional[float] = DEFAULT_TTL_S, *, clock: Callable[[], float] = time.monotonic) -> None:
        for value, name in ((max_entries, "max_entries"), (max_bytes, "max_bytes")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        _positive(ttl_s, "ttl_s", allow_none=True)
        self._max_entries, self._max_bytes, self._ttl_s, self._clock = max_entries, max_bytes, ttl_s, clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Tuple[str, float, int]]" = OrderedDict()
        self._bytes = 0
        self._counts: Dict[str, int] = dict(hits=0, misses=0, puts=0, evictions=0, expirations=0, rejected=0)

    def _drop(self, key: str) -> None:
        _value, _written, size = self._entries.pop(key)
        self._bytes -= size

    def _expired(self, written: float, now: float) -> bool:
        return self._ttl_s is not None and now - written >= self._ttl_s

    def get_many(self, keys: Sequence[str]) -> Dict[str, str]:
        found: Dict[str, str] = {}
        with self._lock:
            now = self._clock()
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and self._expired(entry[1], now):
                    self._drop(key)
                    self._counts["expirations"] += 1
                    entry = None
                if entry is None:
                    self._counts["misses"] += 1
                    continue
                self._entries.move_to_end(key)
                self._counts["hits"] += 1
                found[key] = entry[0]
        return found

    def put_many(self, entries: Mapping[str, str]) -> None:
        with self._lock:
            now = self._clock()
            for key, value in entries.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("cache keys and values must be strings")
                size = len(key.encode("utf-8")) + len(value.encode("utf-8"))
                if key in self._entries:
                    self._drop(key)
                if size > self._max_bytes:
                    self._counts["rejected"] += 1
                    continue
                self._entries[key] = (value, now, size)
                self._bytes += size
                self._counts["puts"] += 1
            self._shrink(now)

    def _shrink(self, now: float) -> None:
        if len(self._entries) <= self._max_entries and self._bytes <= self._max_bytes:
            return
        for key in [key for key, (_value, written, _size) in self._entries.items() if self._expired(written, now)]:
            self._drop(key)
            self._counts["expirations"] += 1
        while self._entries and (len(self._entries) > self._max_entries or self._bytes > self._max_bytes):
            self._drop(next(iter(self._entries)))
            self._counts["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> Dict[str, Union[int, float]]:
        """Counters since creation (``hits``, ``misses``, ``puts``, ``evictions``, ``expirations``, ``rejected``) and the
        current ``entries`` and ``bytes``; ``hit_rate`` is hits over lookups, 0.0 before the first lookup. It first
        removes the entries that have expired. A write of a value larger than ``max_bytes`` removes the old value of
        that key and counts as ``rejected``."""
        with self._lock:
            for key in [key for key, (_value, written, _size) in self._entries.items() if self._expired(written, self._clock())]:
                self._drop(key)
                self._counts["expirations"] += 1
            lookups = self._counts["hits"] + self._counts["misses"]
            return {**self._counts, "entries": len(self._entries), "bytes": self._bytes,
                    "hit_rate": self._counts["hits"] / lookups if lookups else 0.0}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
