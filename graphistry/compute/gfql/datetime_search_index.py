"""Answer a numeric searchAny over a datetime column without rendering it.

``searchAny`` matches the text the viz inspector displays, and for a datetime that text is
``'MMM D YYYY, h:mm:ss a z'`` (see ``wysiwyg.render_datetime_pandas``). A datetime column is only
searched when the term matches ``/^[0-9.-]+$/``, and the separators between the render's
digit-bearing fields are a space, a comma and a colon -- none of which a term may contain. So a
term can never straddle two fields, and a row matches exactly when the term is a substring of one
field: the day, the year, the 12-hour hour, the zero-padded minute or second, or the zone label.

That turns a substring scan over rendered text into membership tests over small integers. Each
field holds at most 60 distinct values, so a term selects a handful of them and the row mask is a
gather. The columns are held as narrow integer arrays, eight bytes a row in total.

The equivalence is a claim about the format, so the tests check it against the render itself over
every legal substring of many timestamps rather than trusting this docstring.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Tuple

from graphistry.compute.typing import ArrayLike, ArrayNamespace, SeriesT

if TYPE_CHECKING:
    import numpy as np

#: Default bytes of index the process keeps; ``GRAPHISTRY_GFQL_DATETIME_INDEX_CACHE_BYTES`` overrides it.
_CACHE_BUDGET_BYTES = 512 * 1024 * 1024
_CACHE_BUDGET_ENV = "GRAPHISTRY_GFQL_DATETIME_INDEX_CACHE_BYTES"


def _cache_budget_bytes() -> int:
    """The eviction budget, from the environment when set to a positive integer."""
    raw = os.environ.get(_CACHE_BUDGET_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return _CACHE_BUDGET_BYTES
        if value > 0:
            return value
    return _CACHE_BUDGET_BYTES


def _is_cudf(s: SeriesT) -> bool:
    return "cudf" in type(s).__module__


def _array_module(s: SeriesT) -> ArrayNamespace:
    """numpy for pandas, cupy for cuDF: the index lives where the column lives."""
    if _is_cudf(s):
        import cupy
        return cupy  # type: ignore[return-value]
    import numpy
    return numpy  # type: ignore[return-value]


def _values(s: SeriesT) -> ArrayLike:
    """The column's array on its own device."""
    return s.values if _is_cudf(s) else s.to_numpy()  # type: ignore[return-value]


class DatetimeIndexCacheThrashWarning(RuntimeWarning):
    """A column's index was rebuilt right after being evicted: the budget is below the working set."""


class DatetimeSearchIndex:
    """Field components of a datetime column, and the mask a numeric term selects."""

    __slots__ = ("day", "year", "year_base", "hour12", "minute", "second",
                 "present", "zone", "zones", "n", "nbytes")

    #: Measured crossover: past this many selected values, a gather replaces the comparisons.
    _COMPARE_UPTO = 11

    def __init__(self, s: SeriesT, tz: str) -> None:
        xp = _array_module(s)
        on_gpu = _is_cudf(s)
        if on_gpu and tz != "UTC":
            # libcudf strftime ignores tz_convert: no zone but UTC can be named on the GPU
            from graphistry.compute.gfql.wysiwyg import CudfTemporalTzUnsupported
            raise CudfTemporalTzUnsupported(tz)
        if on_gpu:
            localized = s
        else:
            localized = (
                s.dt.tz_localize("UTC").dt.tz_convert(tz) if s.dt.tz is None else s.dt.tz_convert(tz)
            )
        self.n = len(s)
        self.present = _values(localized.notna())
        if not bool(self.present.all()):
            import pandas as pd
            filler = (localized[self.present].iloc[0] if bool(self.present.any())
                      else pd.Timestamp(0, tz=None if on_gpu else "UTC"))
            localized = localized.fillna(filler)

        hour_24 = _values(localized.dt.hour)
        self.day = _values(localized.dt.day).astype(xp.int16)
        years = _values(localized.dt.year)
        self.year_base = int(years.min()) if self.n else 0
        self.year = (years - self.year_base).astype(xp.int16)
        self.hour12 = xp.where(hour_24 % 12 == 0, 12, hour_24 % 12).astype(xp.int8)
        self.minute = _values(localized.dt.minute).astype(xp.int8)
        self.second = _values(localized.dt.second).astype(xp.int8)
        if on_gpu:
            self.zone, self.zones = xp.zeros(self.n, dtype=xp.int16), ["UTC"]
        else:
            self.zone, self.zones = _zone_codes(localized)
        self.nbytes = int(sum(a.nbytes for a in (
            self.day, self.year, self.hour12, self.minute, self.second, self.present, self.zone)))

    def _fields(self) -> List[Tuple[ArrayLike, int, Callable[[int], str]]]:
        """The per-row code array, alphabet size and value renderer for each rendered field."""
        year_span = int(self.year.max()) + 1 if self.n else 0
        return [
            (self.day, 32, str),
            (self.year, year_span, lambda v: str(self.year_base + v)),
            (self.hour12, 13, str),
            (self.minute, 60, lambda v: "%02d" % v),
            (self.second, 60, lambda v: "%02d" % v),
            # per row, not per column: +01 must select its own DST regime
            (self.zone, len(self.zones), lambda v: self.zones[v]),
        ]

    def matches(self, term: str) -> ArrayLike:
        """Rows whose rendered text would contain ``term``, as an array on the column's device.

        Each field contributes the rows whose value renders to text containing the term. Three
        things keep that from touching every row once per field:

        * a field NO value of which matches contributes nothing, and if that holds for every
          field the answer is empty without reading a row;
        * a field EVERY value of which matches means every present row matches, whatever the
          other fields say, so the answer is the present mask itself;
        * once the accumulated mask is all true the remaining fields cannot add to it.

        Within a field, comparing against each chosen value costs one pass per value, while
        building a lookup table and gathering through it costs one pass whatever the count.
        ``_COMPARE_UPTO`` is where those meet on this machine, so it is measured rather than
        derived from the format.
        """
        np = _module_of(self.present)

        live: List[Tuple[ArrayLike, int, List[int]]] = []
        for codes, size, render in self._fields():
            selected = [value for value in range(size) if term in render(value)]
            if not selected:
                continue
            if len(selected) == size:
                return self.present.copy()
            live.append((codes, size, selected))
        if not live:
            return np.zeros(self.n, dtype=bool)

        hits = np.zeros(self.n, dtype=bool)
        scratch = np.empty(self.n, dtype=bool)
        last = len(live) - 1
        for position, (codes, size, selected) in enumerate(live):
            if len(selected) <= self._COMPARE_UPTO:
                for value in selected:
                    np.equal(codes, value, out=scratch)
                    np.logical_or(hits, scratch, out=hits)
            else:
                table = np.zeros(size, dtype=bool)
                table[selected] = True
                np.take(table, codes, out=scratch)
                np.logical_or(hits, scratch, out=hits)
            # only worth asking while there is still a field that could add rows
            if position != last and bool(hits.all()):
                break
        np.logical_and(hits, self.present, out=hits)
        return hits


def _module_of(array: ArrayLike) -> ArrayNamespace:
    """The array module an existing array belongs to."""
    if type(array).__module__.startswith("cupy"):
        import cupy
        return cupy  # type: ignore[return-value]
    import numpy
    return numpy  # type: ignore[return-value]


def _zone_codes(localized: SeriesT) -> Tuple[ArrayLike, List[str]]:
    """Per-row zone label as a code plus its table, asked once per UTC offset not per row."""
    import numpy as np
    import pandas as pd

    if len(localized) == 0:
        return np.zeros(0, dtype=np.int8), []
    offset = localized.dt.tz_localize(None).astype("int64") - localized.astype("int64")
    codes, uniques = pd.factorize(offset)
    labels = []
    for value in uniques:
        rows = localized[offset == value]
        labels.append(pd.Timestamp(rows.iloc[0]).strftime("%Z") if len(rows) else "")
    return codes.astype(np.int16), labels


_CacheKey = Tuple[bytes, str, str, int, str]
_CACHE: "OrderedDict[_CacheKey, DatetimeSearchIndex]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
#: Keys evicted most recently; re-inserting one of these is the signature of a thrashing cache.
_RECENTLY_EVICTED: Deque[_CacheKey] = deque(maxlen=8)
#: ``events`` counts rebuilds-after-eviction; ``warned`` marks the one warning per process.
_THRASH: Dict[str, int] = {}


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _RECENTLY_EVICTED.clear()
        _THRASH.clear()


def thrash_events() -> int:
    """How many times an index was rebuilt right after its eviction since the last clear."""
    return _THRASH.get("events", 0)


def _cache_key(s: SeriesT, tz: str) -> Optional[_CacheKey]:
    """Content digest of the column, so an in-place edit cannot serve a stale index.

    Identity is deliberately not used: keying a memo on ``id()`` serves a stale answer once the
    frame is mutated in place. The digest reads the timestamps' own buffer, which is
    ``int64`` nanoseconds whatever the zone, and is taken over a memoryview so nothing is copied.

    The dtype is part of the key because the buffer alone does not say what the integers MEAN:
    the same bytes read as nanoseconds and as microseconds are different instants, and a column of
    each would otherwise share an entry.

    It must be ORDER sensitive -- the index is row-ordered, so a sorted column is a different
    index even though its values are the same. That rules out a sum or an xor, both of which a
    permutation leaves untouched. ``None`` means the column cannot be digested, so it is not
    cached rather than cached wrongly.

    SHA-2 rather than BLAKE2 because this runs once per search over the whole column, and the
    machines it runs on implement SHA-2 as an instruction while BLAKE2 has to be executed; the
    two are equally sound here and the measurement lives in pyg-bench. Truncating to sixteen
    bytes keeps the key the size it was -- a memo this small does not need more.
    """
    import hashlib

    import numpy as np

    try:
        if _is_cudf(s):
            # digested on a host copy; a device digest would need a table larger than the index
            raw = np.ascontiguousarray(s.values.view("int64").get())
        else:
            raw = np.ascontiguousarray(s.to_numpy()).view(np.int64)
        digest = hashlib.sha256(memoryview(raw)).digest()[:16]
    except (TypeError, ValueError, AttributeError):
        return None
    # same bytes on host and device are the same instants but not the same arrays
    return (digest, str(s.dtype), tz, len(s), "cudf" if _is_cudf(s) else "pandas")


def index_for(s: SeriesT, tz: str) -> DatetimeSearchIndex:
    """The index for this column and zone, built once and reused across searches."""
    key = _cache_key(s, tz)
    if key is None:
        return DatetimeSearchIndex(s, tz)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    built = DatetimeSearchIndex(s, tz)
    with _CACHE_LOCK:
        rebuilt_after_eviction = key in _RECENTLY_EVICTED
        _CACHE[key] = built
        _CACHE.move_to_end(key)
        budget = _cache_budget_bytes()
        held = sum(entry.nbytes for entry in _CACHE.values())
        evicted_bytes = 0
        while len(_CACHE) > 1 and held > budget:
            evicted_key, evicted = _CACHE.popitem(last=False)
            _RECENTLY_EVICTED.append(evicted_key)
            held -= evicted.nbytes
            evicted_bytes += evicted.nbytes
        warn_now = False
        if rebuilt_after_eviction:
            _THRASH["events"] = _THRASH.get("events", 0) + 1
            warn_now = not _THRASH.get("warned", 0)
            _THRASH["warned"] = 1
    if warn_now:
        warnings.warn(
            "datetime search index rebuilt immediately after being evicted: the working set "
            "exceeds the cache budget of %d bytes (this index is %d bytes; %d bytes were evicted "
            "to admit it). Every search over this frame is paying a full rebuild. Raise %s."
            % (budget, built.nbytes, evicted_bytes, _CACHE_BUDGET_ENV),
            DatetimeIndexCacheThrashWarning, stacklevel=2)
    return built


from graphistry.compute.gfql.cache_registry import register_clearable_dict  # noqa: E402

register_clearable_dict("_CACHE", _CACHE)
register_clearable_dict("_RECENTLY_EVICTED", _RECENTLY_EVICTED)
register_clearable_dict("_THRASH", _THRASH)
