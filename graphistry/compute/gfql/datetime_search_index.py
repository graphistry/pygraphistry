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

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from graphistry.compute.typing import SeriesT

if TYPE_CHECKING:
    import numpy as np

#: Bytes of index the process keeps. A datetime column costs eight bytes a row, so this holds a
#: 30M-row column and evicts the least recently used beyond that.
_CACHE_BUDGET_BYTES = 512 * 1024 * 1024


class DatetimeSearchIndex:
    """Field components of a datetime column, and the mask a numeric term selects."""

    __slots__ = ("day", "year", "year_base", "hour12", "minute", "second",
                 "present", "zone", "zones", "n", "nbytes")

    def __init__(self, s: SeriesT, tz: str) -> None:
        import numpy as np
        import pandas as pd

        localized = (
            s.dt.tz_localize("UTC").dt.tz_convert(tz) if s.dt.tz is None else s.dt.tz_convert(tz)
        )
        self.n = len(s)
        self.present = localized.notna().to_numpy()
        if not self.present.all():
            filler = (localized[self.present].iloc[0] if self.present.any()
                      else pd.Timestamp(0, tz="UTC"))
            localized = localized.fillna(filler)

        hour_24 = localized.dt.hour.to_numpy()
        self.day = localized.dt.day.to_numpy().astype(np.int16)
        years = localized.dt.year.to_numpy()
        self.year_base = int(years.min()) if self.n else 0
        self.year = (years - self.year_base).astype(np.int16)
        self.hour12 = np.where(hour_24 % 12 == 0, 12, hour_24 % 12).astype(np.int8)
        self.minute = localized.dt.minute.to_numpy().astype(np.int8)
        self.second = localized.dt.second.to_numpy().astype(np.int8)
        self.zone, self.zones = _zone_codes(localized)
        self.nbytes = int(sum(a.nbytes for a in (
            self.day, self.year, self.hour12, self.minute, self.second, self.present, self.zone)))

    def matches(self, term: str) -> "np.ndarray":
        """Rows whose rendered text would contain ``term``."""
        import numpy as np

        hits = np.zeros(self.n, dtype=bool)
        year_span = int(self.year.max()) + 1 if self.n else 0
        fields: List[Tuple["np.ndarray", int, Callable[[int], str]]] = [
            (self.day, 32, str),
            (self.year, year_span, lambda v: str(self.year_base + v)),
            (self.hour12, 13, str),
            (self.minute, 60, lambda v: "%02d" % v),
            (self.second, 60, lambda v: "%02d" % v),
            # per row, not per column: +01 must select its own DST regime
            (self.zone, len(self.zones), lambda v: self.zones[v]),
        ]
        for codes, size, render in fields:
            selected = [value for value in range(size) if term in render(value)]
            if not selected:
                continue
            table = np.zeros(size, dtype=bool)
            table[selected] = True
            hits |= table[codes]
        return hits & self.present


def _zone_codes(localized: SeriesT) -> Tuple["np.ndarray", List[str]]:
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


_CACHE: "OrderedDict[Tuple[bytes, str, str, int], DatetimeSearchIndex]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _cache_key(s: SeriesT, tz: str) -> Optional[Tuple[bytes, str, str, int]]:
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
    """
    import hashlib

    import numpy as np

    try:
        raw = np.ascontiguousarray(s.to_numpy()).view(np.int64)
        digest = hashlib.blake2b(memoryview(raw), digest_size=16).digest()
    except (TypeError, ValueError, AttributeError):
        return None
    return (digest, str(s.dtype), tz, len(s))


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
        _CACHE[key] = built
        _CACHE.move_to_end(key)
        held = sum(entry.nbytes for entry in _CACHE.values())
        while len(_CACHE) > 1 and held > _CACHE_BUDGET_BYTES:
            _, evicted = _CACHE.popitem(last=False)
            held -= evicted.nbytes
    return built


from graphistry.compute.gfql.cache_registry import register_clearable_dict  # noqa: E402

register_clearable_dict("_CACHE", _CACHE)
