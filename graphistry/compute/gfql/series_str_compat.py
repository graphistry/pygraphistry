"""Compatibility wrappers for Series string accessors across pandas and cuDF."""

from __future__ import annotations

import re
from typing import Any, Optional, cast

from graphistry.compute.typing import SeriesT


def _fill_string_mask_na(mask: Any, series: Any, na: Optional[bool]) -> Any:
    if na is None or not hasattr(mask, "where") or not hasattr(series, "isna"):
        return mask
    return mask.where(~series.isna(), na)


def _anchored_fullmatch_pattern(pattern: str) -> str:
    if not pattern.startswith("^"):
        pattern = "^" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + "$"
    return pattern


def _sanitize_regex_pattern(pattern: str) -> str:
    pattern = re.sub(r"\(\?P<[^>]+>", "(", pattern)
    pattern = pattern.replace("(?:", "(")
    return pattern


def _sanitize_extract_pattern(pattern: str) -> str:
    return re.sub(r"\(\?P<[^>]+>", "(", pattern)


def _invoke_str_method(method: Any, pattern: str, *, na: Optional[bool] = None, **kwargs: Any) -> Any:
    if na is None:
        return method(pattern, **kwargs)
    try:
        return method(pattern, na=na, **kwargs)
    except (TypeError, NotImplementedError):
        return method(pattern, **kwargs)




def _pattern_group_names(pattern: str) -> list[Optional[str]]:
    compiled = re.compile(pattern)
    names: list[Optional[str]] = [None] * compiled.groups
    for name, idx in compiled.groupindex.items():
        names[idx - 1] = name
    return names


def _rename_extract_columns(frame: Any, names: list[Optional[str]]) -> Any:
    cols = list(getattr(frame, "columns", []))
    rename_map = {
        cols[i]: name
        for i, name in enumerate(names[: len(cols)])
        if name is not None and cols[i] != name
    }
    if rename_map and hasattr(frame, "rename"):
        return frame.rename(columns=rename_map)
    return frame

def _call_str_method(series: SeriesT, method_name: str, pattern: str, *, na: Optional[bool] = None, **kwargs: Any) -> SeriesT:
    if not hasattr(series.str, method_name):
        if method_name != "fullmatch":
            raise AttributeError(f"String accessor has no method {method_name!r}")
        return _call_str_method(
            series,
            "match",
            _sanitize_regex_pattern(_anchored_fullmatch_pattern(pattern)),
            na=na,
            **kwargs,
        )

    method = getattr(series.str, method_name)
    try:
        result = _invoke_str_method(method, pattern, na=na, **kwargs)
    except Exception:
        # cuDF rejects some regex constructs that pandas accepts,
        # so retry once with a simplified equivalent pattern.
        safe_pattern = _sanitize_regex_pattern(pattern)
        if safe_pattern == pattern:
            raise
        result = _invoke_str_method(method, safe_pattern, na=na, **kwargs)
    return cast(SeriesT, _fill_string_mask_na(result, series, na))


def series_str_match(series: SeriesT, pattern: str, *, na: Optional[bool] = None) -> SeriesT:
    return _call_str_method(series, "match", pattern, na=na)


def series_str_fullmatch(series: SeriesT, pattern: str, *, na: Optional[bool] = None) -> SeriesT:
    return _call_str_method(series, "fullmatch", pattern, na=na)


def series_str_contains(series: SeriesT, pattern: str, *, na: Optional[bool] = None, regex: bool = True) -> SeriesT:
    return _call_str_method(series, "contains", pattern, na=na, regex=regex)


def series_sequence_len(series: SeriesT) -> SeriesT:
    if hasattr(series, "str") and hasattr(series.str, "len"):
        try:
            return cast(SeriesT, series.str.len())
        except Exception:
            pass
    if hasattr(series, "list") and hasattr(series.list, "len"):
        return cast(SeriesT, series.list.len())
    raise AttributeError("Series does not support sequence length via .str.len() or .list.len()")



def series_str_extract(series: SeriesT, pattern: str) -> Any:
    group_names = _pattern_group_names(pattern)
    try:
        result = series.str.extract(pattern)
    except Exception:
        safe_pattern = _sanitize_extract_pattern(pattern)
        if safe_pattern == pattern:
            raise
        result = series.str.extract(safe_pattern)
    return _rename_extract_columns(result, group_names)
