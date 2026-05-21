"""Axis and ring-axis payload validation helpers."""

from typing import Any, Dict, Iterable, Optional, Set, Tuple

from graphistry.models.surfaces.graphistry_frontend.axis import (
    AXIS_BOUNDS_ALLOWED_KEYS,
    AXIS_ROW_ALLOWED_KEYS,
    AXIS_ROW_BOOL_KEYS,
    AXIS_ROW_NUMERIC_KEYS,
    AXIS_ROW_POSITION_KEYS,
)


def _is_non_bool_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _format_keys(keys: Iterable[str]) -> str:
    return ", ".join(sorted(keys))


def is_axis_bounds_payload(v: object) -> bool:
    return axis_bounds_payload_error(v, "bounds") is None


def axis_bounds_payload_error(v: object, path: str) -> Optional[str]:
    if not isinstance(v, dict):
        return f"{path} must be a dictionary"
    allowed_keys = set(AXIS_BOUNDS_ALLOWED_KEYS)
    for key in v.keys():
        if key not in allowed_keys:
            return f"{path} has unexpected key '{key}'; expected keys: {_format_keys(allowed_keys)}"
    if "min" in v and not _is_non_bool_number(v["min"]):
        return f"{path}['min'] must be a number"
    if "max" in v and not _is_non_bool_number(v["max"]):
        return f"{path}['max'] must be a number"
    return None


def is_axis_row_payload(v: object) -> bool:
    return axis_row_payload_error(v, "row") is None


def axis_row_payload_error(v: object, path: str) -> Optional[str]:
    if not isinstance(v, dict):
        return f"{path} must be a dictionary"
    allowed_keys = set(AXIS_ROW_ALLOWED_KEYS)
    for key in v.keys():
        if key not in allowed_keys:
            return f"{path} has unexpected key '{key}'; expected keys: {_format_keys(allowed_keys)}"
    if "label" in v and not isinstance(v["label"], str):
        return f"{path}['label'] must be a string"
    for k in AXIS_ROW_NUMERIC_KEYS:
        if k in v and not _is_non_bool_number(v[k]):
            return f"{path}['{k}'] must be a number"
    for k in AXIS_ROW_BOOL_KEYS:
        if k in v and not isinstance(v[k], bool):
            return f"{path}['{k}'] must be a boolean"
    if "bounds" in v:
        bounds_error = axis_bounds_payload_error(v["bounds"], f"{path}.bounds")
        if bounds_error is not None:
            return bounds_error
    if all(k not in v for k in AXIS_ROW_POSITION_KEYS):
        return (
            f"{path} is missing one of required position keys: "
            f"{_format_keys(AXIS_ROW_POSITION_KEYS)}; expected keys: {_format_keys(allowed_keys)}"
        )
    return None


def is_axis_rows_payload(v: object) -> bool:
    return axis_rows_payload_error(v, "rows") is None


def axis_rows_payload_error(v: object, path: str) -> Optional[str]:
    if not isinstance(v, list):
        return f"{path} must be a list of axis row dictionaries"
    for i, item in enumerate(v):
        row_error = axis_row_payload_error(item, f"{path}[{i}]")
        if row_error is not None:
            return row_error
    return None


_AXIS_SUBTYPE_KEYS: Tuple[str, ...] = ("axis_subtype", "axisKind", "kind")
_AXIS_DECLARATION_KEYS: Set[str] = set(_AXIS_SUBTYPE_KEYS)
_RADIAL_AXIS_ROW_KEYS: Set[str] = set(AXIS_ROW_ALLOWED_KEYS).union(_AXIS_DECLARATION_KEYS)
_LINEAR_AXIS_ROW_KEYS: Set[str] = (
    set(AXIS_ROW_ALLOWED_KEYS).difference({"r"}).union(_AXIS_DECLARATION_KEYS)
)


def _declared_axis_kind(row: Dict[Any, Any]) -> Optional[str]:
    for key in _AXIS_SUBTYPE_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value != "":
            return value.lower()
    return None


def _documented_axis_kind(row: Dict[Any, Any]) -> Optional[str]:
    declared = _declared_axis_kind(row)
    if declared is not None:
        return declared if declared in {"radial", "linear"} else "extension"
    if "r" in row:
        return "radial"
    if "x" in row or "y" in row:
        return "linear"
    if any(key in row for key in AXIS_ROW_ALLOWED_KEYS):
        return "axis"
    return None


def documented_axis_row_payload_error(v: object, path: str) -> Optional[str]:
    if not isinstance(v, dict):
        return f"{path} must be a dictionary"
    kind = _documented_axis_kind(v)
    if kind == "extension" or kind is None:
        return None

    if kind == "radial":
        allowed_keys = _RADIAL_AXIS_ROW_KEYS
        required_msg = "'r'"
        has_required = "r" in v
        label = "radial axis row"
    elif kind == "linear":
        allowed_keys = _LINEAR_AXIS_ROW_KEYS
        required_msg = "one of 'x' or 'y'"
        has_required = "x" in v or "y" in v
        label = "linear axis row"
    else:
        allowed_keys = set(AXIS_ROW_ALLOWED_KEYS).union(_AXIS_DECLARATION_KEYS)
        required_msg = f"one of {_format_keys(AXIS_ROW_POSITION_KEYS)}"
        has_required = any(key in v for key in AXIS_ROW_POSITION_KEYS)
        label = "axis row"

    for key in v.keys():
        if key not in allowed_keys:
            return f"{path} {label} has unexpected key '{key}'; expected keys: {_format_keys(allowed_keys)}"
    if not has_required:
        return f"{path} {label} is missing required key {required_msg}; expected keys: {_format_keys(allowed_keys)}"
    for key in AXIS_ROW_NUMERIC_KEYS:
        if key in v and not _is_non_bool_number(v[key]):
            return f"{path}['{key}'] must be a number"
    for key in AXIS_ROW_BOOL_KEYS:
        if key in v and not isinstance(v[key], bool):
            return f"{path}['{key}'] must be a boolean"
    if "label" in v and not isinstance(v["label"], str):
        return f"{path}['label'] must be a string"
    if "bounds" in v:
        bounds_error = axis_bounds_payload_error(v["bounds"], f"{path}.bounds")
        if bounds_error is not None:
            return bounds_error
    return None


def documented_axis_rows_payload_error(v: object, path: str) -> Optional[str]:
    if not isinstance(v, list):
        return f"{path} must be a list of axis row dictionaries"
    for i, item in enumerate(v):
        row_error = documented_axis_row_payload_error(item, f"{path}[{i}]")
        if row_error is not None:
            return row_error
    return None


def is_ring_continuous_axis_payload(v: object) -> bool:
    return ring_continuous_axis_payload_error(v, "axis") is None


def ring_continuous_axis_payload_error(v: object, path: str) -> Optional[str]:
    rows_error = axis_rows_payload_error(v, path)
    if rows_error is None:
        return None
    if isinstance(v, list):
        if any(isinstance(item, dict) for item in v):
            return rows_error
        for i, item in enumerate(v):
            if not isinstance(item, str):
                return f"{path}[{i}] must be a string label or axis row dictionary"
        return None
    if isinstance(v, dict):
        for key, val in v.items():
            if not _is_non_bool_number(key):
                return f"{path} key {key!r} must be a number"
            if not isinstance(val, str):
                return f"{path}[{key!r}] must be a string"
        return None
    return f"{path} must be axis rows, a list of string labels, or a numeric-to-string label map"


def is_ring_categorical_axis_payload(v: object) -> bool:
    return ring_categorical_axis_payload_error(v, "axis") is None


def ring_categorical_axis_payload_error(v: object, path: str) -> Optional[str]:
    rows_error = axis_rows_payload_error(v, path)
    if rows_error is None:
        return None
    if isinstance(v, list):
        return rows_error
    if isinstance(v, dict):
        for key, val in v.items():
            if not isinstance(val, str):
                return f"{path}[{key!r}] must be a string"
        return None
    return f"{path} must be axis rows or a categorical-to-string label map"
