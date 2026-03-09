from __future__ import annotations

import re
from typing import Any, Optional, cast

from graphistry.compute.typing import SeriesT


_ENTITY_NODE_LABELS_RE = re.compile(r"^\((?P<labels>(?::[A-Za-z_][A-Za-z0-9_]*)*)(?:\s*\{.*\})?\)$")
_ENTITY_EDGE_TYPE_RE = re.compile(r"^\[:(?P<type>[^\]\s]+)(?:\s+\{.*\})?\]$")
_ENTITY_PROPERTIES_RE = re.compile(r"(\{.*\})")
_ENTITY_PROPERTY_VALUE_RE = r"(?:'(?:\\.|[^'\\])*'|true|false|null|[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)|\[[^\]]*\]|\{[^{}]*\})"


def is_entity_text_scalar(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (
        _ENTITY_NODE_LABELS_RE.fullmatch(stripped) is not None
        or _ENTITY_EDGE_TYPE_RE.fullmatch(stripped) is not None
        or stripped in {"()", "[]"}
    )


def entity_labels_series(series: SeriesT) -> SeriesT:
    text = cast(SeriesT, series.astype(str))
    labels = cast(SeriesT, text.str.extract(_ENTITY_NODE_LABELS_RE.pattern, expand=False))
    labels = cast(SeriesT, labels.where(labels.notna(), ""))
    label_body = cast(SeriesT, labels.str.lstrip(":"))
    rendered = cast(SeriesT, "['" + label_body.str.replace(":", "', '", regex=False) + "']")
    return cast(SeriesT, rendered.where(label_body != "", "[]"))


def entity_labels_scalar(value: Any) -> Optional[str]:
    if not is_entity_text_scalar(value):
        return None
    match = _ENTITY_NODE_LABELS_RE.fullmatch(str(value).strip())
    if match is None:
        return "[]"
    labels_txt = match.group("labels") or ""
    if labels_txt == "":
        return "[]"
    labels = [label for label in labels_txt.split(":") if label]
    return "[" + ", ".join(f"'{label}'" for label in labels) + "]"


def entity_type_series(series: SeriesT) -> SeriesT:
    text = cast(SeriesT, series.astype(str))
    return cast(SeriesT, text.str.extract(_ENTITY_EDGE_TYPE_RE.pattern, expand=False))


def entity_type_scalar(value: Any) -> Optional[str]:
    if not is_entity_text_scalar(value):
        return None
    match = _ENTITY_EDGE_TYPE_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    return cast(Optional[str], match.group("type"))


def entity_properties_series(series: SeriesT) -> SeriesT:
    text = cast(SeriesT, series.astype(str))
    props = cast(SeriesT, text.str.extract(_ENTITY_PROPERTIES_RE.pattern, expand=False))
    return cast(SeriesT, props.where(props.notna(), "{}"))


def entity_properties_scalar(value: Any) -> Optional[str]:
    if not is_entity_text_scalar(value):
        return None
    match = _ENTITY_PROPERTIES_RE.search(str(value).strip())
    if match is None:
        return "{}"
    return cast(Optional[str], match.group(1))


def _coerce_property_value_series(raw: SeriesT) -> SeriesT:
    out = cast(SeriesT, raw.astype("object"))
    missing_mask = cast(SeriesT, raw.isna())
    null_mask = cast(SeriesT, raw == "null")
    true_mask = cast(SeriesT, raw == "true")
    false_mask = cast(SeriesT, raw == "false")
    quoted_mask = cast(SeriesT, raw.str.match(r"^'(?:\\.|[^'\\])*'$", na=False))
    numeric_mask = cast(
        SeriesT,
        raw.str.match(r"^[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)$", na=False),
    )
    integer_mask = cast(SeriesT, raw.str.match(r"^[+-]?\d+$", na=False))
    float_mask = cast(SeriesT, numeric_mask & ~integer_mask)

    out = cast(SeriesT, out.where(~missing_mask, None))
    out = cast(SeriesT, out.where(~null_mask, None))
    out = cast(SeriesT, out.where(~true_mask, True))
    out = cast(SeriesT, out.where(~false_mask, False))

    unquoted = cast(SeriesT, raw.str.slice(1, -1))
    unquoted = cast(SeriesT, unquoted.str.replace("\\'", "'", regex=False))
    unquoted = cast(SeriesT, unquoted.str.replace("\\\\", "\\", regex=False))
    out = cast(SeriesT, out.where(~quoted_mask, unquoted))

    if hasattr(raw, "where"):
        integer_values = cast(SeriesT, raw.where(integer_mask, "0").astype("int64"))
        float_values = cast(SeriesT, raw.where(float_mask, "0").astype("float64"))
        out = cast(SeriesT, out.where(~integer_mask, integer_values))
        out = cast(SeriesT, out.where(~float_mask, float_values))

    return out


def entity_property_series(series: SeriesT, prop: str) -> SeriesT:
    text = cast(SeriesT, series.astype(str))
    pattern = rf"(?:\{{|,\s*){re.escape(prop)}:\s*(?P<value>{_ENTITY_PROPERTY_VALUE_RE})(?=,\s*[A-Za-z_][A-Za-z0-9_]*:|\}})"
    raw = cast(SeriesT, text.str.extract(pattern, expand=False))
    return _coerce_property_value_series(raw)


def entity_property_scalar(value: Any, prop: str) -> Any:
    if not is_entity_text_scalar(value):
        return None
    text = str(value).strip()
    pattern = re.compile(
        rf"(?:\{{|,\s*){re.escape(prop)}:\s*(?P<value>{_ENTITY_PROPERTY_VALUE_RE})(?=,\s*[A-Za-z_][A-Za-z0-9_]*:|\}})"
    )
    match = pattern.search(text)
    if match is None:
        return None
    raw = match.group("value")
    if raw == "null":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    try:
        if any(token in raw for token in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except Exception:
        return raw
