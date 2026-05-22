from typing import Any, Dict, Iterable, List, Optional

from graphistry.models.surfaces.graphistry_frontend.axis import AxisKind
from graphistry.models.surfaces.graphistry_frontend.react_settings import (
    REACT_SETTING_NAMES,
    ReactSettingsDict,
)
from graphistry.models.surfaces.graphistry_frontend.url_params import (
    URL_PARAM_NAMES,
    URLParamsDict,
    axis_url_defaults,
)
from graphistry.models.surfaces.graphistry_frontend.settings_value import SettingsValue
from graphistry.models.types import ValidationMode, ValidationParam
from graphistry.util import warn as emit_warn
from graphistry.validate.common import normalize_validation_params


def _issue(
    message: str,
    data: Optional[Dict[str, Any]],
    validate_mode: ValidationMode,
    warn: bool
) -> None:
    error = ValueError({"message": message, "data": data} if data else {"message": message})
    if validate_mode in ("strict", "strict-fast"):
        raise error
    if warn and validate_mode == "autofix":
        emit_warn(f"Settings validation warning: {message} ({data})")


def _is_settings_value(v: Any) -> bool:
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, list):
        return all(_is_settings_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_settings_value(val) for k, val in v.items())
    return False


def _normalize_settings(
    settings: Optional[Dict[str, Any]],
    allowed_keys: Iterable[str],
    validate: ValidationParam = "autofix",
    warn: bool = True,
    label: str = "settings",
) -> Dict[str, SettingsValue]:
    validate_mode, warn = normalize_validation_params(validate, warn)
    out: Dict[str, SettingsValue] = {}
    allowed = frozenset(allowed_keys)
    if settings is None:
        return out
    if not isinstance(settings, dict):
        _issue(
            f"{label} must be a dict",
            {"type": type(settings).__name__},
            validate_mode,
            warn,
        )
        return out
    for k, v in settings.items():
        if k not in allowed:
            _issue(
                f"Unknown {label} key",
                {"key": k},
                validate_mode,
                warn,
            )
            continue
        if not _is_settings_value(v):
            _issue(
                f"Invalid {label} value type",
                {"key": k, "type": type(v).__name__},
                validate_mode,
                warn,
            )
            continue
        out[k] = v
    return out


def normalize_url_params(
    url_params: Optional[Dict[str, Any]],
    validate: ValidationParam = "autofix",
    warn: bool = True,
) -> URLParamsDict:
    return _normalize_settings(
        settings=url_params,
        allowed_keys=URL_PARAM_NAMES,
        validate=validate,
        warn=warn,
        label="url_params",
    )


def normalize_react_settings(
    react_settings: Optional[Dict[str, Any]],
    validate: ValidationParam = "autofix",
    warn: bool = True,
) -> ReactSettingsDict:
    return _normalize_settings(
        settings=react_settings,
        allowed_keys=REACT_SETTING_NAMES,
        validate=validate,
        warn=warn,
        label="react_settings",
    )


def _extract_axis_rows(complex_encodings: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(complex_encodings, dict):
        return None
    node_encodings = complex_encodings.get("node_encodings")
    if not isinstance(node_encodings, dict):
        return None
    for mode in ("default", "current"):
        scoped = node_encodings.get(mode)
        if not isinstance(scoped, dict):
            continue
        axis_encoding = scoped.get("pointAxisEncoding")
        if not isinstance(axis_encoding, dict):
            continue
        rows = axis_encoding.get("rows")
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return rows
    return None


def classify_axis_kind(complex_encodings: Any) -> Optional[AxisKind]:
    rows = _extract_axis_rows(complex_encodings)
    if not rows:
        return None
    for row in rows:
        has_r = "r" in row
        has_y = "y" in row
        if has_r and not has_y:
            return "radial"
        if has_y and not has_r:
            return "linear"
    return None


def apply_axis_url_defaults(
    url_params: Optional[Dict[str, Any]],
    complex_encodings: Any,
) -> Optional[Dict[str, SettingsValue]]:
    kind = classify_axis_kind(complex_encodings)
    if kind is None:
        return url_params if isinstance(url_params, dict) else url_params
    defaults = axis_url_defaults(kind)
    current: Dict[str, SettingsValue] = {}
    if isinstance(url_params, dict):
        current = {k: v for k, v in url_params.items() if _is_settings_value(v)}
    return {**defaults, **current}
