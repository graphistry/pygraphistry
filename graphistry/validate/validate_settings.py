from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Literal, TypeAlias, TypedDict

from graphistry.models.types import ValidationMode, ValidationParam
from graphistry.util import warn as emit_warn

SettingsValue: TypeAlias = Union[None, str, int, float, bool, List[Any], Dict[str, Any]]
AxisKind: TypeAlias = Literal["radial", "linear"]
URLParamsDict: TypeAlias = Dict[str, SettingsValue]
ReactSettingsDict: TypeAlias = Dict[str, SettingsValue]


class AxisBounds(TypedDict, total=False):
    min: Union[int, float]
    max: Union[int, float]


class AxisRow(TypedDict, total=False):
    label: str
    r: Union[int, float]
    x: Union[int, float]
    y: Union[int, float]
    internal: bool
    external: bool
    space: bool
    width: Union[int, float]
    bounds: AxisBounds


AxisRows: TypeAlias = List[AxisRow]
RingCategoricalAxisLabelMap: TypeAlias = Dict[Any, str]
RingContinuousAxisLabelMap: TypeAlias = Dict[Union[int, float], str]
RingContinuousAxis: TypeAlias = Union[RingContinuousAxisLabelMap, List[str], AxisRows]
RingCategoricalAxis: TypeAlias = Union[RingCategoricalAxisLabelMap, AxisRows]

AXIS_BOUNDS_ALLOWED_KEYS: Tuple[str, ...] = ("min", "max")
AXIS_ROW_ALLOWED_KEYS: Tuple[str, ...] = (
    "label", "r", "x", "y", "internal", "external", "space", "width", "bounds",
)
AXIS_ROW_POSITION_KEYS: Tuple[str, ...] = ("r", "x", "y")
AXIS_ROW_BOOL_KEYS: Tuple[str, ...] = ("internal", "external", "space")
AXIS_ROW_NUMERIC_KEYS: Tuple[str, ...] = ("r", "x", "y", "width")

# Canonical settings key exports for downstream integrations
URL_PARAM_NAMES: Tuple[str, ...] = (
    "bg", "bottom", "collections", "collectionsGlobalEdgeColor",
    "collectionsGlobalNodeColor", "dissuadeHubs", "edgeCurvature",
    "edgeInfluence", "edgeOpacity", "favicon", "gravity", "info",
    "labelBackground", "labelColor", "labelOpacity", "left", "linLog",
    "lockedR", "lockedX", "lockedY", "logoAutoInvert", "logoMaxHeight",
    "logoMaxWidth", "logoPosition", "logoUrl", "menu",
    "neighborhoodHighlight", "neighborhoodHighlightHops", "pageTitle",
    "play", "pointOpacity", "pointSize", "pointStrokeWidth",
    "pointsOfInterestMax", "precisionVsSpeed", "right", "scalingRatio",
    "shortenLabels", "showActions", "showArrows", "showCollections",
    "showHistograms", "showInspector", "showLabelOnHover",
    "showLabelPropertiesOnHover", "showLabels", "showPointsOfInterest",
    "showPointsOfInterestLabel", "splashAfter", "strongGravity", "top",
)

REACT_SETTING_NAMES: Tuple[str, ...] = (
    "axes", "backgroundColor", "controls", "edgeCurvature", "edgeInfluence",
    "edgeOpacity", "encodeAxis", "encodeEdgeColor", "encodeEdgeIcons",
    "encodePointColor", "encodePointIcons", "encodePointSize", "exclusions",
    "filters", "gravity", "iframeStyle", "labelBackground", "labelColor",
    "labelOpacity", "linLog", "lockedR", "lockedX", "lockedY",
    "neighborhoodHighlight", "neighborhoodHighlightHops", "play",
    "pointOpacity", "pointSize", "pointsOfInterestMax", "precisionVsSpeed",
    "pruneOrphans", "scalingRatio", "showArrows", "showHistograms",
    "showInfo", "showInspector", "showLabelActions", "showLabelInspector",
    "showLabelOnHover", "showLabelPropertiesOnHover", "showLabels",
    "showMenu", "showPointsOfInterest", "showPointsOfInterestLabel",
    "showSplashScreen", "showTimebars", "showToolbar", "strongGravity",
    "ticks", "type", "workbook",
)

URL_PARAM_NAME_SET = frozenset(URL_PARAM_NAMES)
REACT_SETTING_NAME_SET = frozenset(REACT_SETTING_NAMES)

RADIAL_AXIS_URL_DEFAULTS: Dict[str, SettingsValue] = {
    "play": 0,
    "lockedR": True,
    "splashAfter": False,
}
LINEAR_AXIS_URL_DEFAULTS: Dict[str, SettingsValue] = {
    "play": 0,
    "lockedX": True,
    "lockedY": True,
    "splashAfter": False,
}


def _is_non_bool_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_axis_bounds_payload(v: object) -> bool:
    if not isinstance(v, dict):
        return False
    if any(k not in AXIS_BOUNDS_ALLOWED_KEYS for k in v.keys()):
        return False
    if "min" in v and not _is_non_bool_number(v["min"]):
        return False
    if "max" in v and not _is_non_bool_number(v["max"]):
        return False
    return True


def is_axis_row_payload(v: object) -> bool:
    if not isinstance(v, dict):
        return False
    if any(k not in AXIS_ROW_ALLOWED_KEYS for k in v.keys()):
        return False
    if "label" in v and not isinstance(v["label"], str):
        return False
    for k in AXIS_ROW_NUMERIC_KEYS:
        if k in v and not _is_non_bool_number(v[k]):
            return False
    for k in AXIS_ROW_BOOL_KEYS:
        if k in v and not isinstance(v[k], bool):
            return False
    if "bounds" in v and not is_axis_bounds_payload(v["bounds"]):
        return False
    if all(k not in v for k in AXIS_ROW_POSITION_KEYS):
        return False
    return True


def is_axis_rows_payload(v: object) -> bool:
    return isinstance(v, list) and all(is_axis_row_payload(item) for item in v)


def is_ring_continuous_axis_payload(v: object) -> bool:
    if is_axis_rows_payload(v):
        return True
    if isinstance(v, list):
        return all(isinstance(item, str) for item in v)
    if isinstance(v, dict):
        return all(_is_non_bool_number(k) and isinstance(val, str) for k, val in v.items())
    return False


def is_ring_categorical_axis_payload(v: object) -> bool:
    if is_axis_rows_payload(v):
        return True
    return isinstance(v, dict) and all(isinstance(val, str) for val in v.values())


def normalize_validation_params(
    validate: ValidationParam = "autofix",
    warn: bool = True
) -> Tuple[ValidationMode, bool]:
    if validate is True:
        validate_mode: ValidationMode = "strict"
    elif validate is False:
        validate_mode = "autofix"
        warn = False
    else:
        validate_mode = validate
    return validate_mode, warn


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


def axis_url_defaults(kind: AxisKind) -> Dict[str, SettingsValue]:
    if kind == "radial":
        return dict(RADIAL_AXIS_URL_DEFAULTS)
    return dict(LINEAR_AXIS_URL_DEFAULTS)


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
