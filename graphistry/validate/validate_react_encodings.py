from typing import Any, Dict, List, Optional, Tuple, Union, cast
from typing_extensions import Literal, TypedDict

from graphistry.models.surfaces.graphistry_axis import AxisRows
from graphistry.models.surfaces.graphistry_react import (
    APPLY_ENCODINGS_REACT_KEY_SET,
    REACT_SETTING_NAME_SET,
    ReactSettingsDict,
)
from graphistry.models.types import ValidationMode, ValidationParam
from graphistry.util import warn as emit_warn
from graphistry.validate.common import normalize_validation_params

EncodingVariation = Literal["categorical", "continuous"]
ColorEncodingKey = Literal["encodePointColor", "encodeEdgeColor"]
SizeEncodingKey = Literal["encodePointSize"]
IconEncodingKey = Literal["encodePointIcons", "encodeEdgeIcons"]
AxisEncodingKey = Literal["encodeAxis"]
EncodingOperationKind = Literal["color", "size", "icon", "axis"]


class ColorEncodingOp(TypedDict, total=False):
    kind: Literal["color"]
    key: ColorEncodingKey
    column: str
    variation: EncodingVariation
    categorical_mapping: Dict[Any, Any]
    palette: List[Any]


class SizeEncodingOp(TypedDict, total=False):
    kind: Literal["size"]
    key: SizeEncodingKey
    column: str
    categorical_mapping: Dict[Any, Any]
    default_mapping: Any


class IconEncodingOp(TypedDict, total=False):
    kind: Literal["icon"]
    key: IconEncodingKey
    column: str
    categorical_mapping: Dict[Any, Any]
    continuous_binning: List[Any]
    default_mapping: Any


class AxisEncodingOp(TypedDict):
    kind: Literal["axis"]
    key: AxisEncodingKey
    rows: AxisRows


ReactEncodingOp = Union[ColorEncodingOp, SizeEncodingOp, IconEncodingOp, AxisEncodingOp]


def _issue(
    message: str,
    data: Optional[Dict[str, Any]],
    validate_mode: ValidationMode,
    warn: bool,
) -> None:
    error = ValueError({"message": message, "data": data} if data else {"message": message})
    if validate_mode in ("strict", "strict-fast"):
        raise error
    if warn and validate_mode == "autofix":
        emit_warn("apply_encodings autofix: {} ({})".format(message, data))


def _expect_list_payload(
    raw_value: Any,
    key: str,
    validate_mode: ValidationMode,
    warn: bool,
) -> Optional[List[Any]]:
    if not isinstance(raw_value, list):
        _issue("React encoding payload must be a list", {"key": key, "type": type(raw_value).__name__}, validate_mode, warn)
        return None
    return raw_value


def _expect_column(
    raw_value: Any,
    key: str,
    validate_mode: ValidationMode,
    warn: bool,
) -> Optional[str]:
    if not isinstance(raw_value, str) or raw_value.strip() == "":
        _issue("React encoding column must be a non-empty string", {"key": key, "value": raw_value}, validate_mode, warn)
        return None
    return raw_value


def _parse_color_payload(
    key: ColorEncodingKey,
    raw_value: Any,
    validate_mode: ValidationMode,
    warn: bool,
) -> Optional[ColorEncodingOp]:
    payload = _expect_list_payload(raw_value, key, validate_mode, warn)
    if payload is None or len(payload) == 0:
        _issue("React color encoding payload must include at least [column]", {"key": key}, validate_mode, warn)
        return None
    if len(payload) > 3:
        _issue("React color encoding payload supports at most 3 elements", {"key": key, "len": len(payload)}, validate_mode, warn)
        return None
    column = _expect_column(payload[0], key, validate_mode, warn)
    if column is None:
        return None

    op: ColorEncodingOp = {"kind": "color", "key": key, "column": column}
    if len(payload) == 1:
        return op

    second = payload[1]
    third = payload[2] if len(payload) == 3 else None

    mapping_or_palette: Any = None
    if isinstance(second, str) and second in ("categorical", "continuous"):
        op["variation"] = cast(EncodingVariation, second)
        mapping_or_palette = third
    else:
        mapping_or_palette = second

    if mapping_or_palette is None:
        return op
    if isinstance(mapping_or_palette, dict):
        if op.get("variation") == "continuous":
            _issue("Continuous color payload must use a palette list, not dict mapping", {"key": key}, validate_mode, warn)
            return None
        op["categorical_mapping"] = mapping_or_palette
        return op
    if isinstance(mapping_or_palette, list):
        op["palette"] = mapping_or_palette
        return op
    _issue(
        "React color payload mapping/palette must be dict or list",
        {"key": key, "type": type(mapping_or_palette).__name__},
        validate_mode,
        warn,
    )
    return None


def _parse_size_payload(
    raw_value: Any,
    validate_mode: ValidationMode,
    warn: bool,
) -> Optional[SizeEncodingOp]:
    key: SizeEncodingKey = "encodePointSize"
    payload = _expect_list_payload(raw_value, key, validate_mode, warn)
    if payload is None or len(payload) == 0:
        _issue("React size encoding payload must include at least [column]", {"key": key}, validate_mode, warn)
        return None
    if len(payload) > 3:
        _issue("React size encoding payload supports at most 3 elements", {"key": key, "len": len(payload)}, validate_mode, warn)
        return None
    column = _expect_column(payload[0], key, validate_mode, warn)
    if column is None:
        return None

    op: SizeEncodingOp = {"kind": "size", "key": key, "column": column}
    if len(payload) >= 2:
        mapping = payload[1]
        if mapping is None:
            pass
        elif isinstance(mapping, dict):
            op["categorical_mapping"] = mapping
        else:
            _issue("React size mapping must be a dict", {"key": key, "type": type(mapping).__name__}, validate_mode, warn)
            return None
    if len(payload) == 3:
        op["default_mapping"] = payload[2]
    return op


def _parse_icon_payload(
    key: IconEncodingKey,
    raw_value: Any,
    validate_mode: ValidationMode,
    warn: bool,
) -> Optional[IconEncodingOp]:
    payload = _expect_list_payload(raw_value, key, validate_mode, warn)
    if payload is None or len(payload) == 0:
        _issue("React icon encoding payload must include at least [column]", {"key": key}, validate_mode, warn)
        return None
    if len(payload) > 3:
        _issue("React icon encoding payload supports at most 3 elements", {"key": key, "len": len(payload)}, validate_mode, warn)
        return None
    column = _expect_column(payload[0], key, validate_mode, warn)
    if column is None:
        return None

    op: IconEncodingOp = {"kind": "icon", "key": key, "column": column}
    if len(payload) >= 2:
        mapping = payload[1]
        if mapping is None:
            pass
        elif isinstance(mapping, dict):
            op["categorical_mapping"] = mapping
        elif isinstance(mapping, list):
            op["continuous_binning"] = mapping
        else:
            _issue("React icon mapping must be dict or list", {"key": key, "type": type(mapping).__name__}, validate_mode, warn)
            return None
    if len(payload) == 3:
        op["default_mapping"] = payload[2]
    return op


def _normalize_icon_key(key: str) -> Optional[IconEncodingKey]:
    if key in ("encodePointIcons", "encodePointIcon"):
        return "encodePointIcons"
    if key in ("encodeEdgeIcons", "encodeEdgeIcon"):
        return "encodeEdgeIcons"
    return None


def parse_apply_encodings_ops(
    react_encodings: Optional[ReactSettingsDict],
    validate: ValidationParam = "strict",
    warn: bool = True,
) -> List[ReactEncodingOp]:
    validate_mode, warn = normalize_validation_params(validate, warn)
    out: List[ReactEncodingOp] = []
    if react_encodings is None:
        return out
    if not isinstance(react_encodings, dict):
        _issue("react_encodings must be a dict", {"type": type(react_encodings).__name__}, validate_mode, warn)
        return out

    for key, value in react_encodings.items():
        if key not in APPLY_ENCODINGS_REACT_KEY_SET:
            if key in REACT_SETTING_NAME_SET:
                _issue("React setting key is valid but not handled by apply_encodings()", {"key": key}, validate_mode, warn)
            else:
                _issue("Unknown React setting key", {"key": key}, validate_mode, warn)
            continue

        if key in ("encodePointColor", "encodeEdgeColor"):
            color_op = _parse_color_payload(cast(ColorEncodingKey, key), value, validate_mode, warn)
            if color_op is not None:
                out.append(color_op)
            continue

        if key == "encodePointSize":
            size_op = _parse_size_payload(value, validate_mode, warn)
            if size_op is not None:
                out.append(size_op)
            continue

        icon_key = _normalize_icon_key(key)
        if icon_key is not None:
            icon_op = _parse_icon_payload(icon_key, value, validate_mode, warn)
            if icon_op is not None:
                out.append(icon_op)
            continue

        axis_rows = _expect_list_payload(value, key, validate_mode, warn)
        if axis_rows is None or any(not isinstance(row, dict) for row in axis_rows):
            _issue("encodeAxis must be a list of axis row objects", {"type": type(value).__name__}, validate_mode, warn)
            continue
        out.append({"kind": "axis", "key": "encodeAxis", "rows": axis_rows})

    return out
