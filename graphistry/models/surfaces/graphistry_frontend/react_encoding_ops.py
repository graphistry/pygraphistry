"""Parsed React encoding operation contracts used by apply_encodings workflows."""

from typing import Any, Dict, List, Union
from typing_extensions import Literal, TypeAlias, TypedDict

from graphistry.models.surfaces.graphistry_frontend.axis import AxisRows
from graphistry.models.surfaces.graphistry_frontend.react_settings import (
    ReactColorEncodingKey,
    ReactEncodingVariation,
    ReactIconEncodingKey,
    ReactSizeEncodingKey,
)

AxisEncodingKey: TypeAlias = Literal["encodeAxis"]
EncodingOperationKind: TypeAlias = Literal["color", "size", "icon", "axis"]


class ColorEncodingOp(TypedDict, total=False):
    kind: Literal["color"]
    key: ReactColorEncodingKey
    column: str
    variation: ReactEncodingVariation
    categorical_mapping: Dict[Any, Any]
    palette: List[Any]


class SizeEncodingOp(TypedDict, total=False):
    kind: Literal["size"]
    key: ReactSizeEncodingKey
    column: str
    categorical_mapping: Dict[Any, Any]
    default_mapping: Any


class IconEncodingOp(TypedDict, total=False):
    kind: Literal["icon"]
    key: ReactIconEncodingKey
    column: str
    categorical_mapping: Dict[Any, Any]
    continuous_binning: List[Any]
    default_mapping: Any


class AxisEncodingOp(TypedDict):
    kind: Literal["axis"]
    key: AxisEncodingKey
    rows: AxisRows


ReactEncodingOp = Union[ColorEncodingOp, SizeEncodingOp, IconEncodingOp, AxisEncodingOp]

