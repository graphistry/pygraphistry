"""Graphistry frontend axis payload contracts shared by React encodeAxis and ring layout helpers."""

from typing import Any, Dict, List, Tuple, Union
from typing_extensions import Literal, TypeAlias, TypedDict

AxisKind: TypeAlias = Literal["radial", "linear"]


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
