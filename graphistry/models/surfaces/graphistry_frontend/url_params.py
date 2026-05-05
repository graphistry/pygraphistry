"""Graphistry frontend URL-parameter contracts and axis-layout defaults."""

from typing import Any, Dict, List, Set, Tuple, Union
from typing_extensions import Literal, TypeAlias, TypedDict

from graphistry.models.surfaces.graphistry_frontend.axis import AxisKind
from graphistry.models.surfaces.graphistry_frontend.settings_value import (
    SettingBool,
    SettingNumber,
    SettingString,
    SettingsValue,
)

URLParamsDict: TypeAlias = Dict[str, SettingsValue]
CollectionsSetting: TypeAlias = Union[SettingString, Dict[str, Any], List[Any]]
LogoPositionSetting: TypeAlias = Literal["top", "bottom"]


class KnownURLParamsDict(TypedDict, total=False):
    bg: SettingString
    bottom: SettingNumber
    collections: CollectionsSetting
    collectionsGlobalEdgeColor: SettingString
    collectionsGlobalNodeColor: SettingString
    dissuadeHubs: SettingBool
    edgeCurvature: SettingNumber
    edgeInfluence: SettingNumber
    edgeOpacity: SettingNumber
    favicon: SettingString
    gravity: SettingNumber
    info: SettingBool
    labelBackground: SettingString
    labelColor: SettingString
    labelOpacity: SettingNumber
    left: SettingNumber
    linLog: SettingBool
    lockedR: SettingBool
    lockedX: SettingBool
    lockedY: SettingBool
    logoAutoInvert: SettingBool
    logoMaxHeight: int
    logoMaxWidth: int
    logoPosition: LogoPositionSetting
    logoUrl: SettingString
    menu: SettingBool
    neighborhoodHighlight: SettingBool
    neighborhoodHighlightHops: int
    pageTitle: SettingString
    play: int
    pointOpacity: SettingNumber
    pointSize: SettingNumber
    pointStrokeWidth: SettingNumber
    pointsOfInterestMax: int
    precisionVsSpeed: SettingNumber
    right: SettingNumber
    scalingRatio: SettingNumber
    shortenLabels: SettingBool
    showActions: SettingBool
    showArrows: SettingBool
    showCollections: SettingBool
    showHistograms: SettingBool
    showInspector: SettingBool
    showLabelOnHover: SettingBool
    showLabelPropertiesOnHover: SettingBool
    showLabels: SettingBool
    showPointsOfInterest: SettingBool
    showPointsOfInterestLabel: SettingBool
    splashAfter: SettingBool
    strongGravity: SettingBool
    top: SettingNumber


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

URL_PARAM_NAME_SET = frozenset(URL_PARAM_NAMES)

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


def axis_url_defaults(kind: AxisKind) -> Dict[str, SettingsValue]:
    if kind == "radial":
        return dict(RADIAL_AXIS_URL_DEFAULTS)
    return dict(LINEAR_AXIS_URL_DEFAULTS)


def url_param_keys() -> Tuple[str, ...]:
    return URL_PARAM_NAMES


def _typed_dict_keys(typed_dict_cls: Any) -> Set[str]:
    raw = getattr(typed_dict_cls, "__annotations__", {})
    if not isinstance(raw, dict):
        return set()
    return {str(k) for k in raw.keys()}


if _typed_dict_keys(KnownURLParamsDict) != set(URL_PARAM_NAMES):
    raise ValueError("KnownURLParamsDict keys must match URL_PARAM_NAMES")
