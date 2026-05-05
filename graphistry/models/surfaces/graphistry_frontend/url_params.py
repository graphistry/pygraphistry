"""Graphistry frontend URL-parameter contracts and axis-layout defaults."""

from typing import Any, Dict, Set, Tuple
from typing_extensions import TypeAlias, TypedDict

from graphistry.models.surfaces.graphistry_frontend.axis import AxisKind
from graphistry.models.surfaces.graphistry_frontend.settings_value import SettingsValue

URLParamsDict: TypeAlias = Dict[str, SettingsValue]


class KnownURLParamsDict(TypedDict, total=False):
    bg: SettingsValue
    bottom: SettingsValue
    collections: SettingsValue
    collectionsGlobalEdgeColor: SettingsValue
    collectionsGlobalNodeColor: SettingsValue
    dissuadeHubs: SettingsValue
    edgeCurvature: SettingsValue
    edgeInfluence: SettingsValue
    edgeOpacity: SettingsValue
    favicon: SettingsValue
    gravity: SettingsValue
    info: SettingsValue
    labelBackground: SettingsValue
    labelColor: SettingsValue
    labelOpacity: SettingsValue
    left: SettingsValue
    linLog: SettingsValue
    lockedR: SettingsValue
    lockedX: SettingsValue
    lockedY: SettingsValue
    logoAutoInvert: SettingsValue
    logoMaxHeight: SettingsValue
    logoMaxWidth: SettingsValue
    logoPosition: SettingsValue
    logoUrl: SettingsValue
    menu: SettingsValue
    neighborhoodHighlight: SettingsValue
    neighborhoodHighlightHops: SettingsValue
    pageTitle: SettingsValue
    play: SettingsValue
    pointOpacity: SettingsValue
    pointSize: SettingsValue
    pointStrokeWidth: SettingsValue
    pointsOfInterestMax: SettingsValue
    precisionVsSpeed: SettingsValue
    right: SettingsValue
    scalingRatio: SettingsValue
    shortenLabels: SettingsValue
    showActions: SettingsValue
    showArrows: SettingsValue
    showCollections: SettingsValue
    showHistograms: SettingsValue
    showInspector: SettingsValue
    showLabelOnHover: SettingsValue
    showLabelPropertiesOnHover: SettingsValue
    showLabels: SettingsValue
    showPointsOfInterest: SettingsValue
    showPointsOfInterestLabel: SettingsValue
    splashAfter: SettingsValue
    strongGravity: SettingsValue
    top: SettingsValue


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
