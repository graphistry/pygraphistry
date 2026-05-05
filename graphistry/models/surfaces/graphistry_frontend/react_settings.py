"""Graphistry frontend React-settings contracts and introspection keyspace."""

from typing import Any, Dict, Set, Tuple
from typing_extensions import TypeAlias, TypedDict

from graphistry.models.surfaces.graphistry_frontend.settings_value import SettingsValue

ReactSettingsDict: TypeAlias = Dict[str, SettingsValue]


class KnownReactSettingsDict(TypedDict, total=False):
    axes: SettingsValue
    backgroundColor: SettingsValue
    controls: SettingsValue
    edgeCurvature: SettingsValue
    edgeInfluence: SettingsValue
    edgeOpacity: SettingsValue
    encodeAxis: SettingsValue
    encodeEdgeColor: SettingsValue
    encodeEdgeIcons: SettingsValue
    encodePointColor: SettingsValue
    encodePointIcons: SettingsValue
    encodePointSize: SettingsValue
    exclusions: SettingsValue
    filters: SettingsValue
    gravity: SettingsValue
    iframeStyle: SettingsValue
    labelBackground: SettingsValue
    labelColor: SettingsValue
    labelOpacity: SettingsValue
    linLog: SettingsValue
    lockedR: SettingsValue
    lockedX: SettingsValue
    lockedY: SettingsValue
    neighborhoodHighlight: SettingsValue
    neighborhoodHighlightHops: SettingsValue
    play: SettingsValue
    pointOpacity: SettingsValue
    pointSize: SettingsValue
    pointsOfInterestMax: SettingsValue
    precisionVsSpeed: SettingsValue
    pruneOrphans: SettingsValue
    scalingRatio: SettingsValue
    showArrows: SettingsValue
    showHistograms: SettingsValue
    showInfo: SettingsValue
    showInspector: SettingsValue
    showLabelActions: SettingsValue
    showLabelInspector: SettingsValue
    showLabelOnHover: SettingsValue
    showLabelPropertiesOnHover: SettingsValue
    showLabels: SettingsValue
    showMenu: SettingsValue
    showPointsOfInterest: SettingsValue
    showPointsOfInterestLabel: SettingsValue
    showSplashScreen: SettingsValue
    showTimebars: SettingsValue
    showToolbar: SettingsValue
    strongGravity: SettingsValue
    ticks: SettingsValue
    type: SettingsValue
    workbook: SettingsValue


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

REACT_SETTING_NAME_SET = frozenset(REACT_SETTING_NAMES)
APPLY_ENCODINGS_REACT_KEYS: Tuple[str, ...] = (
    "encodePointColor",
    "encodeEdgeColor",
    "encodePointSize",
    "encodePointIcons",
    "encodePointIcon",
    "encodeEdgeIcons",
    "encodeEdgeIcon",
    "encodeAxis",
)
APPLY_ENCODINGS_REACT_KEY_SET = frozenset(APPLY_ENCODINGS_REACT_KEYS)


def react_setting_keys() -> Tuple[str, ...]:
    return REACT_SETTING_NAMES


def apply_encodings_keys() -> Tuple[str, ...]:
    return APPLY_ENCODINGS_REACT_KEYS


def _typed_dict_keys(typed_dict_cls: Any) -> Set[str]:
    raw = getattr(typed_dict_cls, "__annotations__", {})
    if not isinstance(raw, dict):
        return set()
    return {str(k) for k in raw.keys()}


if _typed_dict_keys(KnownReactSettingsDict) != set(REACT_SETTING_NAMES):
    raise ValueError("KnownReactSettingsDict keys must match REACT_SETTING_NAMES")
