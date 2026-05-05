"""Graphistry frontend React-settings contracts and introspection keyspace."""

from typing import Any, Dict, Set, Tuple
from typing_extensions import Literal, TypeAlias, TypedDict

from graphistry.models.surfaces.graphistry_frontend.settings_value import (
    SettingBool,
    SettingNumber,
    SettingString,
    SettingsValue,
)

ReactSettingsDict: TypeAlias = Dict[str, SettingsValue]
NeighborhoodHighlightSetting: TypeAlias = Literal["incoming", "outgoing", "both", "none", "node"]


class KnownReactSettingsDict(TypedDict, total=False):
    axes: SettingsValue
    backgroundColor: SettingString
    controls: SettingsValue
    edgeCurvature: SettingNumber
    edgeInfluence: SettingNumber
    edgeOpacity: SettingNumber
    encodeAxis: SettingsValue
    encodeEdgeColor: SettingsValue
    encodeEdgeIcons: SettingsValue
    encodePointColor: SettingsValue
    encodePointIcons: SettingsValue
    encodePointSize: SettingsValue
    exclusions: SettingsValue
    filters: SettingsValue
    gravity: SettingNumber
    dissuadeHubs: SettingBool
    iframeStyle: SettingsValue
    labelBackground: SettingString
    labelColor: SettingString
    labelOpacity: SettingNumber
    linLog: SettingBool
    lockedR: SettingBool
    lockedX: SettingBool
    lockedY: SettingBool
    neighborhoodHighlight: NeighborhoodHighlightSetting
    neighborhoodHighlightHops: SettingNumber
    play: int
    pointOpacity: SettingNumber
    pointSize: SettingNumber
    pointsOfInterestMax: SettingNumber
    precisionVsSpeed: SettingNumber
    pruneOrphans: SettingBool
    scalingRatio: SettingNumber
    showArrows: SettingBool
    showHistograms: SettingBool
    showInfo: SettingBool
    showInspector: SettingBool
    showLabelActions: SettingBool
    showLabelInspector: SettingBool
    showLabelOnHover: SettingBool
    showLabelPropertiesOnHover: SettingBool
    showLabels: SettingBool
    showMenu: SettingBool
    showPointsOfInterest: SettingBool
    showPointsOfInterestLabel: SettingBool
    showSplashScreen: SettingBool
    showTimebars: SettingBool
    showToolbar: SettingBool
    strongGravity: SettingBool
    ticks: SettingsValue
    type: SettingString
    workbook: SettingsValue


REACT_SETTING_NAMES: Tuple[str, ...] = (
    "axes", "backgroundColor", "controls", "edgeCurvature", "edgeInfluence",
    "edgeOpacity", "encodeAxis", "encodeEdgeColor", "encodeEdgeIcons",
    "encodePointColor", "encodePointIcons", "encodePointSize", "exclusions",
    "filters", "gravity", "dissuadeHubs", "iframeStyle", "labelBackground", "labelColor",
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
