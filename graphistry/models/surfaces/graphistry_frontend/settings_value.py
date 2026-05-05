"""Recursive JSON-like value type used by Graphistry frontend settings contracts."""

from typing import Dict, List, Union
from typing_extensions import TypeAlias

SettingString: TypeAlias = str
SettingBool: TypeAlias = bool
SettingNumber: TypeAlias = Union[int, float]

# JSON-like settings payload value shared across surface contracts.
SettingsValue: TypeAlias = Union[
    None,
    SettingString,
    SettingNumber,
    SettingBool,
    List["SettingsValue"],
    Dict[str, "SettingsValue"],
]
