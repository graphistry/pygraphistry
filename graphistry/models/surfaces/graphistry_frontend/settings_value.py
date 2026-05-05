"""Recursive JSON-like value type used by Graphistry frontend settings contracts."""

from typing import Dict, List, Union
from typing_extensions import TypeAlias

# JSON-like settings payload value shared across surface contracts.
SettingsValue: TypeAlias = Union[
    None,
    str,
    int,
    float,
    bool,
    List["SettingsValue"],
    Dict[str, "SettingsValue"],
]
