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
