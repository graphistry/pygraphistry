"""Typed row-pipeline alias prefilter hints."""
from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Sequence, TypedDict
from typing_extensions import TypeGuard


class AliasPrefilterSpec(TypedDict, total=False):
    kind: Literal["expr", "search_any"]
    text: str
    term: str
    case_sensitive: bool
    regex: bool
    columns: List[str]


AliasPrefilters = Mapping[str, Sequence[AliasPrefilterSpec]]
MutableAliasPrefilters = Dict[str, List[AliasPrefilterSpec]]


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_alias_prefilter_spec(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("kind")
    if kind == "expr":
        return isinstance(value.get("text"), str)
    if kind == "search_any":
        columns = value.get("columns")
        return isinstance(value.get("term"), str) and (columns is None or _is_string_list(columns))
    return False


def is_alias_prefilters(value: object) -> TypeGuard[MutableAliasPrefilters]:
    if not isinstance(value, dict):
        return False
    for alias, specs in value.items():
        if not isinstance(alias, str) or not isinstance(specs, list):
            return False
        if not all(_is_alias_prefilter_spec(spec) for spec in specs):
            return False
    return True
