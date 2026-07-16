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
        return (
            set(value).issubset({"kind", "text"})
            and isinstance(value.get("text"), str)
        )
    if kind == "search_any":
        if not set(value).issubset(
            {"kind", "term", "case_sensitive", "regex", "columns"}
        ):
            return False
        if not isinstance(value.get("term"), str):
            return False
        if "case_sensitive" in value and not isinstance(value["case_sensitive"], bool):
            return False
        if "regex" in value and not isinstance(value["regex"], bool):
            return False
        if "columns" in value and not _is_string_list(value["columns"]):
            return False
        return True
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
