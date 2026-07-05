import re as _re
from typing import Any, Callable, Dict

from graphistry.compute.gfql.language_defs import GFQL_COMPARISON_BINARY_OPS, GFQL_GROUPBY_AGG_METHODS

_GFQL_STRING_PREDICATE_SCALAR_OPS: Dict[str, Callable[[str, str], bool]] = {
    "contains": lambda left_txt, right_txt: right_txt in left_txt,
    "starts_with": lambda left_txt, right_txt: left_txt.startswith(right_txt),
    "ends_with": lambda left_txt, right_txt: left_txt.endswith(right_txt),
    # openCypher/neo4j `=~`: Java-regex, FULL/anchored match (inline flags like `(?i)` honored).
    "regex": lambda left_txt, right_txt: _re.fullmatch(right_txt, left_txt) is not None,
}

def _series_regex_fullmatch(left_txt: Any, needle: str) -> Any:
    # `=~` on a Series: delegate to the Fullmatch predicate, which carries the engine
    # workarounds (cuDF has no ``.str.fullmatch`` — raw use raised on engine='cudf'
    # while pandas worked; anchored-match emulation + inline-flag translation live
    # there). Local import: predicates.str imports compute modules at module scope.
    from graphistry.compute.predicates.str import Fullmatch
    return Fullmatch(needle)(left_txt)


_GFQL_STRING_PREDICATE_SERIES_OPS: Dict[str, Callable[[Any, str], Any]] = {
    "contains": lambda left_txt, needle: left_txt.str.contains(needle, regex=False),
    "starts_with": lambda left_txt, needle: left_txt.str.startswith(needle),
    "ends_with": lambda left_txt, needle: left_txt.str.endswith(needle),
    "regex": _series_regex_fullmatch,
}

_GFQL_SEQUENCE_FN_SERIES_OPS: Dict[str, Callable[[Any], Any]] = {
    "head": lambda series_value: series_value.str.get(0),
    "tail": lambda series_value: series_value.str.slice(start=1),
    "reverse": lambda series_value: series_value.str[::-1],
}

_GFQL_SEQUENCE_FN_SCALAR_OPS: Dict[str, Callable[[Any], Any]] = {
    "head": lambda value: value[0] if len(value) > 0 else None,
    "tail": lambda value: value[1:],
}


def apply_string_predicate_scalar(left_txt: str, right_txt: str, op_name: str) -> bool:
    fn = _GFQL_STRING_PREDICATE_SCALAR_OPS.get(op_name)
    if fn is None:
        raise ValueError(f"unsupported row expression predicate op: {op_name}")
    return fn(left_txt, right_txt)


def apply_string_predicate_series(left_txt: Any, needle: str, op_name: str) -> Any:
    fn = _GFQL_STRING_PREDICATE_SERIES_OPS.get(op_name)
    if fn is None:
        raise ValueError(f"unsupported row expression predicate op: {op_name}")
    return fn(left_txt, needle)


def eval_sequence_fn_series(series_value: Any, fn_name: str) -> Any:
    fn = _GFQL_SEQUENCE_FN_SERIES_OPS.get(fn_name)
    if fn is None:
        raise ValueError(f"unsupported row expression function: {fn_name}")
    return fn(series_value)


def eval_sequence_fn_scalar(value: Any, fn_name: str) -> Any:
    if fn_name == "reverse":
        if isinstance(value, str):
            return value[::-1]
        if isinstance(value, (list, tuple)):
            return list(reversed(value))
        raise ValueError(f"unsupported row expression function: {fn_name}")
    fn = _GFQL_SEQUENCE_FN_SCALAR_OPS.get(fn_name)
    if fn is None:
        raise ValueError(f"unsupported row expression function: {fn_name}")
    return fn(value)
