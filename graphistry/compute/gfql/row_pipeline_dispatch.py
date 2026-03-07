import operator
from typing import Any, Callable, Dict


GFQL_COMPARISON_BINARY_OPS: Dict[str, Callable[[Any, Any], Any]] = {
    "=": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

GFQL_GROUPBY_AGG_METHODS: Dict[str, str] = {
    "count": "count",
    "count_distinct": "nunique",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "avg": "mean",
    "mean": "mean",
}

_GFQL_STRING_PREDICATE_SCALAR_OPS: Dict[str, Callable[[str, str], bool]] = {
    "contains": lambda left_txt, right_txt: right_txt in left_txt,
    "starts_with": lambda left_txt, right_txt: left_txt.startswith(right_txt),
    "ends_with": lambda left_txt, right_txt: left_txt.endswith(right_txt),
}

_GFQL_STRING_PREDICATE_SERIES_OPS: Dict[str, Callable[[Any, str], Any]] = {
    "contains": lambda left_txt, needle: left_txt.str.contains(needle, regex=False),
    "starts_with": lambda left_txt, needle: left_txt.str.startswith(needle),
    "ends_with": lambda left_txt, needle: left_txt.str.endswith(needle),
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
