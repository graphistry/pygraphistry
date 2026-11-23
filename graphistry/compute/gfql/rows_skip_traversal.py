"""Skip the chain-boundary traversal that a bindings ``rows()`` call never reads.

The bindings builders on every engine rebuild their rows from the base graph, the carried
start nodes and the alias prefilters. When the leading suffix call is such a ``rows()``,
the canonical middle traversal only contributes its output schema. The suffix runs on a
zero-row twin that has that schema, and any error replays the canonical path so errors
stay exactly as they were.
"""
from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type

from graphistry.Engine import is_polars_df
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject, serialize_binding_ops
from graphistry.compute.predicates.comparison import ComparisonPredicate
from graphistry.compute.predicates.numeric import NumericASTPredicate
from graphistry.compute.gfql.cache_registry import register_clearable_callable
from graphistry.compute.gfql.index.types import TraceEngine
from graphistry.compute.typing import DataFrameT
from graphistry.utils.json import JSONVal

FrameSchema = Tuple[Tuple[Tuple[type, object], object], ...]

_FILTER_FIELDS = ("filter_dict", "edge_match", "source_node_match", "destination_node_match")
_TWINS: "OrderedDict[Tuple[object, ...], Tuple[DataFrameT, DataFrameT]]" = OrderedDict()
_TWINS_MAX = 256
_TWINS_LOCK = threading.Lock()


def _clear_twins() -> None:
    with _TWINS_LOCK:
        _TWINS.clear()


register_clearable_callable("_TWINS", _clear_twins)


def bindings_rows_call(middle: List[ASTObject], suffix: List[ASTObject]) -> Optional[ASTCall]:
    """The leading ``rows()`` call when it rebuilds every binding of ``middle`` from the base graph.

    That holds when the call already carries exactly these binding ops, or when the
    named-middle rewrite installs them. A rows call with no binding ops over an
    UNNAMED middle instead reads the traversal-narrowed node table, and a ``source``,
    ``alias_endpoints`` or non-default ``table`` reads something other than bindings.
    ``table`` is compared against "nodes", not None: ``rows()`` always emits it.
    """
    if not (middle and suffix):
        return None
    call = suffix[0]
    if not isinstance(call, ASTCall) or call.function != "rows":
        return None
    params = call.params
    if (
        params.get("source") is not None
        or params.get("alias_endpoints") is not None
        or params.get("table", "nodes") != "nodes"
        or not all(isinstance(op, (ASTNode, ASTEdge)) for op in middle)
    ):
        return None
    binding_ops = params.get("binding_ops")
    if binding_ops is None:
        consumed = any(op._name is not None for op in middle)
    else:
        consumed = binding_ops == serialize_binding_ops(middle)
    return call if consumed else None


def named_middle_rows_rewrite_applies(middle: List[ASTObject], suffix: List[ASTObject]) -> bool:
    """Whether the boundary installs ``binding_ops=middle`` on the caller's ``rows()`` call."""
    call = bindings_rows_call(middle, suffix)
    return call is not None and call.params.get("binding_ops") is None


def may_skip_unread_middle_traversal(
    g: Plottable, middle: List[ASTObject], suffix: List[ASTObject], engine: TraceEngine,
) -> bool:
    """Whether the boundary may answer from the zero-row twin instead of traversing.

    Declines when a frame is unbound (the canonical path raises or materializes), when
    the index policy may serve the traversal (its index decisions stay observable), when a
    multi-hop or hop-labelled edge makes the traversal's output dtypes depend on the data,
    which a zero-row twin cannot reproduce, when a frame carries ``attrs``, which some
    pandas versions propagate through the traversal, when an alias is named like a frame
    column (the traversal then keeps restore columns a zero-row run does not produce), when a
    frame has a categorical column or an unhashable dtype, which the twin key cannot capture exactly,
    when a pandas/cuDF frame has a non-default index, and when a filter after the seed could raise or
    warn by value or column on real data, which a zero-row run and an empty-seed rebuild would not see.
    """
    from graphistry.compute.gfql.index.api import index_policy_may_serve, record_fast_path_decision

    if bindings_rows_call(middle, suffix) is None:
        return False
    if g._nodes is None or g._edges is None or g._node is None:
        reason = "graph frames not bound"
    elif index_policy_may_serve(g):
        reason = "index policy may serve the traversal"
    elif any(isinstance(op, ASTEdge) and not op.is_simple_single_hop() for op in middle):
        reason = "multi-hop and hop-label output columns depend on the traversed data"
    elif any(_has_query_string(op) for op in middle):
        reason = "query strings evaluate only on traversed rows"
    elif _carries_attrs(g._nodes) or _carries_attrs(g._edges):
        reason = "frame attrs may propagate through the traversal, depending on the pandas version"
    elif _alias_collides_with_column(middle, g._nodes, g._edges):
        reason = "an alias is named like a frame column, so the traversal's marker and restore columns depend on the data"
    elif _has_categorical(g._nodes) or _has_categorical(g._edges):
        reason = "categorical columns would need their categories in the twin key"
    elif _frame_schema(g._nodes) is None or _frame_schema(g._edges) is None:
        reason = "a column dtype is not hashable, so the twin cannot be keyed exactly"
    elif _has_custom_index(g._nodes) or _has_custom_index(g._edges):
        reason = "a non-default frame index would need to be part of the twin key"
    else:
        reason = _unsafe_later_filter(g, middle)
    record_fast_path_decision(
        path="rows_bindings_skip_traversal", engine=engine, served=not reason,
        reason=reason or "rows rebuilds bindings from the base graph",
    )
    return not reason


def _carries_attrs(frame: DataFrameT) -> bool:
    return bool(getattr(frame, "attrs", None))


def _column_names(frame: DataFrameT) -> List[str]:
    if is_polars_df(frame):
        schema = frame.collect_schema() if hasattr(frame, "collect_schema") else frame.schema
        return list(schema.names()) if hasattr(schema, "names") else list(schema)
    return list(frame.columns)


def _alias_collides_with_column(middle: List[ASTObject], nodes: DataFrameT, edges: DataFrameT) -> bool:
    aliases = {op._name for op in middle if isinstance(op, (ASTNode, ASTEdge)) and isinstance(op._name, str)}
    return bool(aliases) and not aliases.isdisjoint(_column_names(nodes) + _column_names(edges))


def _has_custom_index(frame: DataFrameT) -> bool:
    """A pandas/cuDF index that is not an unnamed ``RangeIndex``; Polars frames have none."""
    if is_polars_df(frame):
        return False
    index = frame.index
    return type(index).__name__ != "RangeIndex" or index.name is not None


def _op_filters(op: ASTObject) -> List[Tuple[Optional[Dict[str, object]], str]]:
    """The filter dicts of an op with the frame ("nodes" or "edges") each one reads."""
    if isinstance(op, ASTNode):
        return [(op.filter_dict, "nodes")]
    if isinstance(op, ASTEdge):
        return [(op.edge_match, "edges"), (op.source_node_match, "nodes"), (op.destination_node_match, "nodes")]
    return []


def _column_dtype_names(frame: DataFrameT) -> Dict[str, str]:
    if is_polars_df(frame):  # a Polars Object column holds arbitrary Python values, unlike a pandas/cuDF object column of strings
        return {str(column): "polars-object" if str(dtype) == "Object" else str(dtype).lower() for column, dtype in frame.schema.items()}
    return {str(column): str(dtype).lower() for column, dtype in frame.dtypes.items()}


_INT_DTYPE = re.compile(r"(u?)int(8|16|32|64)")
_FLOAT_MAX = {"float16": 65504.0, "float32": 3.4028234663852886e38, "float64": math.inf}
_STRING_DTYPES = frozenset({"object", "str", "utf8", "string", "string[python]", "string[pyarrow]"})
_BOOL_DTYPES = frozenset({"bool", "boolean"})


def _int_range(dtype: str) -> Optional[Tuple[int, int]]:
    """The inclusive ``(low, high)`` of an integer dtype name, or None for any other dtype."""
    match = _INT_DTYPE.fullmatch(dtype)
    if match is None:
        return None
    bits = int(match.group(2))
    return (0, 2 ** bits - 1) if match.group(1) else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


def _scalar_fits(value: object, dtype: str) -> bool:
    """Whether comparing ``value`` to a column of ``dtype`` has an outcome fixed by the scalar's type and the dtype.

    Allowlist of (scalar type, exact dtype): a string on a string column, a bool on a bool column, an int
    inside the dtype's range on an integer column, a float inside the dtype's finite range on a float
    column. Everything else (overflow, nan/inf, None, a type the column cannot hold, nested or
    interval dtypes) can raise or warn by value.
    """
    if isinstance(value, bool):
        return dtype in _BOOL_DTYPES
    if not _is_exact_scalar(value):  # a numpy scalar that subclasses float or str may compare differently
        return False
    if isinstance(value, str):
        return dtype in _STRING_DTYPES
    if isinstance(value, int):
        bounds = _int_range(dtype)
        return bounds is not None and bounds[0] <= value <= bounds[1]
    if isinstance(value, float):
        limit = _FLOAT_MAX.get(dtype)
        return limit is not None and math.isfinite(value) and abs(value) <= limit
    return False


def _is_exact_scalar(value: object) -> bool:
    return type(value) in (str, int, float)


def _numeric_comparand_fits(value: object, dtype: str) -> bool:
    """Whether a ``gt``/``lt``/``eq``-style comparison of a numeric column with ``value`` cannot raise or warn by value.

    An int must fit the integer column's range, or a 64-bit integer on a float column (a wider Python int
    cannot become an engine literal); a float must be finite and inside the float column's range.
    """
    if not _is_exact_scalar(value) or isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        return False
    bounds = _int_range(dtype)
    if bounds is not None:
        return bounds[0] <= value <= bounds[1]
    limit = _FLOAT_MAX.get(dtype)
    if limit is None or not math.isfinite(value) or abs(value) > limit:
        return False
    return not isinstance(value, int) or -(2 ** 63) <= value <= 2 ** 64 - 1


def _unsafe_later_filter(g: Plottable, middle: List[ASTObject]) -> str:
    """Why a filter may raise or warn by value or column on real data, or "" when none can.

    A ``None`` value declines on every op (Polars warns per expression built, so a cached twin changes
    the warning count). The rebuild evaluates the seed op on the real frames after the skipped path may
    have warned, and the twin evaluates later ops on zero rows only, so a filter value on any op is safe
    only when it names an existing column and its (scalar type, dtype) pair is allowlisted; a seed may also
    be a numeric comparison inside the column's range. Any other value (a regex predicate, an overflowing
    int, a type the column cannot hold) can fail on real data, and the replay would then warn twice.
    """
    assert g._nodes is not None and g._edges is not None
    dtypes = {"nodes": _column_dtype_names(g._nodes), "edges": _column_dtype_names(g._edges)}
    for index, op in enumerate(middle):
        for filters, frame in _op_filters(op):
            for column, value in (filters or {}).items():
                if value is None:
                    return "a None filter value warns per expression built on some engines"
                if column not in dtypes[frame]:
                    return "a filter names a column the frame does not have"
                if index == 0 and isinstance(value, (ComparisonPredicate, NumericASTPredicate)):
                    fits = _numeric_comparand_fits(value.val, dtypes[frame][column])
                else:
                    fits = _scalar_fits(value, dtypes[frame][column])
                if not fits:
                    return "a filter value is not an allowlisted (scalar type, dtype) pair for its column, so it can raise or warn by value"
    return ""


_CATEGORICAL_DTYPE_NAMES = frozenset({"CategoricalDtype", "Categorical", "Enum"})


def _has_categorical(frame: DataFrameT) -> bool:
    """A pandas/cuDF categorical or a Polars Categorical/Enum column; costs one dtype read per column."""
    dtypes = frame.schema.values() if is_polars_df(frame) else frame.dtypes.values
    return any(type(dtype).__name__ in _CATEGORICAL_DTYPE_NAMES for dtype in dtypes)


def _has_query_string(op: ASTObject) -> bool:
    if isinstance(op, ASTNode):
        return op.query is not None
    return isinstance(op, ASTEdge) and any(
        query is not None for query in (op.edge_query, op.source_node_query, op.destination_node_query)
    )


def run_without_traversal(
    g: Plottable,
    middle: List[ASTObject],
    engine: TraceEngine,
    traverse: Callable[[Plottable], Plottable],
    run_suffix: Callable[[Plottable], Plottable],
) -> Optional[Plottable]:
    """The suffix's answer on the zero-row twin of ``g``, or None when the canonical path must answer.

    A GFQL, unsupported-feature, Polars or builtin data-evaluation error (a dtype or value
    the engine's rebuild rejects differently than the canonical filters) replays the canonical
    traversal, so a query raises exactly what it raised before, and only failing queries pay twice.
    """
    from graphistry.compute.gfql.index.api import record_fast_path_decision

    inserted: Optional[Tuple[Tuple[object, ...], Tuple[DataFrameT, DataFrameT]]] = None
    try:
        twin, inserted = _zero_row_twin(g, middle, str(engine), traverse)
        return run_suffix(twin)
    except _replayable_errors() as error:
        _discard_inserted_twin(inserted)
        record_fast_path_decision(
            path="rows_bindings_skip_traversal", engine=engine, served=False,
            reason="replayed the traversal after an error in the skipped path", replayed_after=type(error).__name__,
        )
        return None
    except BaseException:
        _discard_inserted_twin(inserted)  # an error the canonical path surfaces itself still leaves no entry behind
        raise


def _discard_inserted_twin(inserted: Optional[Tuple[Tuple[object, ...], Tuple[DataFrameT, DataFrameT]]]) -> None:
    """Remove the cache entry one failed attempt inserted, unless a concurrent attempt has replaced it."""
    if inserted is None:
        return
    twin_key, twin_frames = inserted
    with _TWINS_LOCK:
        if _TWINS.get(twin_key) is twin_frames:
            _TWINS.pop(twin_key)


def _replayable_errors() -> Tuple[Type[BaseException], ...]:
    from graphistry.compute.exceptions import GFQLValidationError

    errors: Tuple[Type[BaseException], ...] = (
        GFQLValidationError, NotImplementedError, TypeError, ValueError, KeyError, ArithmeticError,
    )
    try:
        import polars as pl
    except ImportError:
        return errors
    try:
        import pyarrow.lib
        errors += (pyarrow.lib.ArrowException,)
    except ImportError:
        pass
    return errors + (pl.exceptions.PolarsError,)


def _zero_row_twin(
    g: Plottable, middle: List[ASTObject], engine_key: str, traverse: Callable[[Plottable], Plottable],
) -> Tuple[Plottable, Optional[Tuple[Tuple[object, ...], Tuple[DataFrameT, DataFrameT]]]]:
    """``g`` with the frames its canonical middle traversal would produce, at zero rows, plus the cache entry this call inserted.

    The traversal runs once per input schema and plan shape on a zero-row copy of ``g``,
    outside any caller's index trace, and fresh copies of its frames serve later queries.
    The inserted entry (None on a cache hit) lets a failed attempt remove exactly what it added.
    """
    from graphistry.compute.gfql.index.api import drop_index, suspended_index_trace, with_index_policy

    assert g._nodes is not None and g._edges is not None
    key = (
        engine_key, _ambient_state(), type(g._nodes), type(g._edges), _frame_schema(g._nodes), _frame_schema(g._edges),
        g._node, g._source, g._destination, g._edge, _plan_shape(middle),
    )
    with _TWINS_LOCK:
        frames = _TWINS.get(key)
        if frames is not None:
            _TWINS.move_to_end(key)
    inserted: Optional[Tuple[Tuple[object, ...], Tuple[DataFrameT, DataFrameT]]] = None
    if frames is None:
        empty = with_index_policy(drop_index(g.nodes(g._nodes[:0]).edges(g._edges[:0])), "off")
        with suspended_index_trace():
            out = traverse(empty)
        frames = (_independent(out._nodes), _independent(out._edges))
        with _TWINS_LOCK:
            _TWINS[key] = frames
            inserted = (key, frames)
            if len(_TWINS) > _TWINS_MAX:
                _TWINS.popitem(last=False)
    twin = g.bind()
    twin._nodes, twin._edges = (_fresh(frame) for frame in frames)
    return twin, inserted


def _ambient_state() -> Tuple[object, ...]:
    """Context the traversal reads besides the graph and the plan: restore columns, strictness, target."""
    from graphistry.compute.gfql.identifiers import shadow_restore_wanted
    from graphistry.compute.gfql.lazy import active_target
    from graphistry.compute.gfql.strictness import current_strict_level

    return (shadow_restore_wanted(), current_strict_level(), active_target())


def _independent(frame: DataFrameT) -> DataFrameT:
    """A zero-row frame that owns its buffers, so the cache never pins the caller's column memory."""
    if is_polars_df(frame):
        return frame.clear()  # a zero-row Polars frame keeps the schema and owns no buffers
    return frame.copy(deep=True)


def _fresh(frame: DataFrameT) -> DataFrameT:
    """A zero-row copy for one query: pandas/cuDF frames are deep copies, immutable Polars frames are shared."""
    if is_polars_df(frame):
        return frame  # a Polars frame is immutable, so the cached one is safe to share
    copied = frame.copy(deep=True)
    if hasattr(copied, "attrs"):
        copied.attrs = {}
    return copied


def _frame_schema(df: DataFrameT) -> Optional[FrameSchema]:
    """Typed column labels with their exact dtype objects, or None when a dtype is unhashable.

    Label types keep ``1`` and ``True`` apart; dtype objects compare and hash exactly (tz,
    unit, precision, nested fields), so no repr is ever used as a key.
    """
    items = df.schema.items() if is_polars_df(df) else df.dtypes.items()
    schema = tuple(((type(column), column), dtype) for column, dtype in items)
    try:
        hash(schema)
    except TypeError:
        return None
    return schema


def _plan_shape(middle: List[ASTObject]) -> Tuple[object, ...]:
    """The plan with filter values reduced to their types: values never change output columns."""
    return tuple(
        tuple(sorted(
            (field, _filter_shape(value) if field in _FILTER_FIELDS else repr(value))
            for field, value in op_json.items()
        ))
        for op_json in serialize_binding_ops(middle)
    )


def _filter_shape(filters: JSONVal) -> object:
    if not isinstance(filters, dict):
        return type(filters).__name__
    return tuple(sorted((str(column), _value_shape(value)) for column, value in filters.items()))


def _value_shape(value: JSONVal) -> object:
    if isinstance(value, dict):
        return tuple(sorted(
            (key, item if key == "type" else _value_shape(item)) for key, item in value.items()
        ))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(sorted({type(item).__name__ for item in value})))
    return type(value).__name__
