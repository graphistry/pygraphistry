"""Skip the chain-boundary traversal that a bindings ``rows()`` call never reads.

The bindings builders on every engine rebuild their rows from the base graph, the carried
start nodes and the alias prefilters. When the leading suffix call is such a ``rows()``,
the canonical middle traversal only contributes its output schema. The suffix runs on a
zero-row twin that has that schema, and any error replays the canonical path so errors
stay exactly as they were.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Type

from graphistry.Engine import is_polars_df
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject, serialize_binding_ops
from graphistry.compute.gfql.cache_registry import register_clearable_callable
from graphistry.compute.gfql.index.types import TraceEngine
from graphistry.compute.typing import DataFrameT
from graphistry.utils.json import JSONVal

FrameSchema = Tuple[Tuple[object, object], ...]

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
    which a zero-row twin cannot reproduce, and when a frame carries ``attrs``, which some
    pandas versions propagate through the traversal.
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
    else:
        reason = ""
    record_fast_path_decision(
        path="rows_bindings_skip_traversal", engine=engine, served=not reason,
        reason=reason or "rows rebuilds bindings from the base graph",
    )
    return not reason


def _carries_attrs(frame: DataFrameT) -> bool:
    return bool(getattr(frame, "attrs", None))


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

    Any GFQL, unsupported-feature or polars error replays the canonical traversal, so a
    query raises exactly what it raised before, and only failing queries pay twice.
    """
    from graphistry.compute.gfql.index.api import record_fast_path_decision

    try:
        return run_suffix(_zero_row_twin(g, middle, str(engine), traverse))
    except _replayable_errors() as error:
        record_fast_path_decision(
            path="rows_bindings_skip_traversal", engine=engine, served=False,
            reason=f"replayed the traversal after {type(error).__name__}",
        )
        return None


def _replayable_errors() -> Tuple[Type[BaseException], ...]:
    from graphistry.compute.exceptions import GFQLValidationError

    errors: Tuple[Type[BaseException], ...] = (GFQLValidationError, NotImplementedError)
    try:
        import polars as pl
    except ImportError:
        return errors
    return errors + (pl.exceptions.PolarsError,)


def _zero_row_twin(
    g: Plottable, middle: List[ASTObject], engine_key: str, traverse: Callable[[Plottable], Plottable],
) -> Plottable:
    """``g`` with the frames its canonical middle traversal would produce, at zero rows.

    The traversal runs once per input schema and plan shape on a zero-row copy of ``g``,
    outside any caller's index trace, and fresh copies of its frames serve later queries.
    """
    from graphistry.compute.gfql.index.api import drop_index, suspended_index_trace, with_index_policy

    assert g._nodes is not None and g._edges is not None
    key = (
        engine_key, type(g._nodes), type(g._edges), _frame_schema(g._nodes), _frame_schema(g._edges),
        g._node, g._source, g._destination, g._edge, _plan_shape(middle),
    )
    with _TWINS_LOCK:
        frames = _TWINS.get(key)
        if frames is not None:
            _TWINS.move_to_end(key)
    if frames is None:
        empty = with_index_policy(drop_index(g.nodes(g._nodes[:0]).edges(g._edges[:0])), "off")
        with suspended_index_trace():
            out = traverse(empty)
        frames = (out._nodes, out._edges)
        with _TWINS_LOCK:
            _TWINS[key] = frames
            if len(_TWINS) > _TWINS_MAX:
                _TWINS.popitem(last=False)
    twin = g.bind()
    twin._nodes, twin._edges = (_fresh(frame) for frame in frames)
    return twin


def _fresh(frame: DataFrameT) -> DataFrameT:
    """An unshared zero-row copy without any ``attrs`` (the skip is declined for graphs that carry them)."""
    if is_polars_df(frame):
        return frame
    copied = frame.copy(deep=True)
    if hasattr(copied, "attrs"):
        copied.attrs = {}
    return copied


def _frame_schema(df: DataFrameT) -> FrameSchema:
    """Typed column labels and dtype objects, with category order for categoricals.

    Label types keep ``1`` and ``True`` apart; unordered categoricals compare equal
    across category orders, so the order is keyed explicitly.
    """
    items = df.schema.items() if is_polars_df(df) else df.dtypes.items()
    return tuple(((type(column), column), (dtype, _category_order(dtype))) for column, dtype in items)


def _category_order(dtype: object) -> Tuple[object, ...]:
    categories = getattr(dtype, "categories", None)
    if categories is None:
        return ()
    if hasattr(categories, "to_pandas"):  # cuDF refuses host list conversion on its index
        categories = categories.to_pandas()
    return tuple(categories.to_list() if hasattr(categories, "to_list") else categories)


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
