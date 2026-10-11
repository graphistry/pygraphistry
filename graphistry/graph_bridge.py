"""Graph-level analytic bridge between Polars-bound graphs and pandas/cuDF solvers.

This module stays early-import-safe: optional Polars imports occur only when a bound
Polars graph actually reaches the bridge. Generic frame conversion remains in Engine.py.
"""
from __future__ import annotations

from enum import Enum
import functools
import inspect
from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypeVar, cast

from graphistry.Engine import Engine, EngineAbstract, graph_frames_to_engine, is_polars_df

if TYPE_CHECKING:
    import polars as pl
    from graphistry.Plottable import Plottable
    from graphistry.compute.typing import PolarsFrame

_AnalyticT = TypeVar("_AnalyticT", bound=Callable[..., object])


def _restore_binding_dtypes(
    df: "pl.DataFrame", original: "PolarsFrame", cols: Sequence[Optional[str]]
) -> "pl.DataFrame":
    """Cast binding columns back to their input Polars dtype after a pandas/cuDF round trip.

    The round trip can widen identifiers (nullable Int64 becomes float64 in pandas). Only
    binding columns are restored, and only when the cast is exact: a value that does not
    survive the cast back (0.5 truncated to 0, for example) keeps the analytic's dtype.
    """
    from polars.exceptions import PolarsError

    before = dict(original.collect_schema())
    restored = []
    for c in cols:
        if c is None or c not in before or c not in df.columns or df.schema[c] == before[c]:
            continue
        current = df.get_column(c)
        try:
            cast_col = current.cast(before[c], strict=True)
            if cast_col.cast(current.dtype).equals(current):
                restored.append(cast_col)
        except (PolarsError, TypeError, ValueError, OverflowError):
            continue
    return df.with_columns(restored) if restored else df


def _has_engine_param(fn: Callable[..., object]) -> bool:
    try:
        return "engine" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def bridge_polars_graph(compute_engine: Engine = Engine.PANDAS) -> Callable[[_AnalyticT], _AnalyticT]:
    """Decorate a Plottable analytic that has no Polars implementation.

    Non-Polars graphs pass through untouched. For a graph bound to Polars frames
    (``DataFrame`` or ``LazyFrame``; lazy frames are collected):

    - no ``engine`` argument, ``'auto'`` or ``'polars'``: run on ``compute_engine`` and
      return eager Polars frames;
    - ``'polars-gpu'``: run on cuDF and return eager Polars frames;
    - ``'pandas'`` or ``'cudf'``: run on that engine and return its frames, as requested.

    Node id, source, destination and edge id columns keep their input Polars dtype.
    This matches the GFQL ``call()`` contract for analytics under Polars.
    """
    def decorate(fn: _AnalyticT) -> _AnalyticT:
        has_engine = _has_engine_param(fn)

        @functools.wraps(fn)
        def wrapper(g: "Plottable", *args: object, **kwargs: object) -> object:
            if not (is_polars_df(g._nodes) or is_polars_df(g._edges)):
                return fn(g, *args, **kwargs)
            bound = None
            requested = kwargs.get("engine", EngineAbstract.AUTO)
            if has_engine and "engine" not in kwargs:
                bound = inspect.signature(fn).bind_partial(g, *args, **kwargs)
                requested = bound.arguments.get("engine", EngineAbstract.AUTO)
            requested_value = requested.value if isinstance(requested, Enum) else requested
            if requested_value in (Engine.PANDAS.value, Engine.CUDF.value):
                target, restore = Engine(requested_value), False
            elif requested_value == Engine.POLARS_GPU.value:
                target, restore = Engine.CUDF, True
            elif requested_value in (EngineAbstract.AUTO.value, Engine.POLARS.value):
                target, restore = compute_engine, True
            else:
                return fn(g, *args, **kwargs)  # dask etc.: the analytic reports its own support
            if has_engine and requested_value not in (EngineAbstract.AUTO.value, target.value):
                if bound is not None:
                    bound.arguments["engine"] = target.value
                    args, kwargs = tuple(bound.args[1:]), dict(bound.kwargs)
                else:
                    kwargs = {**kwargs, "engine": target.value}
            out = fn(graph_frames_to_engine(g, target), *args, **kwargs)
            if not restore or not hasattr(out, "_nodes"):
                return out
            result = graph_frames_to_engine(cast("Plottable", out), Engine.POLARS)
            if result._nodes is not None and is_polars_df(g._nodes):
                result = result.nodes(_restore_binding_dtypes(result._nodes, g._nodes, [result._node]), result._node)
            if result._edges is not None and is_polars_df(g._edges):
                result = result.edges(
                    _restore_binding_dtypes(
                        result._edges, g._edges, [result._source, result._destination, result._edge]
                    ),
                    result._source, result._destination, edge=result._edge,
                )
            return result

        return cast(_AnalyticT, wrapper)  # functools.wraps keeps the signature

    return decorate


