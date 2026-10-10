"""Contract of ``bridge_polars_graph``: Plottable analytics without a Polars implementation."""
from typing import Any, Dict, List

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, bridge_polars_graph

pl = pytest.importorskip("polars")

NODES = pd.DataFrame({"id": [0, 1, 2], "v": [1.0, 2.0, 3.0]})
EDGES = pd.DataFrame({"s": [0, 1], "d": [1, 2]})


def _polars_graph(lazy: bool = False) -> Any:
    n, e = pl.from_pandas(NODES), pl.from_pandas(EDGES)
    if lazy:
        n, e = n.lazy(), e.lazy()
    return graphistry.nodes(n, "id").edges(e, "s", "d")


def _recording(calls: List[Dict[str, Any]], with_engine: bool) -> Any:
    """An analytic that records the frame type and engine it ran with, then adds a column."""
    if with_engine:
        def analytic(g: Any, scale: float = 1.0, engine: str = "auto") -> Any:
            calls.append({"frames": type(g._nodes).__module__.split(".")[0], "engine": engine})
            return g.nodes(g._nodes.assign(out=g._nodes["v"] * scale))
    else:
        def analytic(g: Any, scale: float = 1.0) -> Any:  # type: ignore[misc]
            calls.append({"frames": type(g._nodes).__module__.split(".")[0]})
            return g.nodes(g._nodes.assign(out=g._nodes["v"] * scale))
    return analytic


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("engine_kwargs", [{}, {"engine": "auto"}, {"engine": "polars"}], ids=["none", "auto", "polars"])
def test_polars_input_runs_on_compute_engine_and_returns_eager_polars(lazy: bool, engine_kwargs: Dict[str, str]) -> None:
    calls: List[Dict[str, Any]] = []
    fn = bridge_polars_graph(Engine.PANDAS)(_recording(calls, with_engine=True))
    out = fn(_polars_graph(lazy), 2.0, **engine_kwargs)
    assert calls[0]["frames"] == "pandas"
    assert calls[0]["engine"] in ("auto", "pandas")  # 'polars' is rewritten to the compute engine
    assert isinstance(out._nodes, pl.DataFrame) and isinstance(out._edges, pl.DataFrame)
    assert out._nodes["out"].to_list() == [2.0, 4.0, 6.0]
    assert out._edges.to_pandas().equals(EDGES)


def test_explicit_pandas_engine_returns_pandas() -> None:
    calls: List[Dict[str, Any]] = []
    fn = bridge_polars_graph(Engine.PANDAS)(_recording(calls, with_engine=True))
    out = fn(_polars_graph(), engine="pandas")
    assert calls == [{"frames": "pandas", "engine": "pandas"}]
    assert isinstance(out._nodes, pd.DataFrame) and isinstance(out._edges, pd.DataFrame)


def test_positional_engine_is_honored() -> None:
    calls: List[Dict[str, Any]] = []
    fn = bridge_polars_graph(Engine.PANDAS)(_recording(calls, with_engine=True))
    out = fn(_polars_graph(), 1.0, "pandas")
    assert calls[0]["engine"] == "pandas"
    assert isinstance(out._nodes, pd.DataFrame)


def test_analytic_without_engine_parameter() -> None:
    calls: List[Dict[str, Any]] = []
    fn = bridge_polars_graph(Engine.PANDAS)(_recording(calls, with_engine=False))
    out = fn(_polars_graph(lazy=True))
    assert calls == [{"frames": "pandas"}]
    assert isinstance(out._nodes, pl.DataFrame)


def test_non_polars_graph_passes_through_untouched() -> None:
    calls: List[Dict[str, Any]] = []
    fn = bridge_polars_graph(Engine.PANDAS)(_recording(calls, with_engine=True))
    g = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")
    out = fn(g, engine="auto")
    assert calls == [{"frames": "pandas", "engine": "auto"}]
    assert isinstance(out._nodes, pd.DataFrame)
    assert out._edges is EDGES


def test_binding_dtypes_restored_after_round_trip() -> None:
    """An analytic that widens ids (as pandas/igraph rebuilds can) returns the input id dtypes."""
    def widening(g: Any) -> Any:
        nodes = g._nodes.astype({"id": "float64"})
        edges = g._edges.astype({"s": "float64", "d": "float64"})
        return g.nodes(nodes, "id").edges(edges, "s", "d")

    n = pl.from_pandas(NODES).with_columns(pl.col("id").cast(pl.Int32))
    e = pl.from_pandas(EDGES).with_columns(pl.col("s").cast(pl.UInt16), pl.col("d").cast(pl.UInt16))
    out = bridge_polars_graph(Engine.PANDAS)(widening)(graphistry.nodes(n, "id").edges(e, "s", "d"))
    assert out._nodes.schema["id"] == pl.Int32
    assert out._edges.schema["s"] == pl.UInt16 and out._edges.schema["d"] == pl.UInt16
    assert out._nodes["v"].to_list() == [1.0, 2.0, 3.0]


def test_unrepresentable_binding_values_keep_analytic_dtype() -> None:
    def fractional_ids(g: Any) -> Any:
        return g.nodes(g._nodes.assign(id=g._nodes["id"] + 0.5), "id")

    out = bridge_polars_graph(Engine.PANDAS)(fractional_ids)(_polars_graph())
    assert out._nodes.schema["id"] == pl.Float64
    assert out._nodes["id"].to_list() == [0.5, 1.5, 2.5]


def test_other_engines_are_left_to_the_analytic() -> None:
    def analytic(g: Any, engine: str = "auto") -> Any:
        raise ValueError(f"unsupported engine {engine}")

    with pytest.raises(ValueError, match="unsupported engine dask"):
        bridge_polars_graph(Engine.PANDAS)(analytic)(_polars_graph(), engine="dask")
