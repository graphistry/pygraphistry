"""H2 FUSED LAZY LANE -- ``_single_hop_grouped_aggregate_fused_polars``.

The single-hop GROUPED AGGREGATE fast path
(``_execute_single_hop_grouped_aggregate_fast_path``) serves the cypher shape::

    MATCH (a {..})-[{..}]->(b {..}) [WHERE ..]
    RETURN <alias>.<prop> AS k [, ..], <agg> AS v [, ..]
    [ORDER BY ..] [LIMIT n]

Its polars branch chained EAGER ops -- two semi-joins, two property joins, a group_by,
a sort and a head, each its own ``lazy().collect(_eager=True)``. The fused lane expresses
the SAME op sequence as ONE lazy plan collected once. It is strictly additive: every
decline falls through to the untouched eager twin.

WHAT THESE TESTS PIN (and why each exists):

* ENGAGEMENT: which shapes the fused lane serves, which it DECLINES, and which never
  reach it at all because the fast path itself declines first. Enumerated, not inherited.
* THE ORDER-TOTALITY GATE, WHICH IS THE CORRECTNESS CRUX. The eager twin's row order for
  a shape whose ORDER BY does not name every group key comes from ``maintain_order=True``
  group FIRST-APPEARANCE order over an eager join output. A lazy plan may reorder or
  re-side those joins. MEASURED on this build (polars 1.42, an ungated variant of the same
  plan vs the eager twin, 4 graph sizes x 4 seeds x 4 order-undetermined shapes): 64
  comparisons, **47 divergent**, and for the LIMIT-bearing shapes the divergence is a
  different ROW SET, not merely a different row order. So the gate is load-bearing and the
  decline is a measurement, not a precaution. ``test_..._order_totality_gate_declines_*``
  pins it; ``test_..._total_order_shapes_are_stable`` pins the positive side.
* THE HOISTED METADATA GUARD (a fused lane must not answer a query the fast path
  DECLINES): a property column missing from its alias' node frame makes the eager twin
  return ``None`` MID-CHAIN, which declines the whole fast path and lets the query raise
  the polars ``NotImplementedError``. A fused plan built before that check would have
  answered instead. ``test_..._declines_missing_property_column`` pins the raise.
* MULTIPLICITY, null ordering, degenerate bindings, empty matches, non-numeric ids.

DISCLOSED PRE-EXISTING DIVERGENCE (NOT introduced here, and deliberately NOT fixed here):
the polars branch's property lookup is not deduplicated by node id while the pandas branch
does ``drop_duplicates(subset=[node_col])``, so a node table carrying the SAME id twice
multiplies matched rows on polars and not on pandas. The fused lane reproduces the eager
polars behaviour exactly -- ``test_..._duplicate_node_rows_match_the_eager_twin`` compares
against the eager twin for that reason, and says so.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
import graphistry.compute.gfql_fast_paths as gfql_fast_paths_module
import graphistry.compute.gfql_unified as gfql_unified_module


# --------------------------------------------------------------------------- fixtures

def _base_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "kind": ["P", "P", "P", "P", "C", "C", "C", "C"],
        "age": [20, 30, 40, 50, None, None, None, None],
        "city": [None, None, None, None, "LA", "NY", "SF", "LA"],
        "country": [None, None, None, None, "US", "US", "US", "MX"],
    })
    edges = pd.DataFrame({
        "s": [1, 2, 3, 4, 1, 2, 3, 4, 1],
        "d": [5, 5, 6, 7, 8, 6, 8, 8, 5],
        "rel": ["L", "L", "L", "L", "L", "L", "L", "L", "X"],
    })
    return nodes, edges


def _dup_node_rows_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate rows on the END (city) side of the hop."""
    nodes, edges = _base_data()
    return pd.concat([nodes, nodes.iloc[[4, 5]]], ignore_index=True), edges


def _dup_start_node_rows_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate rows on the START (person) side of the hop. Kept SEPARATE from the end-side
    fixture: a mutation that turns the START semi-join into a non-deduplicated inner join is
    invisible unless the duplicates sit on that arm (found exactly that way)."""
    nodes, edges = _base_data()
    return pd.concat([nodes, nodes.iloc[[0, 1]]], ignore_index=True), edges


def _self_loops_parallel_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    extra = pd.DataFrame({"s": [5, 1, 1, 6], "d": [5, 5, 5, 6], "rel": ["L", "L", "L", "L"]})
    return nodes, pd.concat([edges, extra], ignore_index=True)


def _string_ids_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return (
        nodes.assign(id=[f"n{i}" for i in nodes["id"]]),
        edges.assign(s=[f"n{i}" for i in edges["s"]], d=[f"n{i}" for i in edges["d"]]),
    )


def _dangling_endpoints_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    extra = pd.DataFrame({"s": [99, 1], "d": [5, 98], "rel": ["L", "L"]})
    return nodes, pd.concat([edges, extra], ignore_index=True)


def _empty_edges_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return nodes, edges.iloc[:0].copy()


def _no_matching_nodes_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return nodes.assign(kind="Z"), edges


def _null_props_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    nodes = nodes.copy()
    nodes.loc[nodes["id"].isin([6, 7]), "city"] = None
    nodes.loc[nodes["id"] == 2, "age"] = None
    return nodes, edges


def _all_null_group_key_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    nodes = nodes.copy()
    nodes["city"] = None
    return nodes, edges


_GRAPHS: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]] = {
    "base": _base_data,
    "dup_node_rows": _dup_node_rows_data,
    "dup_start_node_rows": _dup_start_node_rows_data,
    "self_loops_parallel": _self_loops_parallel_data,
    "string_ids": _string_ids_data,
    "dangling_endpoints": _dangling_endpoints_data,
    "empty_edges": _empty_edges_data,
    "no_matching_nodes": _no_matching_nodes_data,
    "null_props": _null_props_data,
    "all_null_group_key": _all_null_group_key_data,
}


# ------------------------------------------------------------------------- shape corpus

# SERVED: the sort names every group key, so it is TOTAL over the output rows.
Q_COUNT_STAR = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC LIMIT 3")
Q_COUNT_STAR_NO_LIMIT = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC")
Q_AVG = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) WHERE c.country = 'US' "
    "RETURN c.city AS city, avg(p.age) AS a ORDER BY a ASC, city ASC LIMIT 5")
Q_COUNT_ALIAS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(p) AS n "
    "ORDER BY n DESC, city ASC LIMIT 3")
Q_COUNT_PROP = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(p.age) AS n "
    "ORDER BY n DESC, city ASC")
Q_SUM = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, sum(p.age) AS s "
    "ORDER BY s DESC, city ASC")
Q_TWO_GROUP_KEYS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.country AS co, c.city AS city, "
    "count(*) AS n ORDER BY n DESC, co ASC, city ASC")
Q_TWO_AGGS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n, "
    "avg(p.age) AS a ORDER BY city ASC")
Q_GROUP_START_PROP = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN p.age AS age, count(*) AS n "
    "ORDER BY age ASC")
Q_GROUP_KEY_DESC = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY city DESC")
Q_UNFILTERED_ENDS = (
    "MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, count(*) AS n ORDER BY n DESC, city ASC")
Q_UNFILTERED_AVG_DESC = (
    "MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, avg(p.age) AS a "
    "ORDER BY a DESC, city ASC LIMIT 3")
Q_NO_EDGE_FILTER = (
    "MATCH (p {kind:'P'})-[]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC")
Q_WHERE_ON_START = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) WHERE p.age >= 30 "
    "RETURN c.city AS city, count(*) AS n ORDER BY n DESC, city ASC")

_SERVED_SHAPES: List[Tuple[str, str]] = [
    ("count_star_total_limit", Q_COUNT_STAR),
    ("count_star_total_nolimit", Q_COUNT_STAR_NO_LIMIT),
    ("avg_total_limit", Q_AVG),
    ("count_alias", Q_COUNT_ALIAS),
    ("count_prop", Q_COUNT_PROP),
    ("sum_agg", Q_SUM),
    ("two_group_keys", Q_TWO_GROUP_KEYS),
    ("two_aggs_count_avg", Q_TWO_AGGS),
    ("group_start_prop", Q_GROUP_START_PROP),
    ("group_key_desc", Q_GROUP_KEY_DESC),
    ("unfiltered_ends", Q_UNFILTERED_ENDS),
    ("unfiltered_ends_avg_desc", Q_UNFILTERED_AVG_DESC),
    ("no_edge_filter", Q_NO_EDGE_FILTER),
    ("where_on_start", Q_WHERE_ON_START),
]

# DECLINED by the fused lane -- the fast path still answers via the eager twin.
Q_PARTIAL_ORDER_LIMIT = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC LIMIT 2")
Q_PARTIAL_ORDER = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC")
Q_NO_ORDER = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n")
Q_NO_ORDER_LIMIT = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n LIMIT 2")
Q_OUT_COL_COLLIDES_SRC = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS s, count(*) AS n "
    "ORDER BY n DESC, s ASC")

_DECLINED_SHAPES: List[Tuple[str, str]] = [
    ("partial_order_with_limit", Q_PARTIAL_ORDER_LIMIT),
    ("partial_order_no_limit", Q_PARTIAL_ORDER),
    ("no_order_by", Q_NO_ORDER),
    ("no_order_by_with_limit", Q_NO_ORDER_LIMIT),
    ("out_col_collides_with_src", Q_OUT_COL_COLLIDES_SRC),
]

# NEVER REACHED: the fast path's own shape guard declines before the fused lane exists.
_UNREACHED_SHAPES: List[Tuple[str, str]] = [
    ("undirected",
     "MATCH (p {kind:'P'})-[{rel:'L'}]-(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
     "ORDER BY n DESC, city ASC"),
    ("reverse",
     "MATCH (c {kind:'C'})<-[{rel:'L'}]-(p {kind:'P'}) RETURN c.city AS city, count(*) AS n "
     "ORDER BY n DESC, city ASC"),
    ("two_hop",
     "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'})-[{rel:'L'}]->(x) RETURN x.city AS city, "
     "count(*) AS n ORDER BY n DESC, city ASC"),
    ("named_edge",
     "MATCH (p {kind:'P'})-[r {rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
     "ORDER BY n DESC, city ASC"),
    ("no_aggregate",
     "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city ORDER BY city ASC"),
    ("aggregate_without_group_key",
     "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN count(*) AS n"),
    ("skip_and_limit",
     "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
     "ORDER BY n DESC, city ASC SKIP 1 LIMIT 2"),
]


# ------------------------------------------------------------------------------ helpers

def _require_polars() -> Any:
    return pytest.importorskip("polars")


def _require_polars_gpu() -> None:
    """polars-gpu needs cudf_polars AND a working device. Skips LOUDLY with the reason so a
    CPU-only run reports a coverage boundary instead of a silent green."""
    pl = _require_polars()
    pytest.importorskip("cudf_polars")
    try:
        pl.DataFrame({"a": [1, 2]}).lazy().filter(pl.col("a") > 0).collect(
            engine=pl.GPUEngine(raise_on_fail=True)
        )
    except Exception as exc:  # pragma: no cover - CPU-only CI
        pytest.skip(f"cudf_polars installed but the GPU collect probe failed: {exc}")


def _graph(
    engine: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    node_key: str = "id",
    src: str = "s",
    dst: str = "d",
) -> Plottable:
    if engine == "pandas":
        return graphistry.nodes(nodes_df, node_key).edges(edges_df, src, dst)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        try:
            _ = cudf.Series([1, 2, 3])
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"cudf installed but the runtime is unavailable: {exc}")
        return graphistry.nodes(cudf.from_pandas(nodes_df), node_key).edges(
            cudf.from_pandas(edges_df), src, dst)
    if engine == "polars-gpu":
        _require_polars_gpu()
    pl = _require_polars()
    return graphistry.nodes(pl.from_pandas(nodes_df), node_key).edges(
        pl.from_pandas(edges_df), src, dst)


def _records(result: Plottable) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Row-order AND column-order sensitive comparison value, NaN normalized to None so
    pandas/polars nulls compare equal.

    The COLUMN list is part of it deliberately: python dict equality ignores key order, so a
    comparison over records alone cannot see a projection that returns the right values under
    the wrong column order -- a mutation that reorders the group-by keys survived until this
    carried the schema too."""
    df = result._nodes
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    rows = [
        {k: (None if v is None or (isinstance(v, float) and v != v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    return [str(c) for c in df.columns], rows


def _probe_fused(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """One entry per fused-lane CALL: True=served, False=declined. An empty list means the
    lane was never reached (the non-polars / fast-path-declines contract)."""
    calls: List[bool] = []
    original = gfql_fast_paths_module._single_hop_grouped_aggregate_fused_polars

    def probe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(
        gfql_fast_paths_module, "_single_hop_grouped_aggregate_fused_polars", probe)
    return calls


def _force_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the eager twin as the oracle arm for the differential."""
    monkeypatch.setattr(
        gfql_fast_paths_module, "_single_hop_grouped_aggregate_fused_polars",
        lambda *a, **k: None)


def _probe_fast_path(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """One entry per FAST PATH call: True=served, False=declined."""
    calls: List[bool] = []
    original = gfql_unified_module._execute_single_hop_grouped_aggregate_fast_path

    def probe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(
        gfql_unified_module, "_execute_single_hop_grouped_aggregate_fast_path", probe)
    return calls


# ------------------------------------------------------------------------ engagement

@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("label,query", _SERVED_SHAPES, ids=[s[0] for s in _SERVED_SHAPES])
def test_grouped_aggregate_fused_polars_serves_total_order_shapes(
    engine: str, label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shape whose ORDER BY names all group keys is SERVED, and answers what pandas
    answers -- row order included."""
    nodes, edges = _base_data()
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))
    graph = _graph(engine, nodes, edges)

    calls = _probe_fused(monkeypatch)
    result = _records(graph.gfql(query, engine=engine))

    assert calls == [True], f"{label}: fused lane must serve on {engine}"
    assert result == oracle, f"{label}: fused lane diverged from the pandas oracle"


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("label,query", _DECLINED_SHAPES, ids=[s[0] for s in _DECLINED_SHAPES])
def test_grouped_aggregate_fused_polars_declines_are_called_and_hand_back(
    engine: str, label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DECLINE contract: the lane is REACHED, returns None, and the eager twin still
    answers -- so a decline can never hide a wrong answer, only forgo a speedup."""
    nodes, edges = _base_data()
    graph = _graph(engine, nodes, edges)

    with monkeypatch.context() as eager_ctx:
        _force_eager(eager_ctx)
        eager = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    calls = _probe_fused(monkeypatch)
    result = _records(graph.gfql(query, engine=engine))

    assert calls == [False], f"{label}: must DECLINE, not serve"
    assert result == eager, f"{label}: declining changed the answer"


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("label,query", _UNREACHED_SHAPES, ids=[s[0] for s in _UNREACHED_SHAPES])
def test_grouped_aggregate_fused_polars_is_never_reached_for_unsupported_shapes(
    engine: str, label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These shapes are declined by the FAST PATH's own guard, so the fused lane is not even
    called. Enumerated rather than assumed: the fast path is confirmed to have been CALLED
    and to have declined, which is what makes the empty fused-call list meaningful."""
    nodes, edges = _base_data()
    fast_calls = _probe_fast_path(monkeypatch)
    fused_calls = _probe_fused(monkeypatch)

    _graph(engine, nodes, edges).gfql(query, engine=engine)

    assert fast_calls == [False], f"{label}: expected the fast path itself to decline"
    assert fused_calls == [], f"{label}: fused lane must not be reached"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_grouped_aggregate_fused_polars_is_never_reached_by_dataframe_engines(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fused lane is polars-only; pandas/cuDF keep their own eager branch untouched."""
    nodes, edges = _base_data()
    graph = _graph(engine, nodes, edges)

    fused_calls = _probe_fused(monkeypatch)
    result = _records(graph.gfql(Q_COUNT_STAR, engine=engine))

    assert fused_calls == []
    assert result == (["city", "n"],
                      [{"city": "LA", "n": 5}, {"city": "NY", "n": 2}, {"city": "SF", "n": 1}])


# ---------------------------------------------------------------------- differential

@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
@pytest.mark.parametrize("label,query", _SERVED_SHAPES, ids=[s[0] for s in _SERVED_SHAPES])
def test_grouped_aggregate_fused_polars_matches_eager_twin_and_pandas(
    engine: str, graph_name: str, label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DIFFERENTIAL: fused == eager twin == pandas oracle, ORDER-SENSITIVELY, and the fused
    lane really ran (otherwise the comparison is vacuous).

    The two duplicate-node-row fixtures are compared against the eager twin only -- see the
    module docstring's disclosed pre-existing polars/pandas multiplicity divergence."""
    nodes, edges = _GRAPHS[graph_name]()
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))

    with monkeypatch.context() as eager_ctx:
        _force_eager(eager_ctx)
        eager = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    calls = _probe_fused(monkeypatch)
    fused = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    assert calls == [True], f"{graph_name}/{label}: lane did not serve -- differential vacuous"
    assert fused == eager, f"{graph_name}/{label}: fused lane diverged from the eager twin"
    if graph_name not in ("dup_node_rows", "dup_start_node_rows"):
        assert fused == oracle, f"{graph_name}/{label}: fused lane diverged from pandas"


@pytest.mark.parametrize("graph_name,query,fused_rows,pandas_rows", [
    # END-side duplicate rows: the property lookup for the group key multiplies matches.
    ("dup_node_rows", Q_COUNT_STAR,
     [{"city": "LA", "n": 7}, {"city": "NY", "n": 4}, {"city": "SF", "n": 1}],
     [{"city": "LA", "n": 5}, {"city": "NY", "n": 2}, {"city": "SF", "n": 1}]),
    # START-side duplicate rows multiply ONLY when a start property is projected: count(*)
    # needs none, so the same fixture is inert here and multiplies under sum(p.age).
    ("dup_start_node_rows", Q_COUNT_STAR,
     [{"city": "LA", "n": 5}, {"city": "NY", "n": 2}, {"city": "SF", "n": 1}],
     [{"city": "LA", "n": 5}, {"city": "NY", "n": 2}, {"city": "SF", "n": 1}]),
    ("dup_start_node_rows", Q_SUM,
     [{"city": "LA", "s": 230.0}, {"city": "NY", "s": 100.0}, {"city": "SF", "s": 50.0}],
     [{"city": "LA", "s": 160.0}, {"city": "NY", "s": 70.0}, {"city": "SF", "s": 50.0}]),
])
def test_grouped_aggregate_fused_polars_duplicate_node_rows_match_the_eager_twin(
    graph_name: str, query: str, fused_rows: List[Dict[str, Any]],
    pandas_rows: List[Dict[str, Any]]
) -> None:
    """DISCLOSURE, pinned. A node table carrying the same id twice multiplies matched rows on
    the polars branch (its property lookup is not deduplicated) and does NOT on pandas. That
    asymmetry predates this change; the fused lane reproduces the EAGER POLARS answer exactly
    and does not quietly 'fix' it inside a performance change.

    Both ARMS of the hop are covered separately: a mutation that turns the START semi-join
    into a non-deduplicated inner join is invisible against end-side duplicates alone."""
    pl = _require_polars()
    nodes, edges = _GRAPHS[graph_name]()
    graph = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    pandas_graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    columns = list(fused_rows[0])
    assert _records(graph.gfql(query, engine="polars")) == (columns, fused_rows)
    assert _records(pandas_graph.gfql(query, engine="pandas")) == (columns, pandas_rows)


# ------------------------------------------------------- the order-totality gate (crux)

def _many_group_graph(n_people: int, n_cities: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    person_ids = np.arange(n_people)
    city_ids = np.arange(n_people, n_people + n_cities)
    nodes = pd.DataFrame({
        "id": np.concatenate([person_ids, city_ids]),
        "kind": ["P"] * n_people + ["C"] * n_cities,
        "age": np.concatenate([rng.integers(18, 80, n_people), np.full(n_cities, np.nan)]),
        "city": [None] * n_people + [f"city{i}" for i in range(n_cities)],
    })
    edges = pd.DataFrame({
        "s": rng.choice(person_ids, size=n_people * 2),
        "d": rng.choice(city_ids, size=n_people * 2),
        "rel": "L",
    })
    return nodes, edges


@pytest.mark.parametrize("label,query", [
    ("no_order_by", Q_NO_ORDER),
    ("no_order_by_with_limit", Q_NO_ORDER_LIMIT),
    ("partial_order", Q_PARTIAL_ORDER),
    ("partial_order_with_limit", Q_PARTIAL_ORDER_LIMIT),
])
def test_grouped_aggregate_order_totality_gate_declines_undetermined_order(
    label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CRUX. A shape whose ORDER BY does not name every group key has a result ORDER the
    query does not determine -- the eager twin falls back to ``maintain_order=True`` group
    first-appearance order over an EAGER join output, which a lazy plan may legitimately
    change. With LIMIT that stops being cosmetic and changes which ROWS come back.

    Measured on this build with an ungated variant of the same lazy plan: 64 comparisons over
    4 graph sizes x 4 seeds x these 4 shapes, 47 divergent from the eager twin, LIMIT-bearing
    divergences differing in ROW SET. The gate is therefore a measurement, not a precaution.
    """
    nodes, edges = _many_group_graph(2000, 23, seed=0)
    graph = _graph("polars", nodes, edges)

    calls = _probe_fused(monkeypatch)
    graph.gfql(query, engine="polars")

    assert calls == [False], f"{label}: order is not determined by the query -- must DECLINE"


@pytest.mark.parametrize("n_people,n_cities,seed", [(200, 7, 0), (2000, 23, 1), (20000, 61, 2)])
@pytest.mark.parametrize("label,query", [
    ("count_star", Q_COUNT_STAR_NO_LIMIT),
    ("count_star_limit", Q_COUNT_STAR),
    ("group_key_desc", Q_GROUP_KEY_DESC),
])
def test_grouped_aggregate_total_order_shapes_are_stable_at_scale(
    n_people: int, n_cities: int, seed: int, label: str, query: str,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSITIVE side of the gate: with a total sort the fused plan is order-identical to the
    eager twin and to pandas even on graphs with many tied groups, where an unstable sort or
    a changed group order would show immediately."""
    nodes, edges = _many_group_graph(n_people, n_cities, seed)
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))

    with monkeypatch.context() as eager_ctx:
        _force_eager(eager_ctx)
        eager = _records(_graph("polars", nodes, edges).gfql(query, engine="polars"))

    calls = _probe_fused(monkeypatch)
    fused = _records(_graph("polars", nodes, edges).gfql(query, engine="polars"))

    assert calls == [True]
    assert fused == eager, f"{label}: fused diverged from the eager twin at scale"
    assert fused == oracle, f"{label}: fused diverged from pandas at scale"


# ------------------------------------------------- the hoisted metadata guard (CD4 trap)

@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_grouped_aggregate_fused_polars_declines_missing_property_column(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fused lane MUST NOT answer a query the fast path DECLINES.

    When a projected property column is absent from its alias' node frame, the eager twin
    discovers it MID-CHAIN and returns None from the WHOLE fast path -- and this query then
    has no native polars lowering, so it raises. A fused plan built before that check would
    have silently answered instead. The guard is hoisted ahead of plan construction; this
    test pins the RAISE, and pins that the lane was reached and declined."""
    nodes, edges = _base_data()
    graph = _graph(engine, nodes, edges)
    query = ("MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.nope AS x, "
             "count(*) AS n ORDER BY n DESC, x ASC")

    calls = _probe_fused(monkeypatch)
    with pytest.raises(NotImplementedError):
        graph.gfql(query, engine=engine)

    assert calls == [False], "missing property column must DECLINE the fused lane"


# ------------------------------------------------------------- unit-level decline surface

def test_grouped_aggregate_fused_polars_declines_non_eager_polars_frames() -> None:
    """LazyFrame / non-polars inputs belong to the eager twin: schema probes on a LazyFrame
    warn and cost, and a pandas frame has no polars ``.lazy()``."""
    pl = _require_polars()
    fused = gfql_fast_paths_module._single_hop_grouped_aggregate_fused_polars
    nodes, edges = _base_data()
    pl_nodes = pl.from_pandas(nodes)
    pl_edges = pl.from_pandas(edges)
    kwargs: Dict[str, Any] = dict(
        node_col="id", src_col="s", dst_col="d", start_alias="p", end_alias="c",
        needed_by_alias={"p": [], "c": [("city", "city")]},
        group_keys=["city"], agg_specs=[("n", "count", None)],
        order_keys=[("n", True), ("city", False)], limit_value=None,
    )

    served = fused(pl_nodes, pl_nodes, pl_edges, **kwargs)
    assert served is not None and served.columns == ["city", "n"]

    assert fused(pl_nodes.lazy(), pl_nodes, pl_edges, **kwargs) is None
    assert fused(pl_nodes, pl_nodes.lazy(), pl_edges, **kwargs) is None
    assert fused(pl_nodes, pl_nodes, pl_edges.lazy(), **kwargs) is None
    assert fused(nodes, nodes, edges, **kwargs) is None


def test_grouped_aggregate_fused_polars_declines_degenerate_and_colliding_bindings() -> None:
    """Unit-level decline surface: source and destination bound to the SAME edge column (the
    twin's ``select([src, dst])`` cannot name a column twice), a projected column colliding
    with an endpoint column or with the internal lookup key, an untranslatable aggregate, and
    a missing property column."""
    pl = _require_polars()
    fused = gfql_fast_paths_module._single_hop_grouped_aggregate_fused_polars
    nodes, edges = _base_data()
    pl_nodes = pl.from_pandas(nodes)
    pl_edges = pl.from_pandas(edges)

    def call(**overrides: Any) -> Optional[Any]:
        kwargs: Dict[str, Any] = dict(
            node_col="id", src_col="s", dst_col="d", start_alias="p", end_alias="c",
            needed_by_alias={"p": [], "c": [("city", "city")]},
            group_keys=["city"], agg_specs=[("n", "count", None)],
            order_keys=[("n", True), ("city", False)], limit_value=None,
        )
        kwargs.update(overrides)
        return fused(pl_nodes, pl_nodes, pl_edges, **kwargs)

    assert call() is not None
    assert call(src_col="s", dst_col="s") is None, "src == dst binding must decline"
    assert call(needed_by_alias={"p": [], "c": [("s", "city")]},
                group_keys=["s"], order_keys=[("n", True), ("s", False)]) is None
    assert call(needed_by_alias={"p": [], "c": [("__gfql_t3_c_id__", "city")]},
                group_keys=["__gfql_t3_c_id__"],
                order_keys=[("n", True), ("__gfql_t3_c_id__", False)]) is None
    assert call(needed_by_alias={"p": [], "c": [("city", "absent_column")]}) is None
    assert call(agg_specs=[("n", "stddev", "city")]) is None
    assert call(agg_specs=[("n", "avg", None)]) is None
    assert call(order_keys=[]) is None
    assert call(order_keys=[("n", True)]) is None, "ORDER BY must name every group key"


def test_grouped_aggregate_fused_polars_supports_min_and_max_through_the_ast_surface() -> None:
    """``min``/``max`` over an alias property are declined by the CYPHER LOWERING (multi-source
    residual, #1273), so they never reach this lane from cypher today -- but the fast path is
    reachable from a hand-built chain, so the translator covers them and is pinned here."""
    pl = _require_polars()
    fused = gfql_fast_paths_module._single_hop_grouped_aggregate_fused_polars
    nodes, edges = _base_data()
    result = fused(
        pl.from_pandas(nodes), pl.from_pandas(nodes), pl.from_pandas(edges),
        node_col="id", src_col="s", dst_col="d", start_alias="p", end_alias="c",
        needed_by_alias={"p": [("age", "age")], "c": [("city", "city")]},
        group_keys=["city"],
        agg_specs=[("lo", "min", "age"), ("hi", "max", "age")],
        order_keys=[("city", False)], limit_value=None,
    )
    assert result is not None
    assert result.to_dicts() == [
        {"city": "LA", "lo": 20.0, "hi": 50.0},
        {"city": "NY", "lo": 30.0, "hi": 40.0},
        {"city": "SF", "lo": 50.0, "hi": 50.0},
    ]


# ------------------------------------------------------------------ degenerate bindings

@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_grouped_aggregate_fused_polars_node_key_named_like_the_source_column(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node key may share its name with the edge SOURCE column -- that name then appears
    on both sides of the semi-join and of the property lookup."""
    nodes, edges = _base_data()
    nodes = nodes.rename(columns={"id": "s"})
    oracle = _records(
        _graph("pandas", nodes, edges, node_key="s").gfql(Q_COUNT_STAR, engine="pandas"))

    calls = _probe_fused(monkeypatch)
    result = _records(
        _graph(engine, nodes, edges, node_key="s").gfql(Q_COUNT_STAR, engine=engine))

    assert calls == [True]
    assert result == oracle


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_grouped_aggregate_fused_polars_empty_match_returns_no_groups(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty match produces no GROUPS (unlike a bare ``count(*)``, which openCypher counts
    as 0 over no rows -- that shape is served by a different fast path)."""
    nodes, edges = _empty_edges_data()
    oracle = _records(_graph("pandas", nodes, edges).gfql(Q_COUNT_STAR, engine="pandas"))

    calls = _probe_fused(monkeypatch)
    result = _records(_graph(engine, nodes, edges).gfql(Q_COUNT_STAR, engine=engine))

    assert calls == [True]
    assert result == oracle == (["city", "n"], [])


def _null_group_key_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One edge lands on a PERSON row, whose ``city`` is null -- so the grouped result really
    carries a NULL group key and the null-placement rule becomes observable."""
    nodes, edges = _base_data()
    extra = pd.DataFrame({"s": [1, 2], "d": [2, 3], "rel": ["L", "L"]})
    return nodes, pd.concat([edges, extra], ignore_index=True)


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("query,expected_first_city", [
    ("MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, count(*) AS n ORDER BY city ASC", "LA"),
    ("MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, count(*) AS n ORDER BY city DESC", None),
])
def test_grouped_aggregate_fused_polars_places_nulls_the_opencypher_way(
    engine: str, query: str, expected_first_city: Optional[str],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """openCypher orders NULL as the LARGEST value: last on ASC, first on DESC. polars
    defaults nulls-first, so the twin pins ``nulls_last`` per key and the fused plan must
    carry the same pin -- pinned here on a group key that is null for the person rows."""
    nodes, edges = _null_group_key_data()
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))

    calls = _probe_fused(monkeypatch)
    result = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    assert calls == [True]
    assert result == oracle
    assert result[1][0]["city"] == expected_first_city


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_grouped_aggregate_fused_polars_null_aggregate_value_ordering(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same null-largest rule on the AGGREGATE column, where the null comes from averaging an
    all-null group -- and with LIMIT, so getting it wrong returns a different ROW."""
    nodes, edges = _null_group_key_data()
    query = ("MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, avg(p.age) AS a "
             "ORDER BY a ASC, city ASC LIMIT 2")
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))

    calls = _probe_fused(monkeypatch)
    result = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    assert calls == [True]
    assert result == oracle


# ------------------------------------------------------- the benchmark shapes themselves

_GB_SHAPES: List[Tuple[str, str]] = [
    ("q1", "MATCH (f {node_type:'Person'})-[{rel:'FOLLOWS'}]->(p {node_type:'Person'}) "
           "RETURN p.node_id AS personID, count(f) AS numFollowers "
           "ORDER BY numFollowers DESC, personID ASC LIMIT 3"),
    ("q3", "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
           "WHERE c.country = 'United States' "
           "RETURN c.city AS city, avg(p.age) AS averageAge ORDER BY averageAge ASC, city ASC LIMIT 5"),
    ("q4", "MATCH (p {node_type:'Person'})-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
           "WHERE p.age >= 30 AND p.age <= 40 "
           "RETURN c.country AS countries, count(*) AS personCounts "
           "ORDER BY personCounts DESC, countries ASC LIMIT 3"),
]


def _gb_shaped_graph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A miniature of the graph-benchmark schema the q1/q3/q4 lane measures."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3, 10, 11, 12],
        "node_id": [0, 1, 2, 3, 10, 11, 12],
        "node_type": ["Person"] * 4 + ["City"] * 3,
        "age": [25, 33, 38, 41, None, None, None],
        "city": [None, None, None, None, "Austin", "Boston", "Lyon"],
        "country": [None, None, None, None, "United States", "United States", "France"],
    })
    edges = pd.DataFrame({
        "s": [0, 1, 2, 3, 0, 1, 2, 3, 0],
        "d": [1, 2, 3, 0, 10, 10, 11, 12, 2],
        "rel": ["FOLLOWS", "FOLLOWS", "FOLLOWS", "FOLLOWS",
                "LIVES_IN", "LIVES_IN", "LIVES_IN", "LIVES_IN", "FOLLOWS"],
    })
    return nodes, edges


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("label,query", _GB_SHAPES, ids=[s[0] for s in _GB_SHAPES])
def test_grouped_aggregate_fused_polars_serves_the_graph_benchmark_shapes(
    engine: str, label: str, query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STRUCTURAL LOCK-IN for the three benchmark cells this lane exists to move: q1, q3 and
    q4 must be SERVED (not merely fast) and must answer what pandas answers."""
    nodes, edges = _gb_shaped_graph()
    oracle = _records(_graph("pandas", nodes, edges).gfql(query, engine="pandas"))

    calls = _probe_fused(monkeypatch)
    result = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))

    assert calls == [True], f"{label}: the benchmark shape must be served by the fused lane"
    assert result == oracle


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_grouped_aggregate_fused_polars_benchmark_shapes_match_the_eager_twin(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same three shapes, differential against the eager twin over every fixture graph."""
    for (label, query), (graph_name, data_fn) in itertools.product(_GB_SHAPES, _GRAPHS.items()):
        nodes, edges = data_fn()
        if "node_id" not in nodes.columns:
            nodes = nodes.assign(node_id=nodes["id"], node_type=nodes["kind"])
            nodes = nodes.assign(node_type=nodes["node_type"].map({"P": "Person", "C": "City"}))
            edges = edges.assign(rel=edges["rel"].map({"L": "LIVES_IN", "X": "FOLLOWS"}))
        with monkeypatch.context() as eager_ctx:
            _force_eager(eager_ctx)
            eager = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))
        fused = _records(_graph(engine, nodes, edges).gfql(query, engine=engine))
        assert fused == eager, f"{graph_name}/{label}: fused diverged from the eager twin"
