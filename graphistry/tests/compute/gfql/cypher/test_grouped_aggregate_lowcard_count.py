"""LOW-CARDINALITY PURE-``count(*)`` FORMULATION GATE --
``gfql_fast_paths._low_cardinality_pure_count_plan``.

Inside the fused single-hop grouped-aggregate lane, a single-key pure ``count(*)`` has a
second, value-identical polars formulation: ``value_counts`` instead of
``group_by(maintain_order=True).agg(pl.len())``. It is not a drop-in. Measured on
dgx-spark (polars 1.35.2, 20 threads, interleaved, 90 samples/arm/cell):

* ``group_by`` carries a FLAT ~2 ms coordination cost that exists only at LOW group
  cardinality and vanishes between 32 and 64 groups;
* ``value_counts`` has no such cost but scales WORSE with input rows -- at 1,000,000 rows
  it loses even at 2 groups (4.137 ms -> 8.591 ms).

So the choice is a ROUTING decision under two static bounds, and BOTH are needed: applied
ungated, the same formulation makes the graph-benchmark q1 cell (~20,000 groups over
~200,000 rows) 2.7 ms SLOWER at 20k and 8.6 ms slower at 100k.

WHAT THESE TESTS PIN:

* **The bounds are UPPER bounds.** ``test_admitted_shapes_respect_the_measured_bounds``
  reaches into the lane's own work frame on every admission across the whole graph x shape
  corpus and asserts the REALIZED group cardinality and input rows are inside the
  thresholds. That is the soundness claim itself, checked against data rather than argued:
  an under-estimating bound would route a big aggregate into the slow formulation.
* **Which shapes are admitted and which are DECLINED**, enumerated, including the ones
  that decline for a REASON THAT IS NOT CARDINALITY (a second property-bearing alias
  breaks the row bound's derivation; duplicate node ids break it too).
* **Value identity against the unmodified product.** Every comparison runs the same query
  twice -- once with the gate live, once with it forced to decline, which IS the
  pre-change code -- row-order and column-order sensitively, plus dtypes, plus a pandas
  oracle.
* **Engine reach**: pandas and cudf must never enter the lane at all; polars and
  polars-gpu must.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
import graphistry.compute.gfql_fast_paths as gfql_fast_paths_module


MAX_GROUPS = gfql_fast_paths_module._LOWCARD_COUNT_MAX_GROUPS
MAX_INPUT_ROWS = gfql_fast_paths_module._LOWCARD_COUNT_MAX_INPUT_ROWS


# --------------------------------------------------------------------------- fixtures

def _base_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Four persons, four cities, one non-matching edge label. The CITY frame is the
    group-key side and has 4 rows, comfortably inside ``MAX_GROUPS``."""
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


def _null_group_key_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    nodes = nodes.copy()
    nodes.loc[nodes["id"].isin([6, 7]), "city"] = None
    return nodes, edges


def _all_null_group_key_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    nodes = nodes.copy()
    nodes["city"] = None
    return nodes, edges


def _empty_edges_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return nodes, edges.iloc[:0].copy()


def _no_matching_nodes_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return nodes.assign(kind="Z"), edges


def _string_ids_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    return (
        nodes.assign(id=[f"n{i}" for i in nodes["id"]]),
        edges.assign(s=[f"n{i}" for i in edges["s"]], d=[f"n{i}" for i in edges["d"]]),
    )


def _self_loops_parallel_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    extra = pd.DataFrame({"s": [5, 1, 1, 6], "d": [5, 5, 5, 6], "rel": ["L", "L", "L", "L"]})
    return nodes, pd.concat([edges, extra], ignore_index=True)


def _dangling_endpoints_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_data()
    extra = pd.DataFrame({"s": [99, 1], "d": [5, 98], "rel": ["L", "L"]})
    return nodes, pd.concat([edges, extra], ignore_index=True)


def _dup_end_node_rows_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The CITY frame carries an id twice. The lane deliberately does not dedup property
    lookups, so this MULTIPLIES matched rows -- which is exactly what breaks the
    ``rows <= edge frame height`` derivation, so the gate must decline it."""
    nodes, edges = _base_data()
    return pd.concat([nodes, nodes.iloc[[4, 5]]], ignore_index=True), edges


def _dup_start_node_rows_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicates on the PERSON side. That side contributes no property column for the
    q4 shape, so it only feeds a semi-join and cannot multiply: the gate still admits."""
    nodes, edges = _base_data()
    return pd.concat([nodes, nodes.iloc[[0, 1]]], ignore_index=True), edges


def _wide_group_key_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``MAX_GROUPS + 1`` cities: the O(1) height bound cannot certify low cardinality even
    though every city here shares ONE country. This is the q1-shaped decline in miniature."""
    n_cities = MAX_GROUPS + 1
    # DTYPES ARE EXPLICIT ON PURPOSE. Letting ``age`` fall out as an OBJECT column of ints
    # and Nones makes cudf raise MixedTypeError on ingest -- a fixture defect that a
    # CPU-only run cannot see, and that showed up only in the RAPIDS image.
    persons = pd.DataFrame({
        "id": np.arange(1, 21, dtype="int64"),
        "kind": "P",
        "age": np.arange(20, 40, dtype="float64"),
        "city": pd.Series([None] * 20, dtype="object"),
        "country": pd.Series([None] * 20, dtype="object"),
    })
    cities = pd.DataFrame({
        "id": np.arange(1000, 1000 + n_cities, dtype="int64"),
        "kind": "C",
        "age": np.full(n_cities, np.nan, dtype="float64"),
        "city": pd.Series([f"city{i}" for i in range(n_cities)], dtype="object"),
        "country": pd.Series(["US"] * n_cities, dtype="object"),
    })
    rng = np.random.default_rng(3)
    edges = pd.DataFrame({
        "s": rng.integers(1, 21, 200).astype("int64"),
        "d": rng.integers(1000, 1000 + n_cities, 200).astype("int64"),
        "rel": "L",
    })
    return pd.concat([persons, cities], ignore_index=True), edges


def _exactly_max_groups_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``MAX_GROUPS`` cities exactly -- the last admitted height."""
    nodes, edges = _wide_group_key_data()
    keep = nodes["kind"].ne("C") | nodes["id"].lt(1000 + MAX_GROUPS)
    nodes = nodes[keep].reset_index(drop=True)
    edges = edges[edges["d"] < 1000 + MAX_GROUPS].reset_index(drop=True)
    return nodes, edges


def _many_edges_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``MAX_INPUT_ROWS + 1`` edges over a 3-city frame: cardinality is tiny, the ROW bound
    is what declines it."""
    n_persons = 50
    persons = pd.DataFrame({
        "id": np.arange(1, n_persons + 1, dtype="int64"),
        "kind": "P",
        "age": (np.arange(20, 20 + n_persons, dtype="float64") % 60),
        "city": pd.Series([None] * n_persons, dtype="object"),
        "country": pd.Series([None] * n_persons, dtype="object"),
    })
    cities = pd.DataFrame({
        "id": np.array([1000, 1001, 1002], dtype="int64"),
        "kind": "C",
        "age": np.full(3, np.nan, dtype="float64"),
        "city": pd.Series(["LA", "NY", "SF"], dtype="object"),
        "country": pd.Series(["US", "US", "MX"], dtype="object"),
    })
    m = MAX_INPUT_ROWS + 1
    rng = np.random.default_rng(5)
    edges = pd.DataFrame({
        "s": rng.integers(1, n_persons + 1, m).astype("int64"),
        "d": rng.integers(1000, 1003, m).astype("int64"),
        "rel": "L",
    })
    return pd.concat([persons, cities], ignore_index=True), edges


_GRAPHS: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]] = {
    "base": _base_data,
    "null_group_key": _null_group_key_data,
    "all_null_group_key": _all_null_group_key_data,
    "empty_edges": _empty_edges_data,
    "no_matching_nodes": _no_matching_nodes_data,
    "string_ids": _string_ids_data,
    "self_loops_parallel": _self_loops_parallel_data,
    "dangling_endpoints": _dangling_endpoints_data,
    "dup_end_node_rows": _dup_end_node_rows_data,
    "dup_start_node_rows": _dup_start_node_rows_data,
    "exactly_max_groups": _exactly_max_groups_data,
    "wide_group_key": _wide_group_key_data,
}


# ------------------------------------------------------------------------- shape corpus

# ADMITTED on the `base`-shaped graphs: single group key, pure count(*), only the group-key
# alias carries a property, and the fused lane's own order-totality gate is satisfied.
Q_COUNT_STAR = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC LIMIT 3")
Q_COUNT_STAR_NO_LIMIT = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC")
Q_COUNT_ALIAS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(p) AS n "
    "ORDER BY n DESC, city ASC LIMIT 3")
Q_GROUP_KEY_DESC = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.country AS co, count(*) AS n "
    "ORDER BY co DESC")
Q_WHERE_ON_START = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) WHERE p.age >= 30 "
    "RETURN c.city AS city, count(*) AS n ORDER BY n DESC, city ASC")
Q_NO_EDGE_FILTER = (
    "MATCH (p {kind:'P'})-[]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC, city ASC")
Q_UNFILTERED_ENDS = (
    "MATCH (p)-[{rel:'L'}]->(c) RETURN c.city AS city, count(*) AS n ORDER BY n DESC, city ASC")

_ADMITTED_SHAPES: List[Tuple[str, str]] = [
    ("count_star_limit", Q_COUNT_STAR),
    ("count_star_no_limit", Q_COUNT_STAR_NO_LIMIT),
    ("count_alias", Q_COUNT_ALIAS),
    ("group_key_desc", Q_GROUP_KEY_DESC),
    ("where_on_start", Q_WHERE_ON_START),
    ("no_edge_filter", Q_NO_EDGE_FILTER),
    ("unfiltered_ends", Q_UNFILTERED_ENDS),
]

# DECLINED by the gate, for reasons that are NOT the two thresholds. The fused lane still
# serves each of these -- through the unchanged group_by formulation.
Q_AVG = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) WHERE c.country = 'US' "
    "RETURN c.city AS city, avg(p.age) AS a ORDER BY a ASC, city ASC LIMIT 5")
Q_SUM = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, sum(p.age) AS s "
    "ORDER BY s DESC, city ASC")
Q_COUNT_PROP = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(p.age) AS n "
    "ORDER BY n DESC, city ASC")
Q_TWO_GROUP_KEYS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.country AS co, c.city AS city, "
    "count(*) AS n ORDER BY n DESC, co ASC, city ASC")
Q_TWO_AGGS = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n, "
    "avg(p.age) AS a ORDER BY city ASC")
Q_SECOND_ALIAS_PROP = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) WHERE p.age >= 0 "
    "RETURN c.city AS city, count(p.age) AS n, count(*) AS m ORDER BY city ASC")
Q_GROUP_START_PROP = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN p.age AS age, count(*) AS n "
    "ORDER BY age ASC")

_DECLINED_SHAPES: List[Tuple[str, str]] = [
    ("avg_not_a_count", Q_AVG),
    ("sum_not_a_count", Q_SUM),
    ("count_over_property", Q_COUNT_PROP),
    ("two_group_keys", Q_TWO_GROUP_KEYS),
    ("two_aggregates", Q_TWO_AGGS),
    ("second_alias_carries_a_property", Q_SECOND_ALIAS_PROP),
]

_ALL_SHAPES = _ADMITTED_SHAPES + _DECLINED_SHAPES + [("group_start_prop", Q_GROUP_START_PROP)]

# Shapes whose ORDER BY is not total over the output rows: the FUSED lane declines them
# before the gate is ever consulted, so the gate must record zero calls.
Q_PARTIAL_ORDER = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n "
    "ORDER BY n DESC")
Q_NO_ORDER = (
    "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS city, count(*) AS n")


# ------------------------------------------------------------------------------ helpers

def _require_polars() -> Any:
    return pytest.importorskip("polars")


def _require_polars_gpu() -> None:
    pl = _require_polars()
    pytest.importorskip("cudf_polars")
    try:
        pl.DataFrame({"a": [1, 2]}).lazy().filter(pl.col("a") > 0).collect(
            engine=pl.GPUEngine(raise_on_fail=True)
        )
    except Exception as exc:  # pragma: no cover - CPU-only CI
        pytest.skip(f"cudf_polars installed but the GPU collect probe failed: {exc}")


def _graph(engine: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Plottable:
    if engine == "pandas":
        return graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        try:
            _ = cudf.Series([1, 2, 3])
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"cudf installed but the runtime is unavailable: {exc}")
        return graphistry.nodes(cudf.from_pandas(nodes_df), "id").edges(
            cudf.from_pandas(edges_df), "s", "d")
    if engine == "polars-gpu":
        _require_polars_gpu()
    pl = _require_polars()
    return graphistry.nodes(pl.from_pandas(nodes_df), "id").edges(
        pl.from_pandas(edges_df), "s", "d")


def _engine_arg(engine: str) -> str:
    return "polars" if engine == "polars-gpu" else engine


def _records(result: Plottable) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Row-order AND column-order sensitive comparison value, plus the DTYPE list.

    The dtypes travel because the two formulations produce the count column independently
    (``pl.len()`` vs ``value_counts``' own field) -- an equal-valued column at a different
    width would be a real divergence that record equality alone cannot see."""
    df = result._nodes
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    rows = [
        {k: (None if v is None or (isinstance(v, float) and v != v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    return [str(c) for c in df.columns], [str(d) for d in df.dtypes], rows


def _probe_gate(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """One entry per gate CALL: True=admitted, False=declined. Empty means the fused lane
    never got as far as consulting it."""
    calls: List[bool] = []
    original = gfql_fast_paths_module._low_cardinality_pure_count_plan

    def probe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(gfql_fast_paths_module, "_low_cardinality_pure_count_plan", probe)
    return calls


def _force_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the gate off entirely. What runs then IS the unmodified product."""
    monkeypatch.setattr(
        gfql_fast_paths_module,
        "_low_cardinality_pure_count_plan",
        lambda *args, **kwargs: None,
    )


def _realized(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, int]]:
    """On every ADMISSION, collect the lane's own work frame and record the REALIZED input
    rows and group cardinality -- the two quantities the static bounds claim to bound."""
    seen: List[Dict[str, int]] = []
    original = gfql_fast_paths_module._low_cardinality_pure_count_plan

    def probe(work_lf: Any, **kwargs: Any) -> Any:
        result = original(work_lf, **kwargs)
        if result is not None:
            group_key = list(kwargs["group_keys"])[0]
            frame = work_lf.collect()
            seen.append({
                "rows": frame.height,
                "cardinality": frame.get_column(group_key).n_unique(),
                "edge_rows": int(kwargs["edge_rows"]),
            })
        return result

    monkeypatch.setattr(gfql_fast_paths_module, "_low_cardinality_pure_count_plan", probe)
    return seen


# ------------------------------------------------------------------- the soundness claim

@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
@pytest.mark.parametrize("shape_name,query", _ALL_SHAPES)
def test_admitted_shapes_respect_the_measured_bounds(
    monkeypatch: pytest.MonkeyPatch, graph_name: str, shape_name: str, query: str
) -> None:
    """THE SOUNDNESS TEST. Whenever the gate admits, the REALIZED group cardinality must be
    <= MAX_GROUPS and the REALIZED aggregate input rows <= MAX_INPUT_ROWS.

    The bounds are static and O(1) (a node-frame height and an edge-frame height), so this
    is where the claim that they are UPPER bounds gets checked against data instead of
    argued. An under-estimating bound shows up here as an admission whose realized numbers
    are outside the thresholds -- i.e. a shape routed into the formulation the crossover
    sweep says is the slower one."""
    _require_polars()
    nodes_df, edges_df = _GRAPHS[graph_name]()
    graph = _graph("polars", nodes_df, edges_df)
    realized = _realized(monkeypatch)
    graph.gfql(query, engine="polars")
    for entry in realized:
        assert entry["cardinality"] <= MAX_GROUPS, (
            f"{graph_name}/{shape_name}: admitted with {entry['cardinality']} groups, over "
            f"the committed bound of {MAX_GROUPS} -- the cardinality bound UNDER-estimated"
        )
        assert entry["rows"] <= MAX_INPUT_ROWS, (
            f"{graph_name}/{shape_name}: admitted with {entry['rows']} aggregate input "
            f"rows, over the committed bound of {MAX_INPUT_ROWS}"
        )
        assert entry["rows"] <= entry["edge_rows"], (
            f"{graph_name}/{shape_name}: the property join MULTIPLIED rows "
            f"({entry['edge_rows']} edges -> {entry['rows']} rows), so the edge-height row "
            "bound does not hold and this shape should have declined"
        )


# ------------------------------------------------------- value identity (differential)

@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf", "polars-gpu"])
@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
@pytest.mark.parametrize("shape_name,query", _ALL_SHAPES)
def test_gate_is_value_identical_to_the_unmodified_product(
    monkeypatch: pytest.MonkeyPatch, engine: str, graph_name: str, shape_name: str, query: str
) -> None:
    """Differential against the gate forced OFF, which is byte-for-byte the pre-change code
    path. Row order, column order, dtypes and values all travel."""
    nodes_df, edges_df = _GRAPHS[graph_name]()
    graph = _graph(engine, nodes_df, edges_df)
    live = _records(graph.gfql(query, engine=_engine_arg(engine)))

    with monkeypatch.context() as ctx:
        _force_decline(ctx)
        baseline = _records(graph.gfql(query, engine=_engine_arg(engine)))

    assert live == baseline, f"{engine}/{graph_name}/{shape_name} diverged from the gate-off product"


@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
@pytest.mark.parametrize("shape_name,query", _ALL_SHAPES)
def test_polars_matches_the_pandas_oracle(graph_name: str, shape_name: str, query: str) -> None:
    """A second, independent reference: the pandas branch never enters this lane at all.

    The two duplicate-id graphs are excluded from the VALUE comparison because the polars
    branch's property lookup is not deduplicated by node id while the pandas branch's is --
    a pre-existing divergence disclosed by #1823, not something this gate introduces. It
    bites on whichever arm carries a property column, hence BOTH fixtures. The gate-off
    differential above still covers them, and it is the comparison that matters here."""
    _require_polars()
    if graph_name in {"dup_end_node_rows", "dup_start_node_rows"}:
        pytest.skip("pre-existing polars-vs-pandas dedup divergence, disclosed in #1823")
    nodes_df, edges_df = _GRAPHS[graph_name]()
    _, _, pandas_rows = _records(_graph("pandas", nodes_df, edges_df).gfql(query, engine="pandas"))
    _, _, polars_rows = _records(_graph("polars", nodes_df, edges_df).gfql(query, engine="polars"))
    assert polars_rows == pandas_rows, f"{graph_name}/{shape_name} polars != pandas"


# -------------------------------------------------------------------------- engagement

@pytest.mark.parametrize("shape_name,query", _ADMITTED_SHAPES)
def test_admitted_shapes_are_admitted(
    monkeypatch: pytest.MonkeyPatch, shape_name: str, query: str
) -> None:
    _require_polars()
    nodes_df, edges_df = _base_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(query, engine="polars")
    assert calls == [True], f"{shape_name} should be admitted, got {calls}"


@pytest.mark.parametrize("shape_name,query", _DECLINED_SHAPES)
def test_declined_shapes_are_declined(
    monkeypatch: pytest.MonkeyPatch, shape_name: str, query: str
) -> None:
    """These reach the gate and are turned away -- the fused lane still answers them, via
    the unchanged group_by. A decline is the safe outcome, never a wrong one."""
    _require_polars()
    nodes_df, edges_df = _base_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(query, engine="polars")
    assert calls == [False], f"{shape_name} should be declined, got {calls}"


def test_cardinality_bound_declines_a_wide_group_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_GROUPS + 1 city rows. Every one of them carries country 'US', so the TRUE
    cardinality of ``c.country`` is 1 -- the O(1) height bound cannot see that, and the
    resulting decline is the deliberate cost of not paying for an exact count."""
    _require_polars()
    nodes_df, edges_df = _wide_group_key_data()
    calls = _probe_gate(monkeypatch)
    result = _graph("polars", nodes_df, edges_df).gfql(Q_GROUP_KEY_DESC, engine="polars")
    assert calls == [False]
    assert len(result._nodes) == 1, "the true cardinality really was 1; the bound was loose"


def test_cardinality_bound_admits_exactly_max_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary is inclusive: a node frame of exactly MAX_GROUPS rows is admitted."""
    _require_polars()
    nodes_df, edges_df = _exactly_max_groups_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(Q_GROUP_KEY_DESC, engine="polars")
    assert calls == [True]


def test_row_bound_declines_a_large_edge_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAX_INPUT_ROWS + 1 edges over a 3-city frame: cardinality is tiny, so only the ROW
    bound can decline this. It must."""
    _require_polars()
    nodes_df, edges_df = _many_edges_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(Q_GROUP_KEY_DESC, engine="polars")
    assert calls == [False]


def test_row_bound_admits_exactly_max_input_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_polars()
    nodes_df, edges_df = _many_edges_data()
    edges_df = edges_df.iloc[:MAX_INPUT_ROWS].reset_index(drop=True)
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(Q_GROUP_KEY_DESC, engine="polars")
    assert calls == [True]


def test_duplicate_group_alias_node_ids_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate ids in the group-key alias frame make the property join MULTIPLY rows, so
    ``rows <= edge frame height`` stops holding. The gate declines rather than reason about
    a bound it can no longer prove."""
    _require_polars()
    nodes_df, edges_df = _dup_end_node_rows_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(Q_COUNT_STAR, engine="polars")
    assert calls == [False]


def test_duplicate_other_alias_node_ids_still_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicates on the side that contributes NO property column feed only a semi-join,
    which cannot multiply -- so the bound survives and the gate admits. Kept separate from
    the test above so a change that collapses the two sides is visible."""
    _require_polars()
    nodes_df, edges_df = _dup_start_node_rows_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(Q_COUNT_STAR, engine="polars")
    assert calls == [True]


@pytest.mark.parametrize("shape_name,query", [
    ("partial_order", Q_PARTIAL_ORDER),
    ("no_order_by", Q_NO_ORDER),
])
def test_fused_lane_order_gate_runs_first(
    monkeypatch: pytest.MonkeyPatch, shape_name: str, query: str
) -> None:
    """The fused lane's own order-totality gate declines these BEFORE any plan is built, so
    the low-cardinality gate is never consulted. Recorded so a reordering that consulted it
    first -- and thereby changed which shapes the fused lane can serve -- is visible."""
    _require_polars()
    nodes_df, edges_df = _base_data()
    calls = _probe_gate(monkeypatch)
    _graph("polars", nodes_df, edges_df).gfql(query, engine="polars")
    assert calls == [], f"{shape_name} reached the gate; it should not have"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_non_polars_engines_never_reach_the_gate(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    nodes_df, edges_df = _base_data()
    graph = _graph(engine, nodes_df, edges_df)
    calls = _probe_gate(monkeypatch)
    graph.gfql(Q_COUNT_STAR, engine=engine)
    assert calls == [], f"{engine} reached a polars-only gate"


def test_polars_gpu_engine_reaches_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """polars-gpu shares the CPU-collected fused lane, so it is admitted identically. A
    SKIP here is a coverage boundary, not a pass."""
    nodes_df, edges_df = _base_data()
    graph = _graph("polars-gpu", nodes_df, edges_df)
    calls = _probe_gate(monkeypatch)
    graph.gfql(Q_COUNT_STAR, engine="polars")
    assert calls == [True]


# ------------------------------------------------------- unit contract of the gate itself

def _gate(
    *,
    group_keys: Sequence[str] = ("city",),
    agg_specs: Sequence[Tuple[str, str, Optional[str]]] = (("n", "count", None),),
    needed_by_alias: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    frames: Optional[Dict[str, Any]] = None,
    edge_rows: int = 10,
    node_col: str = "id",
) -> Any:
    pl = _require_polars()
    if needed_by_alias is None:
        needed_by_alias = {"p": [], "c": [("city", "city")]}
    if frames is None:
        frames = {
            "p": pl.DataFrame({"id": [1, 2, 3]}),
            "c": pl.DataFrame({"id": [10, 11], "city": ["LA", "NY"]}),
        }
    work = pl.DataFrame({"city": ["LA", "NY", "LA"]}).lazy()
    return gfql_fast_paths_module._low_cardinality_pure_count_plan(
        work,
        node_col=node_col,
        group_keys=group_keys,
        agg_specs=agg_specs,
        needed_by_alias=needed_by_alias,
        frames_by_alias=frames,
        edge_rows=edge_rows,
    )


def test_gate_unit_admits_the_canonical_shape() -> None:
    assert _gate() is not None


@pytest.mark.parametrize("case,kwargs", [
    ("two_group_keys", {"group_keys": ("city", "country")}),
    ("no_group_keys", {"group_keys": ()}),
    ("two_aggregates", {"agg_specs": (("n", "count", None), ("m", "count", None))}),
    ("no_aggregates", {"agg_specs": ()}),
    ("avg_aggregate", {"agg_specs": (("a", "avg", "age"),)}),
    # The next two are UNREACHABLE from the cypher surface -- the fast path refuses a
    # non-count aggregate without an expression alias long before the fused lane is built
    # -- but the gate is a standalone function and its contract is checked here, not
    # inherited. Mutation testing found this: dropping ``func != "count"`` from the guard
    # SURVIVED the whole suite until these two cases existed.
    ("avg_without_an_expression", {"agg_specs": (("a", "avg", None),)}),
    ("sum_without_an_expression", {"agg_specs": (("s", "sum", None),)}),
    ("count_over_property", {"agg_specs": (("n", "count", "age"),)}),
    ("out_alias_equals_group_key", {"agg_specs": (("city", "count", None),)}),
    ("edge_rows_over_bound", {"edge_rows": MAX_INPUT_ROWS + 1}),
])
def test_gate_unit_declines(case: str, kwargs: Dict[str, Any]) -> None:
    assert _gate(**kwargs) is None, f"{case} should decline"


def test_gate_unit_declines_when_the_group_key_has_no_owning_alias() -> None:
    assert _gate(needed_by_alias={"p": [], "c": [("other", "city")]}) is None


def test_gate_unit_declines_when_two_aliases_own_the_group_key() -> None:
    """DISCLOSED: the ``len(owners) != 1`` guard is PROVABLY REDUNDANT against the check
    that immediately follows it, and mutation testing says so -- relaxing it to
    ``len(owners) < 1`` survives the entire suite. Two owners means a second alias with a
    NON-EMPTY property list, which the ``other alias carries properties`` check declines
    anyway, so no input can distinguish the two forms. The guard is kept because it states
    the precondition the height bound depends on; this test pins the OUTCOME, which is all
    that is observable."""
    pl = _require_polars()
    assert _gate(
        needed_by_alias={"p": [("city", "city")], "c": [("city", "city")]},
        frames={
            "p": pl.DataFrame({"id": [1, 2], "city": ["LA", "NY"]}),
            "c": pl.DataFrame({"id": [10, 11], "city": ["LA", "NY"]}),
        },
    ) is None


def test_gate_unit_declines_when_the_other_alias_carries_properties() -> None:
    pl = _require_polars()
    assert _gate(
        needed_by_alias={"p": [("age", "age")], "c": [("city", "city")]},
        frames={
            "p": pl.DataFrame({"id": [1, 2], "age": [3, 4]}),
            "c": pl.DataFrame({"id": [10, 11], "city": ["LA", "NY"]}),
        },
    ) is None


def test_gate_unit_declines_a_tall_owner_frame() -> None:
    pl = _require_polars()
    tall = MAX_GROUPS + 1
    assert _gate(frames={
        "p": pl.DataFrame({"id": [1]}),
        "c": pl.DataFrame({"id": list(range(tall)), "city": ["LA"] * tall}),
    }) is None


def test_gate_unit_admits_an_owner_frame_of_exactly_max_groups() -> None:
    pl = _require_polars()
    assert _gate(frames={
        "p": pl.DataFrame({"id": [1]}),
        "c": pl.DataFrame({"id": list(range(MAX_GROUPS)), "city": ["LA"] * MAX_GROUPS}),
    }) is not None


def test_gate_unit_declines_duplicate_owner_node_ids() -> None:
    pl = _require_polars()
    assert _gate(frames={
        "p": pl.DataFrame({"id": [1]}),
        "c": pl.DataFrame({"id": [10, 10], "city": ["LA", "NY"]}),
    }) is None


def test_gate_unit_declines_a_missing_node_id_column() -> None:
    pl = _require_polars()
    assert _gate(frames={
        "p": pl.DataFrame({"id": [1]}),
        "c": pl.DataFrame({"other": [10, 11], "city": ["LA", "NY"]}),
    }) is None


def test_gate_unit_declines_a_non_polars_owner_frame() -> None:
    assert _gate(frames={
        "p": pd.DataFrame({"id": [1]}),
        "c": pd.DataFrame({"id": [10, 11], "city": ["LA", "NY"]}),
    }) is None


def test_gate_unit_serves_a_group_key_literally_named_count() -> None:
    """``value_counts`` names its output column ``count`` by default, which would collide.
    The lane passes ``name=`` instead of renaming, so this shape is SERVED, and it must
    agree with the group_by twin."""
    pl = _require_polars()
    work = pl.DataFrame({"count": ["a", "b", "a"]}).lazy()
    plan = gfql_fast_paths_module._low_cardinality_pure_count_plan(
        work,
        node_col="id",
        group_keys=["count"],
        agg_specs=[("n", "count", None)],
        needed_by_alias={"p": [], "c": [("count", "label")]},
        frames_by_alias={
            "p": pl.DataFrame({"id": [1]}),
            "c": pl.DataFrame({"id": [10, 11], "label": ["a", "b"]}),
        },
        edge_rows=3,
    )
    assert plan is not None
    twin = work.group_by(["count"], maintain_order=True).agg(pl.len().alias("n")).collect()
    got = plan.collect()
    assert got.schema == twin.schema
    assert sorted(got.rows()) == sorted(twin.rows())


@pytest.mark.parametrize("values,dtype", [
    (["a", None, "b", None, "a"], "String"),
    ([None, None], "String"),
    ([], "String"),
    ([1, 2, 1, 3], "Int64"),
    ([True, False, True, None], "Boolean"),
    ([1.0, float("nan"), float("nan"), None, 1.0], "Float64"),
])
def test_gate_unit_matches_the_group_by_twin_on_awkward_keys(
    values: List[Any], dtype: str
) -> None:
    """Nulls, all-null, empty input, NaN-as-its-own-group and boolean keys: the two
    formulations must agree on rows AND on schema, or the routing decision would be a
    semantic one."""
    pl = _require_polars()
    work = pl.DataFrame({"city": values}, schema={"city": getattr(pl, dtype)}).lazy()
    plan = gfql_fast_paths_module._low_cardinality_pure_count_plan(
        work,
        node_col="id",
        group_keys=["city"],
        agg_specs=[("n", "count", None)],
        needed_by_alias={"p": [], "c": [("city", "city")]},
        frames_by_alias={
            "p": pl.DataFrame({"id": [1]}),
            "c": pl.DataFrame({"id": [10, 11], "city": ["LA", "NY"]}),
        },
        edge_rows=len(values),
    )
    assert plan is not None
    twin = work.group_by(["city"], maintain_order=True).agg(pl.len().alias("n")).collect()
    got = plan.collect()
    assert got.schema == twin.schema

    def canon(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        """NaN is its own group in BOTH formulations, but ``nan != nan`` would make the
        comparison below fail on agreement. Normalize it to a sentinel first."""
        normalized = [
            (("__nan__" if isinstance(r[0], float) and r[0] != r[0] else r[0]), r[1])
            for r in rows
        ]
        return sorted(normalized, key=lambda r: (r[0] is None, str(r[0]), r[1]))

    assert canon(got.rows()) == canon(twin.rows())
