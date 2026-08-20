"""Round-004 row-multiplicity + aggregate-identity pins (#1899).

openCypher RETURN/WITH operate on the BAG of pattern-match rows: a projection
of one endpoint keeps one row per match (never a deduplicated node set), an
ungrouped aggregate over an empty row stream still emits its identity row
(count -> 0, sum -> 0, collect -> []), and list subscripts index from the end
for negative keys.

Every expected value is HAND-COMPUTED on the fixture below (issue #1899);
engine agreement is not evidence. Fixture: nodes 1-5 (Ann/Bob/Cat/Dan/Eve,
null age/city cases), edges 1->2, 1->3, 2->3, 3->4.
"""
import math
from typing import Optional

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError  # noqa: F401  (negative controls)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import cudf  # noqa: F401
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
cudf_only = pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]
ALL_ENGINES = ENGINES + [pytest.param("cudf", marks=cudf_only)]


def _run(query: str, engine: str, edges: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Ann", "Bob", "Cat", "Dan", "Eve"],
        "age": [30.0, 40.0, 25.0, None, 35.0],
        "city": ["NYC", "NYC", "SF", "SF", None],
    })
    if edges is None:
        edges = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 3, 3, 4]})
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    elif engine == "cudf":
        import cudf as _cudf
        g = graphistry.nodes(_cudf.from_pandas(nodes), "id").edges(_cudf.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _scalar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if hasattr(v, "item"):
        try:
            return _scalar(v.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(v, (list, tuple)) or type(v).__name__ == "ndarray":
        return [_scalar(x) for x in v]
    return v


def _bag(df: pd.DataFrame, col: str):
    return sorted((_scalar(v) for v in df[col]), key=lambda x: (x is None, str(x)))


# ===========================================================================
# 1. Single-endpoint projection multiplicity (bag semantics)
# ===========================================================================
# Edge bag: (1,2),(1,3),(2,3),(3,4). a-side bag [1,1,2,3]; b-side [2,3,3,4].


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("MATCH (a)-->(b) RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) RETURN b.id AS id", [2, 3, 3, 4]),
    ("MATCH (a)-->() RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WITH a.id AS id RETURN id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WITH a RETURN a.id AS id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) WHERE b.id >= 3 RETURN a.id AS id", [1, 2, 3]),
], ids=["a_side", "b_side", "anon_endpoint", "with_property_stage", "with_pure_carry", "post_where"])
def test_single_endpoint_projection_keeps_multiplicity(query, expected, engine):
    """#1899 item 1: one row per pattern match -- node 1 has two out-edges, so
    its id appears twice; the old node-set projection silently deduplicated."""
    assert _bag(_run(query, engine), "id") == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_single_endpoint_distinct_still_dedupes(engine):
    """Negative control: RETURN DISTINCT is the user-requested dedupe."""
    assert _bag(_run("MATCH (a)-->(b) RETURN DISTINCT a.id AS id", engine), "id") == [1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_endpoint_order_by_keeps_multiplicity(engine):
    """ORDER BY over the multiplicity bag: ordered compare."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS id ORDER BY id", engine)
    assert [_scalar(v) for v in df["id"]] == [1, 1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_pair_projection_control_unchanged(engine):
    """Control: the pair projection was already row-correct and must stay so."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS x, b.id AS y", engine)
    got = sorted((int(r["x"]), int(r["y"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (1, 3), (2, 3), (3, 4)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(strict=True, reason="#1899 residual: whole-row endpoint projection still "
                   "collapses multiplicity (b bound twice to node 3 must yield two rows)")
def test_whole_row_endpoint_projection_multiplicity_residual(engine):
    """Residual pin: `RETURN b` (whole entity) over the same match should be a
    4-row bag (node 3 twice). Flip when the whole-row lane joins binding rows."""
    df = _run("MATCH (a)-->(b) RETURN b", engine)
    assert len(df) == 4


# ===========================================================================
# 2. Grouped-aggregate fast path: output column named like the node id
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_output_named_id_serves(engine):
    """#1899 item 2: `a.id AS id` collides with the node-id column inside the
    single-hop fast path's lookup rename -- was a raw pandas KeyError."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS id, count(*) AS c", engine)
    got = sorted((int(r["id"]), int(r["c"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (2, 1), (3, 1)]


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_output_id_sourced_from_other_prop(engine):
    """The reverse collision: an output NAMED `id` sourced from another
    property must not corrupt the join key."""
    df = _run("MATCH (a)-->(b) RETURN a.city AS id, count(*) AS c", engine)
    got = sorted(((_scalar(r["id"]), int(r["c"])) for r in df.to_dict("records")),
                 key=lambda t: str(t[0]))
    assert got == [("NYC", 3), ("SF", 1)]


# ===========================================================================
# 3. Empty-UNWIND ungrouped aggregate identities
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("UNWIND [] AS x RETURN count(*) AS out", 0),
    ("UNWIND [] AS x RETURN count(x) AS out", 0),
    ("UNWIND [] AS x RETURN sum(x) AS out", 0),
    ("UNWIND [] AS x RETURN min(x) AS out", None),
], ids=["count_star", "count_x", "sum", "min"])
def test_empty_unwind_ungrouped_aggregate_identity_row(query, expected, engine):
    """#1899 item 3: an ungrouped aggregate over zero rows emits ONE row with
    the aggregate identity (count -> 0, sum -> 0, min -> null)."""
    df = _run(query, engine)
    assert len(df) == 1
    assert _scalar(df["out"][0]) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_collect_identity_row(engine):
    df = _run("UNWIND [] AS x RETURN collect(x) AS out", engine)
    assert len(df) == 1
    assert _scalar(df["out"][0]) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_multi_aggregate_identity_row(engine):
    df = _run("UNWIND [] AS x RETURN count(*) AS c, sum(x) AS s", engine)
    assert df.to_dict("records") == [{"c": 0, "s": 0}]


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_unwind_non_aggregate_still_zero_rows(engine):
    """Negative control: a non-aggregate RETURN over the empty stream stays
    zero rows -- the identity row is aggregate-only."""
    assert len(_run("UNWIND [] AS x RETURN x", engine)) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_nonempty_unwind_aggregate_unaffected(engine):
    df = _run("UNWIND [1, 2] AS x RETURN count(*) AS c, sum(x) AS s", engine)
    assert df.to_dict("records") == [{"c": 2, "s": 3}]


# ===========================================================================
# 4. Negative list subscripts (pandas serves; polars parity-or-NIE)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("UNWIND [[1,2,3]] AS l RETURN l[-1] AS v", [3]),
    ("UNWIND [[1,2,3]] AS l RETURN l[-3] AS v", [1]),
    ("UNWIND [[1,2,3]] AS l RETURN l[-4] AS v", [None]),
    ("UNWIND [[1,2,3]] AS l RETURN l[2] AS v", [3]),
    ("UNWIND [[1,2,3]] AS l RETURN l[3] AS v", [None]),
    ("UNWIND [[1,2],[5]] AS l RETURN l[-1] AS v", [2, 5]),
], ids=["last", "neg_first", "neg_out_of_range", "pos_last", "pos_out_of_range", "per_row"])
def test_negative_list_subscript_indexes_from_end(query, expected, engine):
    """#1899 item 4: l[-1] is the last element (openCypher); out-of-range in
    either direction is null. polars declines UNWIND honestly (parity-or-NIE)."""
    if engine == "polars":
        with pytest.raises(NotImplementedError):
            _run(query, "polars")
        return
    df = _run(query, engine)
    assert [_scalar(v) for v in df["v"]] == expected


# ===========================================================================
# 5. Round-005 amplification (mutation-audit round for #1899/#1901)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,value_if_emitted", [
    ("UNWIND [] AS x RETURN count(*) + 1 AS c", 1),
    ("UNWIND [] AS x RETURN abs(sum(x)) AS c", 0),
    ("UNWIND [] AS x RETURN avg(x) * 2 AS c", None),
], ids=["count_plus_1", "abs_sum", "avg_times_2"])
def test_empty_compound_aggregate_never_fabricates(query, value_if_emitted, engine):
    """A compound aggregate item over an empty stream must never emit the bare
    identity under the internal postagg column: either the identity row is
    declined (0 rows, named residual below) or the post-aggregate expression is
    evaluated (count(*) + 1 -> 1). A fabricated {'__cypher_postagg__': 0} is
    the silent-wrong this pins out."""
    df = _run(query, engine)
    assert not any(str(col).startswith("__cypher") for col in df.columns)
    if len(df):
        assert list(df.columns) == ["c"]
        assert _scalar(df["c"][0]) == value_if_emitted


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_mixed_compound_aggregate_never_fabricates(engine):
    """Mixed pure + compound items: emitting a partial row that drops the
    compound alias (or leaks an internal column) is silent-wrong."""
    df = _run("UNWIND [] AS x RETURN count(*) AS c, count(*) + 1 AS d", engine)
    assert not any(str(col).startswith("__cypher") for col in df.columns)
    if len(df):
        assert sorted(df.columns) == ["c", "d"]
        assert _scalar(df["c"][0]) == 0 and _scalar(df["d"][0]) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_compound_aggregate_identity_residual(engine):
    df = _run("UNWIND [] AS x RETURN count(*) + 1 AS c", engine)
    assert len(df) == 1 and _scalar(df["c"][0]) == 1


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("query", [
    "UNWIND [] AS x RETURN count(*) AS c SKIP 1",
    "UNWIND [] AS x RETURN count(*) AS c LIMIT 0",
], ids=["skip_past_identity", "limit_zero"])
def test_empty_aggregate_identity_row_respects_paging_removal(query, engine):
    """SKIP past / LIMIT 0 must remove the synthesized identity row -- a
    synthesis that ignores the final stage's paging would emit it anyway."""
    assert len(_run(query, engine)) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "UNWIND [] AS x RETURN count(*) AS c SKIP 0",
    "UNWIND [] AS x RETURN count(*) AS c LIMIT 1",
], ids=["skip_zero", "limit_one"])
def test_empty_aggregate_identity_row_survives_noop_paging(query, engine):
    df = _run(query, engine)
    assert len(df) == 1 and _scalar(df["c"][0]) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,true_value", [
    ("UNWIND [] AS x WITH count(*) AS c RETURN count(c) AS out", 1),
    ("UNWIND [] AS x WITH count(*) AS c RETURN min(c) AS out", 0),
    ("UNWIND [] AS x WITH count(*) AS c RETURN collect(c) AS out", [0]),
], ids=["count_of_count", "min_of_count", "collect_of_count"])
def test_chained_ungrouped_aggregate_never_fabricates(query, true_value, engine):
    """An aggregate OF an ungrouped aggregate sees ONE row (the earlier identity,
    c = 0), never an empty stream: count(c) is 1, min(c) is 0, collect(c) is [0].
    Synthesizing the final aggregate's bare identity fabricates 0/null/[] --
    either decline (0 rows, residual below) or emit the true chained value."""
    df = _run(query, engine)
    if len(df):
        got = df["out"][0]
        if isinstance(true_value, list):
            assert [_scalar(x) for x in got] == true_value
        else:
            assert _scalar(got) == true_value


@pytest.mark.parametrize("engine", ENGINES)
def test_chained_ungrouped_aggregate_identity_residual(engine):
    df = _run("UNWIND [] AS x WITH count(*) AS c RETURN count(c) AS out", engine)
    assert len(df) == 1 and _scalar(df["out"][0]) == 1


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_grouped_aggregate_over_empty_stream_stays_zero_rows(engine):
    """Grouped aggregates over zero rows emit zero groups -- the identity row
    is ungrouped-only (openCypher)."""
    assert len(_run("UNWIND [] AS x RETURN x AS k, count(*) AS c", engine)) == 0


@pytest.mark.parametrize("engine", ["pandas", pytest.param("polars", marks=polars_only)])
def test_edges_only_graph_projection_keeps_multiplicity(engine):
    """Bag semantics on a graph with no node table: nodes are materialized from
    edge endpoints, and a-side keeps one row per edge ([1, 1, 2, 3])."""
    edges = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 3, 3, 4]})
    e = pl.from_pandas(edges) if engine == "polars" else edges
    g = graphistry.edges(e, "s", "d")
    out = g.gfql("MATCH (a)-->(b) RETURN a.id AS x", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert _bag(out, "x") == [1, 1, 2, 3]


@polars_only
def test_polars_integral_float_endpoints_join_node_ids():
    """Float64 endpoint columns holding integral values join Int64 node ids
    losslessly (pandas NaN-promotion artifact); bag multiplicity preserved."""
    edges = pd.DataFrame({"s": [1.0, 1.0, 2.0, 3.0], "d": [2.0, 3.0, 3.0, 4.0]})
    assert _bag(_run("MATCH (a)-->(b) RETURN a.id AS x", "polars", edges=edges), "x") == [1, 1, 2, 3]


@polars_only
def test_polars_nonintegral_float_endpoint_declines_typed():
    """A truly non-integral endpoint cannot cast losslessly: polars declines
    with its typed NIE rather than silently coercing or crashing."""
    edges = pd.DataFrame({"s": [1.0, 2.5], "d": [2.0, 3.0]})
    with pytest.raises(NotImplementedError):
        _run("MATCH (a)-->(b) RETURN a.id AS x", "polars", edges=edges)


@pytest.mark.parametrize("engine", ENGINES)
def test_leading_optional_match_keeps_multiplicity(engine):
    """A leading OPTIONAL MATCH binds nothing before it, so nothing can go
    unmatched and it is a plain MATCH for row purposes: same [1, 1, 2, 3] bag."""
    assert _bag(_run("OPTIONAL MATCH (a)-->(b) RETURN a.id AS x", engine), "x") == [1, 1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_optional_match_no_match_null_extension_preserved(engine):
    """The reason the OPTIONAL guard exists: a no-match OPTIONAL still emits
    one null-extended row; an inner-join binding lane would drop it."""
    df = _run("OPTIONAL MATCH (a {name: 'Zed'})-->(b) RETURN a.id AS x", engine)
    assert len(df) == 1 and _scalar(df["x"][0]) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_parallel_edge_multiplicity(engine):
    """A selective seed does not change bag semantics: Ann has two edges to Bob,
    so the seeded hop is [2, 2], the same bag the unseeded control below keeps."""
    edges = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 2, 3, 4]})
    assert _bag(_run("MATCH (a {name: 'Ann'})-->(b) RETURN b.id AS x", engine, edges=edges), "x") == [2, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_unseeded_parallel_edges_keep_multiplicity(engine):
    edges = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 2, 3, 4]})
    assert _bag(_run("MATCH (a)-->(b) RETURN b.id AS x", engine, edges=edges), "x") == [2, 2, 3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_bag_paging_after_order_by(engine):
    """SKIP/LIMIT page the multiplicity bag, not the deduplicated set:
    sorted bag [1, 1, 2, 3], SKIP 1 LIMIT 2 -> [1, 2]."""
    df = _run("MATCH (a)-->(b) RETURN a.id AS x ORDER BY x SKIP 1 LIMIT 2", engine)
    assert [_scalar(v) for v in df["x"]] == [1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_forcing_does_not_open_unsupported_lanes(engine):
    """Multiplicity forcing must not widen the supported surface: disconnected
    multi-pattern and repeated-alias projections stay typed declines."""
    for query in [
        "MATCH (a), (b)-->(c) RETURN a.id AS x",
        "MATCH (a)-->(b), (c)-->(d) RETURN a.id AS x",
        "MATCH (a)-->(a) RETURN a.id AS x",
    ]:
        with pytest.raises(GFQLValidationError):
            _run(query, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_pure_carry_does_not_leak_out_of_scope_alias(engine):
    """Flattening `WITH a` must not resurrect b: the binder scope error stays."""
    with pytest.raises(GFQLValidationError):
        _run("MATCH (a)-->(b) WITH a RETURN b.id AS x", engine)


@pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")
def test_cudf_bag_and_identity_parity():
    """cuDF engine (dataframe ops only): bag multiplicity, empty-stream count
    identity, and the id-named grouped output all hand-checked."""
    assert _bag(_run("MATCH (a)-->(b) RETURN a.id AS x", "cudf"), "x") == [1, 1, 2, 3]
    df = _run("UNWIND [] AS x RETURN count(*) AS c", "cudf")
    assert len(df) == 1 and _scalar(df["c"][0]) == 0
    grouped = _run("MATCH (a)-->(b) RETURN a.id AS id, count(*) AS c", "cudf")
    got = sorted((int(r["id"]), int(r["c"])) for r in grouped.to_dict("records"))
    assert got == [(1, 2), (2, 1), (3, 1)]
