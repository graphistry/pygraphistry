"""Round-005 var-length path TRAIL semantics + lane-consistency pins (#1903).

openCypher trail semantics, hand-computed per fixture: each distinct edge
SEQUENCE is one row; a relationship binds at most once per path (edges never
repeat, nodes may); one MATCH has ONE cardinality regardless of what the
RETURN projects; a plain-MATCH shortestPath with no path emits NO row (only
OPTIONAL MATCH null-extends).

polars is parity-or-NIE: it matches the pandas oracle or declines honestly.
Fixtures: DIAMOND a->b1->c, a->b2->c, c->d; TRI directed 3-cycle; SELF
self-loop s->s plus s->t; PARA parallel a->b x2 plus b->c; LINE p1->p2->p3->p4
with disconnected q1.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]

FIXTURES = {
    "DIAMOND": (
        pd.DataFrame({"id": ["a", "b1", "b2", "c", "d"]}),
        pd.DataFrame({
            "s": ["a", "a", "b1", "b2", "c"],
            "d": ["b1", "b2", "c", "c", "d"],
            "type": ["KNOWS"] * 5,
        }),
    ),
    "TRI": (
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "a"], "type": ["KNOWS"] * 3}),
    ),
    "SELF": (
        pd.DataFrame({"id": ["s", "t"]}),
        pd.DataFrame({"s": ["s", "s"], "d": ["s", "t"], "type": ["KNOWS"] * 2}),
    ),
    "PARA": (
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "a", "b"], "d": ["b", "b", "c"], "type": ["KNOWS"] * 3}),
    ),
    "LINE": (
        pd.DataFrame({"id": ["p1", "p2", "p3", "p4", "q1"]}),
        pd.DataFrame({"s": ["p1", "p2", "p3"], "d": ["p2", "p3", "p4"], "type": ["KNOWS"] * 3}),
    ),
}


def _run(fixture: str, query: str, engine: str) -> pd.DataFrame:
    nodes, edges = FIXTURES[fixture]
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


#: A polars decline must NAME the gap; these are the capabilities it may cite.
DECLINE_PHRASES = (
    "not yet hop-gated",                                # #1748 min_hops>1 node-alias window
    "undirected min_hops>1",                            # var-length feature gate
    "require terminating variable-length segments",     # unbounded walk into a reachable cycle
    "does not yet natively support cypher row op",      # row-op surface gap
)

#: Every (fixture, query) polars is currently allowed to decline -- keyed on the
#: pair because the same shape routes differently per graph (``-[*]->`` serves on
#: the acyclic DIAMOND and declines on the cyclic TRI). Membership is asserted in
#: BOTH directions, so a shape that starts declining -- or quietly starts serving
#: -- fails instead of passing silently. Without the table every polars cell here
#: is vacuous: 14 of the 28 (fixture, query) pairs below decline today.
POLARS_DECLINED = frozenset({
    ("DIAMOND", "MATCH (x {id:'a'})-[*1..2]->(m)-[]->(y) RETURN y.id AS y"),
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y"),
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]->(y {id:'c'}) RETURN count(*) AS y"),
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]->(y) RETURN x.id AS x, y.id AS y"),
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]->(y) RETURN y.id AS y"),
    ("DIAMOND", "MATCH (x {id:'a'})-[*3]->(y) RETURN y.id AS y"),
    ("DIAMOND", "MATCH (x {id:'c'})<-[*2]-(y) RETURN y.id AS y"),
    ("PARA", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y"),
    ("PARA", "MATCH (x {id:'a'})-[*2]->(y) RETURN y.id AS y"),
    ("SELF", "MATCH (x {id:'s'})-[*2]->(y) RETURN y.id AS y"),
    ("TRI", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y"),
    ("TRI", "MATCH (x {id:'a'})-[*3]-(y) RETURN y.id AS y"),
    ("TRI", "MATCH (x {id:'a'})-[*3]->(y) RETURN y.id AS y"),
    ("TRI", "MATCH (x {id:'a'})-[*]->(y) RETURN y.id AS y"),
})


def _bag(fixture: str, query: str, engine: str, col: str = "y"):
    """Sorted value bag, or ``('DECLINED', message)`` when the engine declines."""
    try:
        df = _run(fixture, query, engine)
    except NotImplementedError as exc:
        return ("DECLINED", str(exc))
    except GFQLValidationError as exc:
        return ("DECLINED", str(exc))
    return sorted(str(v) for v in df[col])


def _is_decline(got) -> bool:
    return isinstance(got, tuple) and bool(got) and got[0] == "DECLINED"


def _assert_polars_routing(fixture, query, got) -> None:
    """A decline passes only if it was TABLED and it SAYS what is missing."""
    assert _is_decline(got) == ((fixture, query) in POLARS_DECLINED), (
        f"polars routing changed for ({fixture}, {query}): "
        f"{'declined but not tabled' if _is_decline(got) else 'tabled as declined but served'}"
    )
    if _is_decline(got):
        assert any(phrase in got[1] for phrase in DECLINE_PHRASES), \
            f"decline did not name the gap: {got[1]}"


def _assert_bag(fixture, query, engine, expected, col="y"):
    got = _bag(fixture, query, engine, col)
    if engine == "polars":
        _assert_polars_routing(fixture, query, got)
        if _is_decline(got):
            return
    assert got == sorted(str(v) for v in expected), f"{query}: got {got}"


# ===========================================================================
# 1+2. Lane consistency: single-alias endpoint projections match the row lane
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fixture,query,expected", [
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]->(y) RETURN y.id AS y", ["c", "c"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*1..2]->(y) RETURN y.id AS y", ["b1", "b2", "c", "c"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*]->(y) RETURN y.id AS y", ["b1", "b2", "c", "c", "d", "d"]),
    ("DIAMOND", "MATCH (x {id:'c'})<-[*2]-(y) RETURN y.id AS y", ["a", "a"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y", ["c", "c"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*0..2]->(y) RETURN y.id AS y", ["a", "b1", "b2", "c", "c"]),
    ("PARA", "MATCH (x {id:'a'})-[*1]->(y) RETURN y.id AS y", ["b", "b"]),
    ("PARA", "MATCH (x {id:'a'})-[*2]->(y) RETURN y.id AS y", ["c", "c"]),
    ("TRI", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y", ["b", "c"]),
], ids=["d_exact2", "d_range", "d_unbounded", "d_reverse", "d_undirected",
        "d_zero_hop", "p_parallel1", "p_parallel2", "t_undirected_bfs_prune"])
def test_single_alias_projection_keeps_path_multiplicity(fixture, query, expected, engine):
    """#1903 items 1-2: the single-alias endpoint projection now rides the
    binding-row lane -- path multiplicity preserved (D01 gave 1 row, oracle 2),
    BFS visited-pruning gone (TRI undirected [*2] gave [], oracle [b,c])."""
    _assert_bag(fixture, query, engine, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_one_match_one_cardinality(engine):
    """The headline invariant: RETURN y.id, count(*), and RETURN x.id, y.id
    agree on cardinality for the same MATCH."""
    queries = ["MATCH (x {id:'a'})-[*2]->(y) RETURN y.id AS y",
               "MATCH (x {id:'a'})-[*2]->(y {id:'c'}) RETURN count(*) AS y",
               "MATCH (x {id:'a'})-[*2]->(y) RETURN x.id AS x, y.id AS y"]
    bag, cnt, pair = [_bag("DIAMOND", q, engine) for q in queries]
    if engine == "polars":
        for query, got in zip(queries, (bag, cnt, pair)):
            _assert_polars_routing("DIAMOND", query, got)
        if any(_is_decline(got) for got in (bag, cnt, pair)):
            return
    assert bag == ["c", "c"] and cnt == ["2"] and pair == ["c", "c"]


# ===========================================================================
# 3-5. Trail semantics: edges never repeat within a path
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fixture,query,expected", [
    ("SELF", "MATCH (x {id:'s'})-[*2]->(y) RETURN y.id AS y", ["t"]),
    ("SELF", "MATCH (x {id:'s'})-[*1..2]->(y) RETURN y.id AS y", ["s", "t", "t"]),
    ("SELF", "MATCH (x {id:'s'})-[*1]-(y) RETURN y.id AS y", ["s", "t"]),
    ("TRI", "MATCH (x {id:'a'})-[*1..4]->(y) RETURN y.id AS y", ["a", "b", "c"]),
    ("TRI", "MATCH (x {id:'a'})-[*]->(y) RETURN y.id AS y", ["a", "b", "c"]),
    ("TRI", "MATCH (x {id:'a'})-[]->(m)-[]->(y) RETURN y.id AS y", ["c"]),
    ("PARA", "MATCH (x {id:'a'})-[]->(m)-[]-(y) RETURN y.id AS y", ["a", "a", "c", "c"]),
    ("PARA", "MATCH (x {id:'a'})-[*2]-(y) RETURN y.id AS y", ["a", "a", "c", "c"]),
    ("TRI", "MATCH (x {id:'a'})-[*3]-(y) RETURN y.id AS y", ["a", "a"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*1..2]->(m)-[]->(y) RETURN y.id AS y", ["c", "c", "d", "d"]),
], ids=["selfloop_no_reuse", "selfloop_range", "selfloop_undirected_once",
        "cycle_bounded_terminates", "cycle_unbounded_terminates",
        "fixed_2hop_cross_unique", "parallel_return_trip_legal",
        "undirected_parallel_backtrack", "undirected_cycle_closes",
        "midpattern_varlen_then_hop"])
def test_trail_relationship_uniqueness(fixture, query, expected, engine):
    """#1903 items 3-5: an edge binds at most once per path (self-loop walk
    fabrication gone; cross-element reuse gone; a return trip over a PARALLEL
    edge stays legal; nodes may repeat)."""
    _assert_bag(fixture, query, engine, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_undirected_selfloop_single_hop_binds_once(engine):
    """Addendum A-1: both orientations of one self-loop are the SAME binding --
    one row, count 1 (was 2 on both engines)."""
    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0], "d": [0], "type": ["REL"]})
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    try:
        rows = g.gfql("MATCH (a)-[r]-(b) RETURN a.id AS y", engine=engine)._nodes
        cnt = g.gfql("MATCH (a)-[r]-(b) RETURN count(*) AS y", engine=engine)._nodes
    except NotImplementedError:
        assert engine == "polars"
        return
    rows = rows.to_pandas() if hasattr(rows, "to_pandas") else rows
    cnt = cnt.to_pandas() if hasattr(cnt, "to_pandas") else cnt
    assert len(rows) == 1 and int(cnt["y"][0]) == 1


# ===========================================================================
# 6. shortestPath: plain MATCH unreachable -> NO row; OPTIONAL null-extends
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_plain_match_unreachable_no_row(engine):
    q = ("MATCH (a {id:'p1'}), (b {id:'q1'}), p = shortestPath((a)-[*]-(b)) "
         "RETURN length(p) AS y")
    try:
        df = _run("LINE", q, engine)
    except NotImplementedError:
        assert engine == "polars"
        return
    assert len(df) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_bound_below_actual_no_row(engine):
    q = ("MATCH (a {id:'p1'}), (b {id:'p4'}), p = shortestPath((a)-[*..2]-(b)) "
         "RETURN length(p) AS y")
    try:
        df = _run("LINE", q, engine)
    except NotImplementedError:
        assert engine == "polars"
        return
    assert len(df) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_optional_unreachable_null_row(engine):
    q = ("MATCH (a {id:'p1'}), (b {id:'q1'}) "
         "OPTIONAL MATCH p = shortestPath((a)-[*]-(b)) RETURN length(p) AS y")
    try:
        df = _run("LINE", q, engine)
    except NotImplementedError:
        assert engine == "polars"
        return
    assert len(df) == 1 and pd.isna(df["y"][0])


# ===========================================================================
# 8. Grammar: [*..M] omitted lower bound (defaults to 1)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_open_min_bound_parses_and_serves(engine):
    _assert_bag("DIAMOND", "MATCH (x {id:'a'})-[*..3]->(y) RETURN y.id AS y",
                engine, ["b1", "b2", "c", "c", "d", "d"])


@pytest.mark.parametrize("engine", ENGINES)
def test_open_min_bound_shortest_path(engine):
    q = ("MATCH (a {id:'p1'}), (b {id:'p4'}), p = shortestPath((a)-[*..3]-(b)) "
         "RETURN length(p) AS y")
    try:
        df = _run("LINE", q, engine)
    except NotImplementedError:
        assert engine == "polars"
        return
    assert [int(v) for v in df["y"]] == [3]


# ===========================================================================
# Correct-inventory regression fences (were green pre-#1903; must stay green)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("fixture,query,expected", [
    ("TRI", "MATCH (x {id:'a'})-[*3]->(y) RETURN y.id AS y", ["a"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*3]->(y) RETURN y.id AS y", ["d", "d"]),
    ("DIAMOND", "MATCH (x {id:'a'})-[*1]-(y) RETURN y.id AS y", ["b1", "b2"]),
    ("LINE", "MATCH (x {id:'q1'})-[*1..3]-(y) RETURN y.id AS y", []),
], ids=["cycle_exact_closes", "diamond_exact3", "undirected_1_no_double", "component_isolation"])
def test_correct_inventory_stays_green(fixture, query, expected, engine):
    _assert_bag(fixture, query, engine, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_shortest_path_ties_and_typed_fences(engine):
    for q, fixture, expected in [
        ("MATCH (a {id:'a'}), (b {id:'c'}), p = shortestPath((a)-[*]->(b)) RETURN length(p) AS y", "DIAMOND", [2]),
        ("MATCH (a {id:'p1'}), (b {id:'p4'}), p = shortestPath((a)-[*]-(b)) RETURN length(p) AS y", "LINE", [3]),
    ]:
        try:
            df = _run(fixture, q, engine)
        except NotImplementedError:
            assert engine == "polars"
            continue
        assert [int(v) for v in df["y"]] == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_agg_lane_consistent(engine):
    q = "MATCH (x {id:'a'})-[*1..3]->(y) RETURN y.id AS y, count(*) AS n"
    try:
        df = _run("DIAMOND", q, engine)
    except NotImplementedError:
        assert engine == "polars"
        return
    got = sorted((str(r["y"]), int(r["n"])) for r in df.to_dict("records"))
    assert got == [("b1", 1), ("b2", 1), ("c", 2), ("d", 2)]


# ===========================================================================
# Residual pins (strict xfail, #1903)
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(strict=True, reason="#1903 addendum A-2 residual: the seeded typed-hop "
                   "lane (fast path AND its fallback) projects the destination NODE SET -- "
                   "parallel edges from a unique seed collapse ([1,2] vs bag [1,1,2])")
def test_seeded_parallel_edge_multiplicity_residual(engine):
    nodes = pd.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "b"]})
    edges = pd.DataFrame({"s": [0, 0, 0], "d": [1, 1, 2], "type": ["KNOWS"] * 3})
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    df = g.gfql("MATCH (a {id:0})-[{type:'KNOWS'}]->(b) RETURN b.id AS y", engine=engine)._nodes
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    assert sorted(int(v) for v in df["y"]) == [1, 1, 2]


@pytest.mark.xfail(strict=True, reason="#1903 residual: polars unbounded fixed point on a "
                   "reachable cycle still raises the terminating-segments error (its "
                   "node-frontier probe cannot see trail exhaustion); pandas serves it")
@polars_only
def test_polars_unbounded_cycle_residual():
    _assert_bag("TRI", "MATCH (x {id:'a'})-[*]->(y) RETURN y.id AS y", "polars", ["a", "b", "c"])
    df = _run("TRI", "MATCH (x {id:'a'})-[*]->(y) RETURN y.id AS y", "polars")
    assert sorted(df["y"]) == ["a", "b", "c"]
