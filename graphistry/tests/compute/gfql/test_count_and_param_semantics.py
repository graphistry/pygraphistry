"""openCypher conformance pins for the count(*) lane and for parameter equality (#1905/#1906).

Every expected value below is a HAND-COMPUTED openCypher oracle over the fixture, never
"whatever the other engine returns". The two families:

* COUNT-TWIN INVARIANT -- one MATCH has one cardinality, so ``RETURN count(*)`` must equal
  the row count of the identical query with a non-aggregate projection. openCypher TRAIL
  semantics say a relationship may bind at most once per pattern (nodes may repeat), so
  every oracle here enumerates ordered tuples of DISTINCT relationships.
* PARAMETER EQUALITY -- ``=`` and property-map equality compare VALUES; a scalar property
  never equals a list. Only ``IN`` is membership.
"""
from typing import Any, List

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLSchemaError

ENGINES = ["pandas", "polars"]


def _rows(result: Any) -> List[dict]:
    frame = result._nodes
    if frame is None:
        return []
    if "polars" in type(frame).__module__:
        return frame.to_dicts()
    return frame.to_dict(orient="records")


def _mk(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _self_loop_graph(engine: str) -> Any:
    """s has a self-loop and an edge to t: edges e0 = s->s, e1 = s->t."""
    return _mk(
        pd.DataFrame({"id": ["s", "t"], "v": [1, 2]}),
        pd.DataFrame({"s": ["s", "s"], "d": ["s", "t"], "rel": ["L", "R"]}),
        engine,
    )


def _mixed_graph(engine: str) -> Any:
    """8 nodes; edges e0 = 0->1, e1 = 0->1 (parallel), e2 = 1->2, e3 = 2->3,
    e4 = 3->4, e5 = 5->5 (self-loop), e6 = 6->0."""
    return _mk(
        pd.DataFrame({"id": [0, 1, 2, 3, 4, 5, 6, 7], "v": [10, 20, 30, 40, 50, 60, 70, 80]}),
        pd.DataFrame({
            "s": [0, 0, 1, 2, 3, 5, 6],
            "d": [1, 1, 2, 3, 4, 5, 0],
            "type": ["K", "K", "K", "L", "L", "K", "L"],
        }),
        engine,
    )


def _parallel_graph(engine: str) -> Any:
    """Two parallel a->b edges and one b->c edge."""
    return _mk(
        pd.DataFrame({"id": ["a", "b", "c"]}),
        pd.DataFrame({"s": ["a", "a", "b"], "d": ["b", "b", "c"], "rel": ["K", "K", "K"]}),
        engine,
    )


def _cycle_graph(engine: str, n: int) -> Any:
    ids = [chr(ord("a") + i) for i in range(n)]
    return _mk(
        pd.DataFrame({"id": ids}),
        pd.DataFrame({"s": ids, "d": ids[1:] + ids[:1]}),
        engine,
    )


# ---------------------------------------------------------------------------
# count-twin invariant (#1905): count(*) == len(rows of the non-aggregate twin)
# ---------------------------------------------------------------------------

# (fixture builder, MATCH text, non-aggregate projection, hand oracle)
_COUNT_TWIN_CASES = [
    # SELF-LOOP CHAIN. r1 = e0 (s->s) leaves r2 in {e1} (e0 is already bound) -> (s, s, t);
    # r1 = e1 ends at t which has no out-edge. 1 binding.
    (_self_loop_graph, "MATCH (a)-->(b)-->(c)", "a.id, b.id, c.id", 1),
    # SELF-LOOP, one hop: both edges leave s. 2 bindings.
    (_self_loop_graph, "MATCH (a)-->(b)", "a.id, b.id", 2),
    # SELF-LOOP, branching comma: two DISTINCT out-edges of s, ordered -> (e0, e1), (e1, e0).
    (_self_loop_graph, "MATCH (a)-->(b), (a)-->(c)", "a.id, b.id, c.id", 2),
    # MIXED chain: middle 1 -> (e0,e2),(e1,e2); middle 2 -> (e2,e3); middle 3 -> (e3,e4);
    # middle 0 -> (e6,e0),(e6,e1); middle 5 would need e5 twice. 6 bindings.
    (_mixed_graph, "MATCH (a)-->(b)-->(c)", "a.id, b.id, c.id", 6),
    # MIXED, same chain spelled as a comma pattern -- same 6 bindings.
    (_mixed_graph, "MATCH (a)-->(b), (b)-->(c)", "a.id, b.id, c.id", 6),
    # MIXED branching out-star: only node 0 has two out-edges -> (e0,e1),(e1,e0). 2 bindings.
    (_mixed_graph, "MATCH (a)-->(b), (a)-->(c)", "a.id, b.id, c.id", 2),
    # MIXED branching in-star: only node 1 has two in-edges -> (e0,e1),(e1,e0). 2 bindings.
    (_mixed_graph, "MATCH (a)-->(b), (c)-->(b)", "a.id, b.id, c.id", 2),
    # MIXED, typed to the K edges {e0, e1, e2, e5}: (e0,e2),(e1,e2); e5 cannot pair with
    # itself. 2 bindings.
    (_mixed_graph, "MATCH (a)-[:K]->(b)-[:K]->(c)", "a.id, b.id, c.id", 2),
    # MIXED, hops typed DIFFERENTLY (typed then untyped), so the two hops read DIFFERENT
    # edge domains and one edge (the K self-loop e5) is in both: (e0,e2),(e1,e2),(e2,e3);
    # e5 would have to bind twice. 3 bindings.
    (_mixed_graph, "MATCH (a)-[:K]->(b)-->(c)", "a.id, b.id, c.id", 3),
    # PARALLEL edges: (a->b #1, b->c), (a->b #2, b->c). 2 bindings.
    (_parallel_graph, "MATCH (a)-->(b)-->(c)", "a.id, b.id, c.id", 2),
    # PARALLEL branching: the two a->b edges, ordered. 2 bindings.
    (_parallel_graph, "MATCH (a)-->(b), (a)-->(c)", "a.id, b.id, c.id", 2),
    # VAR-LENGTH 1..2 over MIXED: 7 one-hop bindings + the 6 two-hop bindings above.
    (_mixed_graph, "MATCH (a)-[*1..2]->(b)", "a.id, b.id", 13),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("builder,match,projection,oracle", _COUNT_TWIN_CASES)
def test_count_star_equals_row_twin_and_oracle(
    engine: str, builder: Any, match: str, projection: str, oracle: int
) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    graph = builder(engine)
    rows = _rows(graph.gfql(f"{match} RETURN {projection}", engine=engine))
    counted = _rows(graph.gfql(f"{match} RETURN count(*) AS n", engine=engine))

    assert len(rows) == oracle
    assert counted == [{"n": oracle}]


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_star_over_branching_pattern(engine: str) -> None:
    # Only node 0 has two out-edges (e0, e1), so it is the only group and contributes the
    # two ordered pairs of DISTINCT relationships; every other node has out-degree <= 1.
    if engine == "polars":
        pytest.importorskip("polars")
    graph = _mixed_graph(engine)
    # No ORDER BY: sorting a grouped comma-pattern count still raises on both engines
    # (pre-existing, unrelated to #1905); one group makes ordering moot anyway.
    result = graph.gfql(
        "MATCH (a)-->(b), (a)-->(c) RETURN a.id AS a, count(*) AS n", engine=engine
    )
    assert _rows(result) == [{"a": 0, "n": 2}]


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_count_star_over_chain_pattern(engine: str) -> None:
    # Grouped on the middle node: 0 -> (e6,e0),(e6,e1); 1 -> (e0,e2),(e1,e2);
    # 2 -> (e2,e3); 3 -> (e3,e4); 5 would need e5 twice.
    if engine == "polars":
        pytest.importorskip("polars")
    graph = _mixed_graph(engine)
    result = graph.gfql(
        "MATCH (a)-->(b)-->(c) RETURN b.id AS b, count(*) AS n ORDER BY b", engine=engine
    )
    assert _rows(result) == [{"b": 0, "n": 2}, {"b": 1, "n": 2}, {"b": 2, "n": 1}, {"b": 3, "n": 1}]


@pytest.mark.parametrize("engine", ENGINES)
def test_count_star_agrees_with_count_alias_on_self_loop_chain(engine: str) -> None:
    # count(a) was already trail-exact while count(*) rode the degree product; pin both.
    if engine == "polars":
        pytest.importorskip("polars")
    graph = _self_loop_graph(engine)
    assert _rows(graph.gfql("MATCH (a)-->(b)-->(c) RETURN count(*) AS n", engine=engine)) == [{"n": 1}]
    assert _rows(graph.gfql("MATCH (a)-->(b)-->(c) RETURN count(a) AS n", engine=engine)) == [{"n": 1}]


# ---------------------------------------------------------------------------
# parameter equality (#1905): `=` and property maps compare values, `IN` is membership
# ---------------------------------------------------------------------------

def _param_graph(engine: str) -> Any:
    return _mk(
        pd.DataFrame({"id": [0, 1, 2, 3], "v": [10, 30, 70, 80], "name": ["a", "b", "c", "d"]}),
        pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "rel": ["K", "L", "K"]}),
        engine,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "query,params",
    [
        ("MATCH (n) WHERE n.v = $l RETURN n.id AS id", {"l": [30, 70]}),
        ("MATCH (n {v: $l}) RETURN n.id AS id", {"l": [30, 70]}),
        ("MATCH (n) WHERE n.name = $l RETURN n.id AS id", {"l": ["a", "d"]}),
        ("MATCH (n {name: $l}) RETURN n.id AS id", {"l": ["a", "d"]}),
    ],
)
def test_list_param_equality_is_never_membership(engine: str, query: str, params: dict) -> None:
    # openCypher: a scalar property is never EQUAL to a list, so this is 0 rows -- not the
    # `IN`-style membership a filter_dict list value would have meant.
    if engine == "polars":
        pytest.importorskip("polars")
    assert _rows(_param_graph(engine).gfql(query, params=params, engine=engine)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "query,params",
    [
        ("MATCH ()-[e]->() WHERE e.rel = $l RETURN e.rel AS rel", {"l": ["K", "L"]}),
        ("MATCH ()-[e {rel: $l}]->() RETURN e.rel AS rel", {"l": ["K"]}),
    ],
)
def test_list_param_equality_on_edge_properties_is_never_membership(
    engine: str, query: str, params: dict
) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    assert _rows(_param_graph(engine).gfql(query, params=params, engine=engine)) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_in_param_is_still_membership(engine: str) -> None:
    # The membership spelling must keep working: only `=` changed.
    if engine == "polars":
        pytest.importorskip("polars")
    result = _param_graph(engine).gfql(
        "MATCH (n) WHERE n.v IN $l RETURN n.id AS id ORDER BY id", params={"l": [30, 70]}, engine=engine
    )
    assert _rows(result) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_param_equality_still_matches(engine: str) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    result = _param_graph(engine).gfql(
        "MATCH (n) WHERE n.v = $x RETURN n.id AS id", params={"x": 30}, engine=engine
    )
    assert _rows(result) == [{"id": 1}]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE n.id = $x RETURN n.id AS id",
        "MATCH (n {id: $x}) RETURN n.id AS id",
    ],
)
def test_string_param_against_numeric_column_is_a_typed_error(engine: str, query: str) -> None:
    # Error-contract parity (#1905): polars used to leak a raw polars.ComputeError here.
    if engine == "polars":
        pytest.importorskip("polars")
    with pytest.raises(GFQLSchemaError) as excinfo:
        _param_graph(engine).gfql(query, params={"x": "1"}, engine=engine)
    assert "incompatible-column-type" in str(excinfo.value)


# ---------------------------------------------------------------------------
# undirected unbounded var-length row lane (#1906)
# ---------------------------------------------------------------------------

def test_undirected_unbounded_rows_match_count_twin_on_c4() -> None:
    # 4-cycle a-b-c-d-a, seeded at a. Trails from a, by length:
    # 1: [ab] -> b, [da] -> d; 2: [ab,bc] -> c, [da,cd] -> c;
    # 3: [ab,bc,cd] -> d, [da,cd,bc] -> b; 4: both ways round -> a, a. 8 bindings.
    graph = _cycle_graph("pandas", 4)
    rows = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y) RETURN y.id AS y"))
    counted = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y) RETURN count(*) AS n"))
    assert len(rows) == 8
    assert counted == [{"n": 8}]
    assert sorted(row["y"] for row in rows) == ["a", "a", "b", "b", "c", "c", "d", "d"]


def test_undirected_unbounded_rows_match_count_twin_on_triangle() -> None:
    # Triangle a-b, b-c, c-a seeded at a: [ab] -> b, [ca] -> c, [ab,bc] -> c,
    # [ca,bc] -> b, and the two 3-edge circuits -> a. 6 bindings.
    graph = _cycle_graph("pandas", 3)
    rows = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y) RETURN y.id AS y"))
    counted = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y) RETURN count(*) AS n"))
    assert len(rows) == 6
    assert counted == [{"n": 6}]


def test_undirected_unbounded_second_element_emits_no_impossible_rows() -> None:
    # Triangle a->b->c->a seeded at a, then a DIRECTED hop off the endpoint. Reaching
    # z = b would need the a-b edge on both elements, which TRAIL forbids: only a and c
    # are reachable, once each ([ab] -> y=b -> z=c, [ca,bc] -> y=b -> z=c is the same
    # edge reuse, ...) -- the enumeration leaves exactly {a, c}.
    graph = _cycle_graph("pandas", 3)
    rows = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y)-[]->(z) RETURN z.id AS z"))
    counted = _rows(graph.gfql("MATCH (x {id:'a'})-[*]-(y)-[]->(z) RETURN count(*) AS n"))
    assert sorted(row["z"] for row in rows) == ["a", "c"]
    assert counted == [{"n": 2}]


def test_undirected_unbounded_declines_typed_on_polars() -> None:
    # polars has no native var-length lane for this shape; parity-or-error by design.
    pytest.importorskip("polars")
    graph = _cycle_graph("polars", 4)
    with pytest.raises(NotImplementedError):
        graph.gfql("MATCH (x {id:'a'})-[*]-(y) RETURN y.id AS y", engine="polars")


# ---------------------------------------------------------------------------
# residuals -- pinned so they stop being invisible
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="#1905: a node label absent from the graph schema hard-errors "
           "[column-not-found]; openCypher treats a missing label as matching nothing "
           "(0 rows). Needs an owner decision: strict-schema vs missing-as-absent.",
)
@pytest.mark.parametrize("query", [
    "MATCH (n:Nope) RETURN n.id AS id",
    "MATCH (n) WHERE n:Nope RETURN n.id AS id",
])
def test_nonexistent_node_label_should_be_zero_rows(query: str) -> None:
    assert _rows(_param_graph("pandas").gfql(query)) == []


@pytest.mark.xfail(
    strict=True,
    reason="#1905: a nonexistent label on an OPTIONAL arm hard-errors [column-not-found]; "
           "openCypher null-extends the mandatory rows instead. Same owner decision. "
           "The raised error's available-columns list also leaks the binder's alias "
           "marker column.",
)
def test_nonexistent_optional_arm_label_should_null_extend() -> None:
    rows = _rows(_param_graph("pandas").gfql(
        "MATCH (a)-->(b) OPTIONAL MATCH (a)-->(c:Nope) RETURN a.id AS a, c.id AS c"
    ))
    assert len(rows) == 3
    assert all(row["c"] is None for row in rows)


@pytest.mark.xfail(
    strict=True,
    reason="#1787 (pre-existing, not #1905): pandas min_hops>=2 hop-window starvation -- "
           "row/pipeline.py's trail expander consumes edge_op.execute output that hop.py "
           "already pruned with its dedup-by-node eccentricity gate, so an emptied frame "
           "starves the trail lane and the window returns 0 rows.",
)
@pytest.mark.parametrize("n,query,oracle", [
    # seeded *4..4 on the 4-cycle: the two 4-edge circuits back to a.
    (4, "MATCH (x {id:'a'})-[*4..4]-(y) RETURN y.id AS y", 2),
    # unseeded *3..3: 4 starts x 2 directions x 1 route = 8 on C4, 6 on the triangle.
    (4, "MATCH (x)-[*3..3]-(y) RETURN y.id AS y", 8),
    (3, "MATCH (x)-[*3..3]-(y) RETURN y.id AS y", 6),
])
def test_undirected_min_hops_window_should_not_starve(n: int, query: str, oracle: int) -> None:
    assert len(_rows(_cycle_graph("pandas", n).gfql(query))) == oracle
