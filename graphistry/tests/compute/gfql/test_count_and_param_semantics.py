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
    if "cudf" in type(frame).__module__:
        frame = frame.to_pandas()
    return frame.to_dict(orient="records")


def _mk(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


#: The count lanes have a THIRD arm (cudf device frames) that ``ENGINES`` does not reach.
ALL_ENGINES = ENGINES + ["cudf"]


def _skip_unless_engine(engine: str) -> None:
    if engine in ("polars", "cudf"):
        pytest.importorskip(engine)


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


# ===========================================================================
# G. Relationship-uniqueness correction: the lanes the count-twin cells above
#    do not reach (review round: mutation-audited additions)
# ===========================================================================


def _two_filter_loop_graph(engine: str) -> Any:
    """Self-loops on BOTH sides of a two-filter split.

    Nodes n1..n4. Edges, with the (rel, w) pair each relationship filter reads:
      e0 = n1->n2 (K, 1)   e1 = n2->n3 (L, 1)
      e2 = n2->n2 (K, 1)   -- self-loop passing BOTH filters
      e4 = n3->n3 (K, 1)   -- a SECOND self-loop passing BOTH filters
      e3 = n4->n4 (K, 2)   -- self-loop passing the FIRST filter ONLY
    """
    return _mk(
        pd.DataFrame({"id": ["n1", "n2", "n3", "n4"]}),
        pd.DataFrame({
            "s": ["n1", "n2", "n2", "n4", "n3"],
            "d": ["n2", "n3", "n2", "n4", "n3"],
            "rel": ["K", "L", "K", "K", "K"],
            "w": [1, 1, 1, 2, 1],
        }),
        engine,
    )


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_two_hop_count_subtracts_only_loops_passing_both_relationship_filters(engine: str) -> None:
    """Hand oracle over ``MATCH (a)-[{rel:'K'}]->(b)-[{w:1}]->(c)``.

    First arm admits {e0, e2, e3, e4}; second admits {e0, e1, e2, e4}. Ordered
    pairs of DISTINCT relationships with ``dst(r1) == src(r2)``:
      r1=e0 (ends n2) -> r2 in {e1, e2}                        2
      r1=e2 (ends n2) -> r2 in {e1} (e2 already bound)         1
      r1=e3 (ends n4) -> no second-arm edge leaves n4          0
      r1=e4 (ends n3) -> no second-arm edge leaves n3          0
    Total 3. The uncorrected degree product is 5 (middle n2: 2x2, middle n3: 1x1),
    so TWO pairs are illegal -- e2 and e4, the self-loops passing BOTH filters.
    That the answer needs 5-2 and not 5-1 is the point: the correction subtracts a
    COUNT, not a "there is a loop" flag. e3 passes the first filter only and must
    not be subtracted at all.
    """
    _skip_unless_engine(engine)
    graph = _two_filter_loop_graph(engine)
    match = "MATCH (a)-[{rel:'K'}]->(b)-[{w:1}]->(c)"
    rows = _rows(graph.gfql(f"{match} RETURN a.id, b.id, c.id", engine=engine))
    counted = _rows(graph.gfql(f"{match} RETURN count(*) AS n", engine=engine))
    assert len(rows) == 3
    assert counted == [{"n": 3}]


def _triple_labeled_loop_graph(engine: str) -> Any:
    """n2 carries ALL THREE labels and a self-loop; n4 carries only :B.

    Nodes: n1(:A), n2(:A:B:C), n3(:C), n4(:B)
    Edges: e0 = n1->n2, e1 = n2->n3, e2 = n2->n2, e3 = n4->n4
    """
    return _mk(
        pd.DataFrame({
            "id": ["n1", "n2", "n3", "n4"],
            "label__A": [True, True, False, False],
            "label__B": [False, True, False, True],
            "label__C": [False, True, True, False],
        }),
        pd.DataFrame({"s": ["n1", "n2", "n2", "n4"], "d": ["n2", "n3", "n2", "n4"]}),
        engine,
    )


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_two_hop_count_subtracts_only_loops_inside_all_three_node_domains(engine: str) -> None:
    """Hand oracle over ``MATCH (a:A)-->(b:B)-->(c:C)``.

    Domains: A={n1,n2}, B={n2,n4}, C={n2,n3}. Ordered DISTINCT-relationship pairs:
      r1=e0 (n1->n2, n1 in A) then r2 in {e1 (->n3 in C), e2 (->n2 in C)}   2
      r1=e2 (n2->n2, n2 in A) then r2 in {e1}  (e2 already bound)           1
      r1=e3 (n4->n4): n4 is not in A, so e3 is not a first-arm edge         0
    Total 3. Degree product is 4, so ONE pair is illegal: e2, whose node n2 sits
    in all three domains. e3's node n4 is only in B and must NOT be subtracted.
    """
    _skip_unless_engine(engine)
    graph = _triple_labeled_loop_graph(engine)
    match = "MATCH (a:A)-->(b:B)-->(c:C)"
    rows = _rows(graph.gfql(f"{match} RETURN a.id, b.id, c.id", engine=engine))
    counted = _rows(graph.gfql(f"{match} RETURN count(*) AS n", engine=engine))
    assert len(rows) == 3
    assert counted == [{"n": 3}]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_two_hop_count_ignores_duplicate_rows_in_a_node_domain(engine: str) -> None:
    """CONTROL for the ``ids_of`` dedup removal: a node listed TWICE in the node
    frame is still ONE node, so the count is unchanged.

    Same graph and oracle as the all-three-domains cell (3), with n2 duplicated.
    A domain semi-join tests existence, so the duplicate must not multiply rows.
    """
    _skip_unless_engine(engine)
    nodes = pd.DataFrame({
        "id": ["n1", "n2", "n2", "n3", "n4"],
        "label__A": [True, True, True, False, False],
        "label__B": [False, True, True, False, True],
        "label__C": [False, True, True, True, False],
    })
    edges = pd.DataFrame({"s": ["n1", "n2", "n2", "n4"], "d": ["n2", "n3", "n2", "n4"]})
    graph = _mk(nodes, edges, engine)
    counted = _rows(graph.gfql("MATCH (a:A)-->(b:B)-->(c:C) RETURN count(*) AS n", engine=engine))
    assert counted == [{"n": 3}]


def _dense_self_loop_graph(engine: str) -> Any:
    """Dense integer ids 0..2 with a typed self-loop, so a DEGREE FACT is buildable
    whose interval covers the whole node domain (what the O(1) branch requires).

    Edges: e0 = 0->1, e1 = 1->2, e2 = 2->2 (self-loop). All type K.
    """
    return _mk(
        pd.DataFrame({"id": [0, 1, 2]}),
        pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 2], "type": ["K", "K", "K"]}),
        engine,
    )


_DENSE_MATCH = "MATCH (a)-[:K]->(b)-[:K]->(c)"
#: (e0,e1) = 0,1,2 and (e1,e2) = 1,2,2. (e2,e2) would bind e2 twice; (e2,e1) does not
#: meet. The uncorrected degree product is 3: indeg(1)*outdeg(1) = 1 and
#: indeg(2)*outdeg(2) = 2*1. Exactly one self-loop, so the oracle is 3 - 1.
_DENSE_ORACLE = 2


# cudf is absent here on purpose: building a degree fact runs cupy's bincount, which
# needs the NVRTC runtime. Where that runtime is missing the cell would fail for an
# environment reason and say nothing about this contract.
@pytest.mark.parametrize("engine", ENGINES)
def test_two_hop_count_over_precomputed_degree_fact_subtracts_its_self_loops(engine: str) -> None:
    """The O(1) degree-fact branch must apply the SAME correction as the scan branches."""
    _skip_unless_engine(engine)
    from graphistry.compute.gfql.index.api import get_registry

    graph = _dense_self_loop_graph(engine).gfql_index_col_stats(edge_type_column="type", engine=engine)
    facts = get_registry(graph).degrees
    assert facts, "fixture no longer builds a degree fact; the branch under test is unreachable"
    assert [fact.self_loops for fact in facts.values()] == [1]

    rows = _rows(graph.gfql(f"{_DENSE_MATCH} RETURN a.id, b.id, c.id", engine=engine))
    counted = _rows(graph.gfql(f"{_DENSE_MATCH} RETURN count(*) AS n", engine=engine))
    assert len(rows) == _DENSE_ORACLE
    assert counted == [{"n": _DENSE_ORACLE}]


def test_dense_two_hop_kernel_never_reads_unknown_self_loops_as_zero() -> None:
    """``self_loops=None`` means UNKNOWN, so the O(1) degree-fact branch must step
    aside and let the scan lane count the loops -- NOT read the unknown as 0 and
    answer the uncorrected degree product (which is 3 here, not the oracle 2).

    Positive twin: the same fact with ``self_loops=1`` answers the oracle from the
    fact branch itself.
    """
    from dataclasses import replace
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.api import get_registry
    from graphistry.compute.gfql_fast_paths import _two_hop_equal_domain_dense_total

    graph = _dense_self_loop_graph("pandas").gfql_index_col_stats(edge_type_column="type")
    fact = next(iter(get_registry(graph).degrees.values()))
    assert fact.self_loops == 1
    nodes, edges = graph._nodes, graph._edges

    def total(degree_fact: Any) -> Any:
        return _two_hop_equal_domain_dense_total(
            nodes, edges, node_col="id", src_col="s", dst_col="d", engine=Engine.PANDAS,
            edge_endpoint_facts=None, domain_interval_hint=(0, 2), degree_fact=degree_fact,
        )

    assert total(fact) == _DENSE_ORACLE
    assert total(replace(fact, self_loops=None)) == _DENSE_ORACLE
    # and the branch really is the thing under test: a fact whose loop count is a LIE
    # changes the answer, so the O(1) branch -- not the scan -- served the first assert.
    assert total(replace(fact, self_loops=0)) == _DENSE_ORACLE + 1


@pytest.mark.parametrize("query,vetoes", [
    # POSITIVE: shortestPath binds one representative route per endpoint pair, a
    # cardinality binding rows cannot reproduce -- so it must keep the source-table lane.
    pytest.param("MATCH (a), (b), p = shortestPath((a)-[*]-(b)) RETURN a.id",
                 True, id="undirected_unbounded_shortest_path_vetoes"),
    # NEGATIVE: a plain undirected unbounded arm is NOT a shortestPath and belongs on
    # the trail-filtered binding lane; vetoing it is what produced the wrong row count.
    pytest.param("MATCH (a)-[*]-(b) RETURN a.id",
                 False, id="plain_undirected_unbounded_does_not_veto"),
    # NEGATIVE: the veto is scoped to the UNDIRECTED unbounded arm, not to shortestPath.
    pytest.param("MATCH (a), (b), p = shortestPath((a)-[*]->(b)) RETURN a.id",
                 False, id="directed_shortest_path_does_not_veto"),
])
def test_binding_row_veto_is_scoped_to_undirected_unbounded_shortest_path(
    query: str, vetoes: bool
) -> None:
    from graphistry.compute.gfql.cypher.parser import parse_cypher
    from graphistry.compute.gfql.cypher.lowering import _binds_one_route_per_pair_undirected

    parsed = parse_cypher(query)
    assert any(_binds_one_route_per_pair_undirected(c) for c in parsed.matches) is vetoes


def test_all_shortest_paths_is_still_a_typed_decline() -> None:
    """The veto's sibling spelling never reaches lowering -- the parser declines it --
    so `allShortestPaths` has no binding-row lane to be scoped out of."""
    from graphistry.compute.exceptions import GFQLValidationError
    from graphistry.compute.gfql.cypher.parser import parse_cypher

    with pytest.raises(GFQLValidationError) as err:
        parse_cypher("MATCH (a), (b), p = allShortestPaths((a)-[*]-(b)) RETURN a.id")
    assert "allShortestPaths" in str(err.value)


# (id, node ids, edge src, edge dst, hand oracle, anti-vacuous at the merge base)
_DEGENERATE_SHAPES = [
    # ONE self-loop and nothing else: a two-hop needs two DISTINCT relationships, so
    # there is no binding. The uncorrected degree product is 1x1 = 1.
    pytest.param(["a"], ["a"], ["a"], 0, id="single_self_loop"),
    # TWO PARALLEL self-loops at one node: the ordered distinct pairs (e0,e1),(e1,e0).
    # The correction must subtract the loop COUNT (2), not a "has a loop" flag:
    # the uncorrected product is 2x2 = 4.
    pytest.param(["a"], ["a", "a"], ["a", "a"], 2, id="parallel_self_loops"),
    # DISCONNECTED components, each a self-loop plus an out-edge: (e0,e1) in one and
    # (e2,e3) in the other. Uncorrected product 4, two loops.
    pytest.param(["a", "b", "c", "d"], ["a", "a", "c", "c"], ["a", "b", "c", "d"], 2,
                 id="two_components_each_with_a_loop"),
    # CONTROL (passes at the merge base too): no edges, so openCypher's
    # count-over-no-rows 0 -- there is nothing for the correction to get wrong.
    pytest.param(["a", "b"], [], [], 0, id="control_empty_edge_frame"),
    # CONTROL: a NULL endpoint is not equal to anything, itself included, so it is
    # neither a self-loop nor a join partner.
    pytest.param(["a", "b"], ["a", None], ["b", "b"], 0, id="control_null_endpoint"),
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("ids,src,dst,oracle", _DEGENERATE_SHAPES)
def test_two_hop_count_on_degenerate_self_loop_shapes(
    engine: str, ids: List[Any], src: List[Any], dst: List[Any], oracle: int
) -> None:
    _skip_unless_engine(engine)
    edges = pd.DataFrame({"s": src, "d": dst}).astype({"s": object, "d": object})
    graph = _mk(pd.DataFrame({"id": ids}), edges, engine)
    match = "MATCH (a)-->(b)-->(c)"
    assert len(_rows(graph.gfql(f"{match} RETURN a.id, b.id, c.id", engine=engine))) == oracle
    assert _rows(graph.gfql(f"{match} RETURN count(*) AS n", engine=engine)) == [{"n": oracle}]


# cudf is absent for the same reason as the degree-fact cell: the dense-integer kernel
# reduces through cupy, which needs the NVRTC runtime.
@pytest.mark.parametrize("engine", ENGINES)
def test_dense_integer_two_hop_count_subtracts_every_self_loop(engine: str) -> None:
    """DENSE-integer kernel with TWO self-loops at one node.

    Nodes 0,1. Edges e0 = 0->0, e1 = 0->0 (parallel loops), e2 = 0->1.
    Ordered DISTINCT-relationship pairs with ``dst(r1) == src(r2)``:
      r1=e0 -> r2 in {e1, e2}     2
      r1=e1 -> r2 in {e0, e2}     2
      r1=e2 (ends 1, out-degree 0)  0
    Total 4. Uncorrected the kernel answers indeg(0)*outdeg(0) = 2*3 = 6, so the
    correction is 2 -- the loop COUNT. Subtracting a flag would answer 5.
    """
    _skip_unless_engine(engine)
    graph = _mk(
        pd.DataFrame({"id": [0, 1]}),
        pd.DataFrame({"s": [0, 0, 0], "d": [0, 0, 1]}),
        engine,
    )
    match = "MATCH (a)-->(b)-->(c)"
    assert len(_rows(graph.gfql(f"{match} RETURN a.id, b.id, c.id", engine=engine))) == 4
    assert _rows(graph.gfql(f"{match} RETURN count(*) AS n", engine=engine)) == [{"n": 4}]


def test_fused_polars_lane_reads_an_empty_scalar_sink_as_zero() -> None:
    """CONTROL (0 is also the merge-base answer): distinct node domains put the count
    on the fused polars lane, and no middle node matches, so BOTH of that lane's
    one-cell sinks collect EMPTY. openCypher counts over no rows as 0, so an empty
    sink must read as 0 rather than raising on a missing cell.
    """
    pytest.importorskip("polars")
    graph = _mk(
        pd.DataFrame({
            "id": ["n1", "n2", "n3"],
            "label__A": [True, False, False],
            "label__B": [False, True, False],
            "label__C": [False, False, True],
        }),
        pd.DataFrame({"s": ["n1"], "d": ["n1"]}),
        "polars",
    )
    assert _rows(graph.gfql("MATCH (a:A)-->(b:B)-->(c:C) RETURN count(*) AS n", engine="polars")) == [{"n": 0}]


def test_fused_polars_lane_declines_to_derive_the_correction_across_two_edge_frames() -> None:
    """``illegal_pairs=None`` asks the lane to DERIVE the correction, which it can only
    do from a single shared edge frame: an illegal pair must pass both relationship
    filters, and neither arm frame alone answers that. On distinct frames it must hand
    the shape back (None), not derive a wrong number from the first arm.

    Positive twin: the same call with ONE shared frame answers instead of declining.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql_fast_paths import _two_hop_count_fused_polars

    nodes = pl.DataFrame({"id": ["n1", "n2", "n3"]})
    first = pl.DataFrame({"s": ["n1", "n2"], "d": ["n2", "n2"]})
    second = pl.DataFrame({"s": ["n2"], "d": ["n3"]})

    def call(second_edges: Any) -> Any:
        return _two_hop_count_fused_polars(
            nodes, nodes, nodes, first, second_edges,
            node_col="id", src_col="s", dst_col="d", alias="n", illegal_pairs=None,
        )

    assert call(second) is None
    assert call(first) is not None


def test_edge_identity_column_already_on_the_frame_is_reused_not_overwritten() -> None:
    """A user column spelled like the internal edge-identity column is the identity the
    trail join reads; re-deriving it would silently rewrite the caller's data.
    """
    from graphistry.compute.gfql.same_path_types import EDGE_IDENTITY_COLUMN
    from graphistry.compute.gfql_unified import _with_edge_identity
    from graphistry.Engine import Engine

    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], EDGE_IDENTITY_COLUMN: [70, 80]})
    graph = graphistry.nodes(pd.DataFrame({"id": ["a", "b", "c"]}), "id").edges(edges, "s", "d")
    out = _with_edge_identity(graph, engine=Engine.PANDAS)
    assert list(out._edges[EDGE_IDENTITY_COLUMN]) == [70, 80]
