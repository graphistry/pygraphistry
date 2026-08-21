"""Whole-entity endpoint projection keeps openCypher bag multiplicity (#1994).

`MATCH (a)-->(b) RETURN b` projects one endpoint of a relationship pattern as a
whole entity. openCypher RETURN operates on the BAG of pattern-match rows, so the
answer has one row per match -- the same count every *property* spelling of the
projection already returned. The old lowering answered the deduplicated node set.

EVERY expected value below is hand-derived from the fixture; engine agreement is
not evidence, and none of these numbers were read off the implementation.

Fixture A (`_graph`): nodes 1..5, edges (1,2) (1,3) (2,3) (3,4).
  Match bag for (a)-->(b), one row per edge:
      (1,2) (1,3) (2,3) (3,4)                       -- 4 rows
  a-side bag [1,1,2,3]   (node 1 has two out-edges)
  b-side bag [2,3,3,4]   (node 3 is bound twice: from 1 and from 2)
  DISTINCT b             [2,3,4]
  Two-hop (a)-->(b)-->(c): (1,2)->(2,3); (1,3)->(3,4); (2,3)->(3,4)
      c-side bag [3,4,4]                            -- 3 rows
  No relationship at all: MATCH (a) RETURN a is the node table, [1,2,3,4,5].

Fixture B (`_parallel_graph`): nodes 1..3, edges (1,2) (1,2) (2,3).
  Two PARALLEL edges 1->2, so the match bag is (1,2) (1,2) (2,3) -- 3 rows,
  b-side [2,2,3], a-side [1,1,2]. A node-set answer collapses this to [2,3].
"""
from typing import List, Optional

import pandas as pd
import pytest

import graphistry

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

ENGINES = [
    "pandas",
    pytest.param("polars", marks=polars_only),
    pytest.param("cudf", marks=cudf_only),
]

_NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Ann", "Bob", "Cat", "Dan", "Eve"],
})
_EDGES = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 3, 3, 4]})

_PARALLEL_NODES = pd.DataFrame({"id": [1, 2, 3], "name": ["Ann", "Bob", "Cat"]})
_PARALLEL_EDGES = pd.DataFrame({"s": [1, 1, 2], "d": [2, 2, 3]})


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str):
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        import cudf as _cudf
        return graphistry.nodes(_cudf.from_pandas(nodes), "id").edges(_cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _run(query: str, engine: str, *, parallel: bool = False) -> pd.DataFrame:
    g = (
        _bind(_PARALLEL_NODES, _PARALLEL_EDGES, engine)
        if parallel
        else _bind(_NODES, _EDGES, engine)
    )
    out = g.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.reset_index(drop=True)


def _bag(df: pd.DataFrame, column: str) -> List[Optional[int]]:
    # py3.13+ pandas renders a null as float nan where 3.12 gave None; normalize per
    # value (never df.where(df.notna(), None), which upcasts whole columns).
    def one(v: object) -> Optional[int]:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return int(v)  # type: ignore[arg-type]
    return sorted((one(v) for v in df[column]), key=lambda x: (x is None, x))


# ===========================================================================
# The defect: one whole-entity output over a relationship pattern
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,column,expected", [
    ("MATCH (a)-->(b) RETURN b", "b.id", [2, 3, 3, 4]),
    ("MATCH (a)-->(b) RETURN a", "a.id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) RETURN b AS n", "n.id", [2, 3, 3, 4]),
    ("MATCH (a)-->(b)-->(c) RETURN c", "c.id", [3, 4, 4]),
], ids=["dst", "src", "aliased", "two_hop_dst"])
def test_whole_entity_endpoint_projection_keeps_bag(query, column, expected, engine):
    """One row per pattern match, not the deduplicated node set (fixture A)."""
    assert _bag(_run(query, engine), column) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,column,expected", [
    ("MATCH (a)-->(b) RETURN b", "b.id", [2, 2, 3]),
    ("MATCH (a)-->(b) RETURN a", "a.id", [1, 1, 2]),
], ids=["dst", "src"])
def test_whole_entity_projection_counts_parallel_edges(query, column, expected, engine):
    """Fixture B: two parallel 1->2 edges are two matches, so node 2 appears twice.

    A node-set answer cannot represent this at all -- it collapses to [2, 3]."""
    assert _bag(_run(query, engine, parallel=True), column) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_with_sibling_property_output(engine):
    """`RETURN b, b.id AS x` mixes a whole entity with a property of the same alias;
    both must see the 4-row bag, and the flattened `b.id` must agree with `x`."""
    df = _run("MATCH (a)-->(b) RETURN b, b.id AS x", engine)
    assert _bag(df, "b.id") == [2, 3, 3, 4]
    assert _bag(df, "x") == [2, 3, 3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_alias_whole_entity_projection_renders(engine):
    """`RETURN a, b` binds both endpoints of every match: 4 rows, paired
    (1,2) (1,3) (2,3) (3,4). polars declined this shape outright before the
    projector learned to read multi-entity binding rows."""
    df = _run("MATCH (a)-->(b) RETURN a, b", engine)
    got = sorted((int(r["a.id"]), int(r["b.id"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (1, 3), (2, 3), (3, 4)]


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_carries_every_field(engine):
    """The flattened entity keeps its non-id fields row-aligned with the bag:
    b-side ids [2,3,3,4] are names [Bob, Cat, Cat, Dan]."""
    df = _run("MATCH (a)-->(b) RETURN b", engine)
    got = sorted((int(r["b.id"]), str(r["b.name"])) for r in df.to_dict("records"))
    assert got == [(2, "Bob"), (3, "Cat"), (3, "Cat"), (4, "Dan")]


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_ordered_bag(engine):
    """ORDER BY over the whole-entity bag keeps every row: ordered compare."""
    df = _run("MATCH (a)-->(b) RETURN b ORDER BY b.id", engine)
    assert [int(v) for v in df["b.id"]] == [2, 3, 3, 4]


# ===========================================================================
# Negative controls: shapes that must NOT change
# ===========================================================================


@pytest.mark.parametrize("engine", ENGINES)
def test_distinct_whole_entity_still_dedupes(engine):
    """DISTINCT is the user-requested dedup: [2,3,4], not the [2,3,3,4] bag."""
    assert _bag(_run("MATCH (a)-->(b) RETURN DISTINCT b", engine), "b.id") == [2, 3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_without_relationship_unchanged(engine):
    """No relationship pattern means no multiplicity to keep: the node table."""
    assert _bag(_run("MATCH (a) RETURN a", engine), "a.id") == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_after_where(engine):
    """A row-level WHERE prunes matches, then the surviving bag is projected:
    (1,3) (2,3) (3,4) keep b.id >= 3, so the a-side is [1,2,3]."""
    assert _bag(_run("MATCH (a)-->(b) WHERE b.id >= 3 RETURN a", engine), "a.id") == [1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_property_projection_bag_unchanged(engine):
    """Control: the property spelling was already bag-correct and must stay so."""
    assert _bag(_run("MATCH (a)-->(b) RETURN b.id AS x", engine), "x") == [2, 3, 3, 4]


@pytest.mark.parametrize("engine", ["pandas", pytest.param("polars", marks=polars_only)])
def test_variable_length_whole_entity_projection_unchanged(engine):
    """A variable-length arm keeps the node-set lane. Its openCypher bag is the
    relationship-unique WALK expansion, not the edge bag this lane counts, so widening it
    here would swap one unvalidated answer for another. Pins the scope of the lane switch.

    Fixture C: edges p0->p1, p1->p2, p2->p4, p1->p0. The backtracked seed p0 is reachable
    at distance 2 but its hop label is null, so the aliased node set is exactly p1 and p2
    (the same answer as before the lane switch). cuDF is excluded: it answers
    ['p0','p1','p2'] here on master too -- a pre-existing variable-length alias-gate
    divergence this projection change neither causes nor touches."""
    edges = pd.DataFrame({"s": ["p0", "p1", "p2", "p1"], "d": ["p1", "p2", "p4", "p0"]})
    if engine == "polars":
        g = graphistry.edges(pl.from_pandas(edges), "s", "d").materialize_nodes(engine="polars")
    else:
        g = graphistry.edges(edges, "s", "d").materialize_nodes()
    out = g.gfql("MATCH (a {id: 'p0'})-[*1..2]-(b) RETURN b", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert sorted(out["b.id"].tolist()) == ["p1", "p2"]


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_carry_into_reentry_unchanged(engine):
    """A whole-row WITH carry feeds a trailing MATCH, which cannot yet tell matched
    from unmatched rows apart on a duplicated prefix, so that carry stays on the
    node-set lane. Pins the scope of the lane switch: re-entry keeps compiling and
    answering exactly as before (#1935 item 1 remains open)."""
    df = _run(
        "MATCH (a)-->(c) WITH a AS p OPTIONAL MATCH (p)-->(z) RETURN p.id AS pid, z.id AS zid",
        engine,
    )
    assert len(df) == 4
