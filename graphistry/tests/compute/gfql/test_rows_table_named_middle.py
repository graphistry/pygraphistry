"""`rows(table=...)` must survive a NAMED middle — on both chain surfaces.

The named-middle rewrite turns `[... named ops ..., rows()]` into
`rows(binding_ops=<middle>)` so a Cypher multi-alias RETURN lowers to a bindings table.
It skipped that rewrite when the call already carried `binding_ops`, `source` or
`alias_endpoints` — but NOT when it carried `table`, which names the output table just as
explicitly. Two distinct failures followed, and the quiet one is the worse:

  * ODD-length named middle -> the rewrite fired and the caller got the BINDINGS table
    instead of the edges table. No error; `table=` was simply ignored.
  * EVEN-length named middle (a path ending on an EDGE — LDBC IS3's edge lookup is exactly
    `(person)-[r:KNOWS]-`) -> the rewritten op list is not an alternating node/edge path, so
    validation hard-errored with "require ... a single connected alternating node/edge path".

Pinned on both engines because the rewrite is duplicated in `compute/chain.py` (generic) and
`gfql/lazy/engine/polars/chain.py` (native polars); fixing one alone leaves the other wrong.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute import ast

pl = pytest.importorskip("polars")


NODES = pd.DataFrame({"id": [0, 1, 2, 3], "firstName": ["a", "b", "c", "d"]})
EDGES = pd.DataFrame({"s": [0, 0, 1], "d": [1, 2, 3],
                      "type": ["KNOWS"] * 3, "creationDate": [30, 10, 20]})

EDGE_COLS = {"s", "d", "type", "creationDate"}


def _graph(engine):
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(
            pl.from_pandas(EDGES), "s", "d")
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _cols(frame):
    return set(frame.columns)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rows_table_edges_after_named_middle_ending_on_an_edge(engine):
    """LDBC IS3's edge lookup: `(n {id})-[r:KNOWS]-` then `rows(table='edges')`.

    The middle is [node, edge] — EVEN length, so the rewrite produced a non-alternating
    binding_ops list and the query raised instead of returning the edges.
    """
    g = _graph(engine)
    out = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.rows(table="edges"),
    ], engine=engine)._nodes
    assert EDGE_COLS <= _cols(out), \
        f"[{engine}] expected the EDGES table, got columns {sorted(_cols(out))}"
    assert len(out) == 2, f"[{engine}] expected node 0's two KNOWS edges, got {len(out)}"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rows_table_edges_after_named_middle_of_odd_length(engine):
    """The SILENT half: an odd-length named middle returned the bindings table instead.

    This is the case that produced no error at all — `table='edges'` was ignored and the
    caller got alias columns (`n`, `r.type`, `f`, ...). Asserting the edge columns are
    present AND the alias columns are absent is what distinguishes the two tables.
    """
    g = _graph(engine)
    out = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.n(name="f"),
        ast.rows(table="edges"),
    ], engine=engine)._nodes
    cols = _cols(out)
    assert EDGE_COLS <= cols, \
        f"[{engine}] expected the EDGES table, got columns {sorted(cols)}"
    assert not any(c.startswith("r.") for c in cols), \
        f"[{engine}] bindings columns leaked into a rows(table='edges') result: {sorted(cols)}"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rows_table_edges_is_unaffected_by_whether_the_middle_is_named(engine):
    """Naming an op is a projection concern; it must not change WHICH table comes back."""
    g = _graph(engine)
    named = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.rows(table="edges"),
    ], engine=engine)._nodes
    unnamed = g.gfql([
        ast.n({"id": 0}),
        ast.e_undirected({"type": "KNOWS"}),
        ast.rows(table="edges"),
    ], engine=engine)._nodes
    assert EDGE_COLS <= _cols(unnamed)
    assert EDGE_COLS <= _cols(named)
    assert len(named) == len(unnamed), \
        f"[{engine}] naming the middle changed the row count: {len(named)} vs {len(unnamed)}"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_bare_rows_after_a_named_middle_still_gets_the_bindings_table(engine):
    """The NEGATIVE side of the boundary: the rewrite must still fire without `table=`.

    Without this, the fix could be "never rewrite", which would silently break every
    Cypher multi-alias RETURN — the thing the rewrite exists to serve.
    """
    g = _graph(engine)
    out = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.n(name="f"),
        ast.rows(),
    ], engine=engine)._nodes
    cols = _cols(out)
    assert any(c == "n" or c.startswith("n.") for c in cols), \
        f"[{engine}] expected a bindings table with alias columns, got {sorted(cols)}"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_explicit_table_nodes_still_rewrites_and_that_is_a_known_limitation(engine):
    """`rows(table='nodes')` DOES still rewrite, and that is deliberate — pin the limitation.

    `rows()` declares `table: str = "nodes"` and always emits it, so an explicit
    `rows(table='nodes')` and a bare `rows()` are byte-identical at the params level. The
    guard therefore cannot honour the explicit spelling without disabling the rewrite for
    every bare `rows()` — which is exactly what a first attempt at this fix did, breaking
    the IS6 bindings path (5 regressions, caught only by re-baselining against master).

    So the contract is: a NON-DEFAULT table opts out of the rewrite; `"nodes"` cannot.
    Distinguishing them would need `rows()` to default `table=None` and resolve later — a
    wire-format change, out of scope here. This test exists so that limitation is pinned
    rather than discovered again.
    """
    g = _graph(engine)
    out = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.n(name="f"),
        ast.rows(table="nodes"),
    ], engine=engine)._nodes
    cols = _cols(out)
    assert any(c == "n" or c.startswith("n.") for c in cols), \
        f"[{engine}] expected the bindings table (documented limitation), got {sorted(cols)}"
