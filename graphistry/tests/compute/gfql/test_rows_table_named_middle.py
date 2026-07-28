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

Pinned on all four engines because the rewrite is duplicated in `compute/chain.py` (generic)
and `gfql/lazy/engine/polars/chain.py` (native polars); fixing one alone leaves the other
wrong. cuDF runs the generic rewrite and polars-gpu the native one, so each surface is
covered by two engines rather than one — "which table comes back" is engine-agnostic
semantics, not a polars question.

ENGINE LIST IS FIXED, NOT PROBED-INTO-EXISTENCE. An engine that cannot run in the current
environment reports SKIPPED with a reason; it never vanishes from the report (the trap in
`polars_test_utils.available_nonpandas_engines()`, which silently shrinks). There is also no
module-level `pytest.importorskip("polars")` any more: it caused the whole file — pandas
params included — to be skipped in the `test-gfql-core` lane, which does not install polars.

COVERAGE BOUNDARY, stated rather than hidden behind a skip: no CI lane runs cuDF or
polars-gpu (`ci-gpu.yml` is hard-disabled and does not install the `polars` extra), so those
two parameters report SKIPPED on CI. They are exercised out of band on the dgx GPU box
against `graphistry/test-rapids-official:26.02-gfql-polars` (`docker run --gpus all` —
omitting `--gpus all` FABRICATES failures rather than skipping). Treat a green CI run as
evidence for pandas + polars only.
"""
from functools import lru_cache
from typing import Any, Optional, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.compute import ast
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason

PANDAS_API_ENGINES: Tuple[str, ...] = ("pandas", "cudf")
POLARS_API_ENGINES: Tuple[str, ...] = ("polars", "polars-gpu")
ENGINES: Tuple[str, ...] = PANDAS_API_ENGINES + POLARS_API_ENGINES


NODES = pd.DataFrame({"id": [0, 1, 2, 3], "firstName": ["a", "b", "c", "d"]})
EDGES = pd.DataFrame({"s": [0, 0, 1], "d": [1, 2, 3],
                      "type": ["KNOWS"] * 3, "creationDate": [30, 10, 20]})

EDGE_COLS = {"s", "d", "type", "creationDate"}


def _frame(engine: str, df: pd.DataFrame) -> Any:
    if engine in POLARS_API_ENGINES:
        pl = pytest.importorskip("polars")
        return pl.from_pandas(df)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(df)
    return df


def _graph(engine):
    return graphistry.nodes(_frame(engine, NODES), "id").edges(
        _frame(engine, EDGES), "s", "d")


@lru_cache(maxsize=None)
def _engine_skip_reason(engine: str) -> Optional[str]:
    """``None`` => this engine MUST run here; a string => a stated, checkable skip reason.

    Two traps this steers between, both hit by trying:

    * An IMPORT-only probe does not discriminate on a box where cudf / cudf_polars import
      against an incomplete CUDA runtime — construction and simple ops succeed and the suite
      then dies inside the first real kernel, which reads as a product failure.
    * A probe that SWALLOWS every exception is worse: reverting the production guard turned
      all 20 parameters here into SKIPS instead of failures, and an intermittent cold-start
      error in a fresh GPU container silently dropped cuDF from an otherwise-green run. A
      skipped GPU parameter reads as evidence of passing, which is exactly the theatre this
      file exists to avoid.

    So: a missing module skips, a recognisable GPU-stack error skips with its text quoted, and
    ANY other failure propagates. The smoke query is a plain traversal — never the shape under
    test, or a regression would disarm its own test.
    """
    return engine_skip_reason(
        engine,
        lambda: _graph(engine).gfql([ast.n(), ast.e_undirected(), ast.n()], engine=engine),
    )


def _require(engine: str) -> None:
    reason = _engine_skip_reason(engine)
    if reason is not None:
        pytest.skip(
            f"engine {engine!r} unavailable here ({reason}) — NOT evidence that it passes; "
            "see the COVERAGE BOUNDARY note in this module's docstring"
        )


def _cols(frame):
    return set(map(str, frame.columns))


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_table_edges_after_named_middle_ending_on_an_edge(engine):
    """LDBC IS3's edge lookup: `(n {id})-[r:KNOWS]-` then `rows(table='edges')`.

    The middle is [node, edge] — EVEN length, so the rewrite produced a non-alternating
    binding_ops list and the query raised instead of returning the edges.
    """
    _require(engine)
    g = _graph(engine)
    out = g.gfql([
        ast.n({"id": 0}, name="n"),
        ast.e_undirected({"type": "KNOWS"}, name="r"),
        ast.rows(table="edges"),
    ], engine=engine)._nodes
    assert EDGE_COLS <= _cols(out), \
        f"[{engine}] expected the EDGES table, got columns {sorted(_cols(out))}"
    assert len(out) == 2, f"[{engine}] expected node 0's two KNOWS edges, got {len(out)}"


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_table_edges_after_named_middle_of_odd_length(engine):
    """The SILENT half: an odd-length named middle returned the bindings table instead.

    This is the case that produced no error at all — `table='edges'` was ignored and the
    caller got alias columns (`n`, `r.type`, `f`, ...). Asserting the edge columns are
    present AND the alias columns are absent is what distinguishes the two tables.
    """
    _require(engine)
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


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_table_edges_is_unaffected_by_whether_the_middle_is_named(engine):
    """Naming an op is a projection concern; it must not change WHICH table comes back."""
    _require(engine)
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


@pytest.mark.parametrize("engine", ENGINES)
def test_bare_rows_after_a_named_middle_still_gets_the_bindings_table(engine):
    """The NEGATIVE side of the boundary: the rewrite must still fire without `table=`.

    Without this, the fix could be "never rewrite", which would silently break every
    Cypher multi-alias RETURN — the thing the rewrite exists to serve.
    """
    _require(engine)
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


@pytest.mark.parametrize("engine", ENGINES)
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
    _require(engine)
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
