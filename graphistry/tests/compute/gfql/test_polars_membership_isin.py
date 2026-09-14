"""#2082: ``membership.is_in_ids`` — the ONE spelling of id-set membership for the polars engine.

polars < 1.28 rejects a List-typed (``implode()``) ``is_in`` RHS whose length differs from the
column (``ComputeError: shapes don't match: expected N elements in 'is_in' comparison, got 1``);
polars >= 1.28 deprecates the bare-``Series`` RHS. RAPIDS 25.02 ships polars 1.21, RAPIDS 26.02
ships 1.35, so the engine must be correct AND warning-free on both.

Oracles are the python-list form ``expr.is_in(ids.to_list())`` (never warned, never failed on
any 1.21..1.35 release in the plans/gfql-2082-polars121 sweep) and hand-written literals.
Engine agreement is not used as evidence.
"""
import warnings

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.lazy.engine.polars import membership
from graphistry.compute.gfql.lazy.engine.polars.membership import (
    IMPLODED_RHS_FLOOR, imploded_rhs_supported, is_in_ids,
)
from .polars_test_utils import edge_pair_set

pl = pytest.importorskip("polars")


# --- the version predicate is a pure function of the version string -------------------------

@pytest.mark.parametrize("version,expected", [
    ("1.21.0", False), ("1.27.1", False), ("1.28.0rc1", False),
    ("1.28.0", True), ("1.28.1", True), ("1.29.0", True), ("1.35.2", True), ("2.0.0", True),
])
def test_imploded_rhs_supported_switches_exactly_at_1_28_0(version, expected):
    assert imploded_rhs_supported(version) is expected
    assert str(IMPLODED_RHS_FLOOR) == "1.28.0"


def test_installed_polars_branch_matches_the_predicate():
    assert membership._installed_polars_implodes() is imploded_rhs_supported(pl.__version__)


def _deprecations(caught):
    """polars raises its ``is_in`` deprecation from Rust: under a ``-W error`` filter it is PRINTED,
    not raised (verified against a mutant), so only record-and-assert can observe it.
    ``CategoricalRemappingWarning`` (local categoricals) is a perf hint, not part of the contract."""
    return [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


# --- set semantics + no warning on the INSTALLED polars, every dtype/shape ------------------

def _scenarios():
    # (label, endpoint values, id values, dtype, expected kept mask from a hand oracle)
    yield "int/mixed", [1, 2, 9, None], [1, 2, 3], pl.Int64, [True, True, False, None]
    yield "int/empty-ids", [1, 2], [], pl.Int64, [False, False]
    yield "int/null-in-ids", [1, 2, None], [1, None], pl.Int64, [True, False, None]
    yield "str/mixed", ["a", "zz", None], ["a", "b"], pl.Utf8, [True, False, None]
    yield "str/empty-ids", ["a"], [], pl.Utf8, [False]
    yield "cat/mixed", ["a", "zz", None], ["a", "b"], pl.Categorical, [True, False, None]
    yield "cat/empty-ids", ["a"], [], pl.Categorical, [False]
    yield "float/mixed", [1.0, 2.5, None], [1.0, 3.0], pl.Float64, [True, False, None]


@pytest.mark.parametrize("label,vals,ids,dtype,expected", list(_scenarios()),
                         ids=[s[0] for s in _scenarios()])
def test_is_in_ids_is_set_membership_and_never_warns(label, vals, ids, dtype, expected):
    df = pl.DataFrame({"x": pl.Series(vals, dtype=dtype)})
    id_series = pl.Series("id", ids, dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = df.select(is_in_ids(pl.col("x"), id_series).alias("m")).get_column("m").to_list()
    oracle = df.select(pl.col("x").is_in(id_series.to_list()).alias("m")).get_column("m").to_list()
    assert got == expected, label
    assert got == oracle, label
    assert _deprecations(caught) == [], label


def test_is_in_ids_keeps_callers_fill_null_contract():
    """Sites chain ``.fill_null(False)``: a null endpoint must come out null BEFORE that fill."""
    df = pl.DataFrame({"x": pl.Series([None, 1], dtype=pl.Int64)})
    ids = pl.Series("id", [1], dtype=pl.Int64)
    raw = df.select(is_in_ids(pl.col("x"), ids).alias("m")).get_column("m").to_list()
    filled = df.select(is_in_ids(pl.col("x"), ids).fill_null(False).alias("m")).get_column("m").to_list()
    assert raw == [None, True] and filled == [False, True]


def test_is_in_ids_on_a_lazyframe_and_a_large_universe():
    """The RHS is a literal, not a same-length column: an id set far longer than the frame."""
    ids = pl.Series("id", list(range(0, 100_000, 2)), dtype=pl.Int64)
    lf = pl.LazyFrame({"x": pl.Series([0, 1, 99_998, 99_999, 250_000], dtype=pl.Int64)})
    got = lf.filter(is_in_ids(pl.col("x"), ids)).collect().get_column("x").to_list()
    assert got == [0, 99_998]


# --- the routed sites, end to end on the polars engine ---------------------------------------

_NODES = pd.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "a"]})
# every endpoint-miss shape: closed, dangling dst, dangling src, both missing, self-loop on missing
_EDGES = pd.DataFrame({"s": [0, 1, 7, 8, 9], "d": [1, 5, 2, 6, 9]})
_CLOSED = {(0, 1)}


def _ids(values, id_dtype) -> "pl.Series":
    """Int64 literals re-typed as the id dtype under test (Categorical only casts from strings)."""
    e = pl.Series(list(values), dtype=pl.Int64)
    return e.cast(pl.Utf8).cast(id_dtype) if id_dtype in (pl.Utf8, pl.Categorical) else e.cast(id_dtype)


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, id_dtype):
    n_pl = pl.from_pandas(nodes).with_columns(_ids(nodes["id"], id_dtype).alias("id"))
    e_pl = pl.from_pandas(edges).with_columns(_ids(edges["s"], id_dtype).alias("s"), _ids(edges["d"], id_dtype).alias("d"))
    return graphistry.nodes(n_pl, "id").edges(e_pl, "s", "d")


def _pairs(pairs, id_dtype):
    return {tuple(_ids(p, id_dtype).to_list()) for p in pairs}


@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.Utf8, pl.Categorical], ids=["int", "str", "cat"])
def test_polars_hop_endpoint_gate_runs_on_the_installed_polars(direction, id_dtype):
    """Unseeded hop through ``_keep_edges_with_both_endpoints_resolvable``: only the closed edge
    survives, in every direction, for int / string / categorical ids (hand oracle)."""
    g = _bind(_NODES, _EDGES, id_dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = g.hop(direction=direction, engine="polars")
    assert edge_pair_set(out) == _pairs(_CLOSED, id_dtype)
    assert _deprecations(caught) == []


@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.Utf8], ids=["int", "str"])
def test_polars_seeded_hop_with_a_target_wavefront_gate(id_dtype):
    """Seeded 2-hop: the universe is the node table widened by the wavefront; a 1-row seed
    is the degenerate 'got 1' shape that tripped polars 1.21."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3]})
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4]})  # 3->4 dangles
    g = _bind(nodes, edges, id_dtype)
    seed = pl.DataFrame({"id": _ids([0], id_dtype)})
    out = g.hop(nodes=seed, hops=2, direction="forward", engine="polars")
    assert edge_pair_set(out) == _pairs({(0, 1), (1, 2)}, id_dtype)


def test_polars_hop_empty_node_table_over_edges_keeps_nothing():
    g = _bind(_NODES.iloc[0:0], _EDGES, pl.Int64)
    out = g.hop(direction="forward", engine="polars")
    assert out._edges.height == 0
