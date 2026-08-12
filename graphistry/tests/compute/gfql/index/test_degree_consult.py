"""The degree-fact consult: slicing and identity.

Both traps pinned here are INVISIBLE to a value test, which is why they get
engagement pins instead:

* the consult SLICES the degree arrays to the domain interval, and an off-by-one
  silently MISCOUNTS -- a wrong answer, not a slow one;
* the fact must be validated against the BOUND edge frame. Anchoring identity to
  the transient filtered subset made every lookup miss: facts built, never used,
  answers still correct. It took an engagement probe to see it.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql_fast_paths as fp
from graphistry.compute.gfql.index.api import get_registry

Q = ("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})"
     "-[{rel:'F'}]->(c {kind:'P'}) RETURN count(*) AS n")


ENGINES = ["pandas", "polars", "cudf"]


def _to(df, engine):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return pl.from_pandas(df)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(df)
    return df


def _graph(n_p=3, n_c=3, engine="pandas", interleave=False):
    """``interleave=True`` shuffles id assignment across labels -- the layout every
    real ingest/join produces, and the one the board's generator happens NOT to.
    Block-contiguous fixtures let 13 pins pass while the real board built 0 facts."""
    ids = list(range(n_p + n_c))
    if interleave:
        rng = np.random.default_rng(0)
        ids = list(rng.permutation(n_p + n_c))
    nodes = pd.DataFrame({"id": ids,
                          "kind": ["P"] * n_p + ["C"] * n_c})
    p_ids = [ids[i] for i in range(min(3, n_p))]
    c_ids = [ids[n_p + i] for i in range(min(2, n_c))]
    while len(p_ids) < 3: p_ids.append(p_ids[0])
    while len(c_ids) < 2: c_ids.append(c_ids[0])
    edges = pd.DataFrame({"s": [p_ids[0], p_ids[1], p_ids[2], p_ids[0], p_ids[0]],
                          "d": [p_ids[1], p_ids[2], p_ids[0], c_ids[0], c_ids[1]],
                          "rel": ["F", "F", "F", "X", "X"]})
    return graphistry.nodes(_to(nodes, engine), "id").edges(_to(edges, engine), "s", "d")


def _run(g, query=Q, engine="pandas"):
    """(value, degree_fact_was_used)."""
    used = []
    real = fp._two_hop_equal_domain_dense_total

    def spy(*a, **k):
        used.append(k.get("degree_fact") is not None)
        return real(*a, **k)

    fp._two_hop_equal_domain_dense_total = spy
    try:
        out = g.gfql(query, engine=engine)._nodes
        val = out.to_pandas().iloc[0, 0] if hasattr(out, "to_pandas") else out.iloc[0, 0]
        return val, any(used)
    finally:
        fp._two_hop_equal_domain_dense_total = real


@pytest.mark.parametrize("engine", ENGINES)
def test_degree_fact_is_built_and_actually_used(engine: str) -> None:
    """Engagement, not just correctness: a built-but-unused fact returns the same
    answer, so only this assertion can tell the difference."""
    base = _graph(engine=engine)
    oracle, _ = _run(base, engine=engine)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel",
                                  engine=engine)
    assert [k for k in get_registry(g).degrees], "no degree facts built"
    value, used = _run(g, engine=engine)
    assert used, "degree fact built but never consulted -- silently dead"
    assert value == oracle


def test_identity_anchors_to_the_bound_frame_not_the_partition() -> None:
    """The regression that made every lookup miss. The fact is COUNTED over the
    rel='F' subset but DESCRIBES a partition of the bound edge frame, so it must
    validate against the frame the consult holds."""
    g = _graph().gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
    reg = get_registry(g)
    fact = reg.get_degree_valid("s", "d", g._edges, "pandas", "rel", "F") \
        or reg.get_degree_valid("s", "d", g._edges,
                                __import__("graphistry").Engine.PANDAS, "rel", "F")
    assert fact is not None, "fact does not validate against the BOUND edge frame"
    assert fact.source_ref is g._edges


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("n_p,n_c", [(3, 3), (5, 1), (2, 8), (7, 2)])
def test_slice_is_exact_across_domain_shapes(n_p: int, n_c: int, engine: str) -> None:
    """The slice picks [lo, hi] out of arrays built over the whole node space. An
    off-by-one at either end silently counts the wrong nodes, so vary where the
    P-domain sits relative to the full range."""
    base = _graph(n_p, n_c, engine=engine)
    oracle, _ = _run(base, engine=engine)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel",
                                  engine=engine)
    value, used = _run(g, engine=engine)
    assert used
    assert value == oracle


def test_gapped_node_space_builds_facts_and_stays_exact() -> None:
    """Density is NOT required for the degree arrays: ids absent from the span
    contribute ZERO to the dot, so a gapped node space builds valid facts. (The
    earlier version of this pin asserted the opposite -- a builder guard strictly
    tighter than the kernel it served, which built nothing on the real board,
    where the node space is gapped. The kernel proves its DOMAIN dense
    separately; the arrays only need the span.)"""
    nodes = pd.DataFrame({"id": [0, 1, 2, 10, 11], "kind": ["P"] * 3 + ["C"] * 2})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2], "rel": ["F", "F"]})
    base = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = ("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})"
         "-[{rel:'F'}]->(c {kind:'P'}) RETURN count(*) AS n")
    oracle, _ = _run(base, q)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
    assert [k for k in get_registry(g).degrees], "gapped space must now build facts"
    value, used = _run(g, q)
    assert value == oracle
    assert used, "P-domain [0,2] is dense, so the kernel must consult the fact"


@pytest.mark.parametrize("seed", range(6))
def test_differential_vs_the_scan_on_random_typed_graphs(seed: int) -> None:
    """Values must be identical with and without the fact, on arbitrary degree
    distributions -- the dot is only an optimization if it is also exact."""
    rng = np.random.default_rng(seed)
    n_p, n_c, m = int(rng.integers(3, 20)), int(rng.integers(1, 6)), int(rng.integers(1, 80))
    x = int(rng.integers(0, 8))
    nodes = pd.DataFrame({"id": list(range(n_p + n_c)),
                          "kind": ["P"] * n_p + ["C"] * n_c})
    edges = pd.DataFrame({
        "s": np.concatenate([rng.integers(0, n_p, m), rng.integers(0, n_p, x)]),
        "d": np.concatenate([rng.integers(0, n_p, m), rng.integers(n_p, n_p + n_c, x)]),
        "rel": np.concatenate([np.full(m, "F"), np.full(x, "X")])})
    base = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    oracle, _ = _run(base)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
    value, used = _run(g)
    assert used
    assert value == oracle


def test_interleaved_ids_currently_build_no_facts_and_stay_correct() -> None:
    """The real-board failure mode, now representable: interleaved ids mean the
    whole node space is dense but per-label ids are scattered... or gapped
    entirely. Today's builder must DECLINE (never miscount) and the answer must
    come from the scan. When positional degrees land, this test flips to
    asserting engagement -- that flip is the acceptance test for A1."""
    base = _graph(4, 3, interleave=True)
    oracle, _ = _run(base)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
    value, used = _run(g)
    assert value == oracle  # correctness regardless
