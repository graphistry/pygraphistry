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
    while len(p_ids) < 3:
        p_ids.append(p_ids[0])
    while len(c_ids) < 2:
        c_ids.append(c_ids[0])
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


@pytest.mark.route_engaged("cypher-fast")
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


@pytest.mark.route_engaged("cypher-fast")
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


@pytest.mark.route_engaged("cypher-fast")
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


@pytest.mark.route_engaged("cypher-fast")
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


def test_interleaved_ids_build_facts_but_kernel_declines_upstream() -> None:
    """The real-board layout after any shuffle/join. Three-part contract:
    facts BUILD (span-based, gaps are harmless zeros), the kernel does NOT
    consult them (its own dense-DOMAIN proof fails on scattered ids and it
    declines before the fact matters), and the answer is correct via the scan.
    If a future change makes scattered domains provable, the used-assertion
    flips -- deliberately, not silently."""
    base = _graph(4, 3, interleave=True)
    oracle, _ = _run(base)
    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
    assert [k for k in get_registry(g).degrees], "span-based build must not refuse gaps"
    value, used = _run(g)
    assert value == oracle
    assert not used, "dense-domain proof cannot pass on scattered ids"


def test_a_fact_covering_a_narrower_span_is_refused() -> None:
    """NEGATIVE side of the consult's span check: a fact whose arrays do not
    cover the requested domain must be IGNORED, because slicing outside its
    range would miscount -- a wrong answer, not a slow one."""
    import graphistry.compute.gfql_fast_paths as fpm
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.build import build_degree_fact

    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})
    narrow = build_degree_fact(edges, "s", "d", 0, 2, Engine.PANDAS)
    assert narrow is not None
    nodes = pd.DataFrame({"id": [0, 1, 2, 3]})
    wider = fpm._two_hop_equal_domain_dense_total(
        nodes, edges, node_col="id", src_col="s", dst_col="d",
        engine=Engine.PANDAS, domain_interval_hint=(0, 3), degree_fact=narrow)
    honest = fpm._two_hop_equal_domain_dense_total(
        nodes, edges, node_col="id", src_col="s", dst_col="d",
        engine=Engine.PANDAS, domain_interval_hint=(0, 3))
    assert wider == honest, "narrow fact must be refused, not sliced out of range"


Q_FILTERED = ("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})"
              "-[{rel:'F'}]->(c {kind:'P'}) "
              "WHERE b.age < 30 AND c.age > 20 RETURN count(*) AS n")


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_endpoint_filters_decline_dense_but_the_fused_count_serves(engine: str) -> None:
    """The q9 shape: same typed two-hop count as q8 plus WHERE filters on the
    endpoints. The precomputed degree product counts ALL paths through each
    midpoint, so consulting it here would be a wrong answer, not a slow one --
    the dense kernel must not even be called. The query still belongs to the
    two_hop_count fast path (its fused arm applies the filters), so the decline
    must not cascade into a full scan.

    The fixture's filtered count differs from its unfiltered count, so a kernel
    that ignored the filters would also fail the value gate.
    """
    from graphistry.tests.compute.gfql.engagement import assert_fast_path

    nodes = pd.DataFrame({"id": [0, 1, 2, 3],
                          "kind": ["P"] * 4,
                          "age": [25, 35, 25, 45]})
    edges = pd.DataFrame({"s": [0, 1, 2, 0], "d": [1, 2, 3, 2],
                          "rel": ["F"] * 4})
    base = graphistry.nodes(_to(nodes, engine), "id").edges(_to(edges, engine), "s", "d")
    unfiltered, _ = _run(base, engine=engine)
    oracle, _ = _run(base, Q_FILTERED, engine=engine)
    assert 0 < oracle < unfiltered, "fixture cannot distinguish a filter-ignoring kernel"

    g = base.gfql_index_col_stats(node_type_column="kind", edge_type_column="rel",
                                  engine=engine)
    assert [k for k in get_registry(g).degrees], "no degree facts built"

    called = []
    real = fp._two_hop_equal_domain_dense_total

    def spy(*a, **k):
        called.append(True)
        return real(*a, **k)

    fp._two_hop_equal_domain_dense_total = spy
    try:
        value, _ = _run(g, Q_FILTERED, engine=engine)
    finally:
        fp._two_hop_equal_domain_dense_total = real
    assert not called, "dense kernel consulted under endpoint filters -- wrong-answer risk"
    assert value == oracle
    assert_fast_path(g, Q_FILTERED, "two_hop_count", served=True, engine=engine)
