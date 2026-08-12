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


def _graph(n_p=3, n_c=3, engine="pandas"):
    nodes = pd.DataFrame({"id": list(range(n_p + n_c)),
                          "kind": ["P"] * n_p + ["C"] * n_c})
    edges = pd.DataFrame({"s": [0, 1, 2, 0, 0], "d": [1, 2, 0, n_p, n_p + 1],
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


def test_gapped_node_space_builds_no_degree_facts() -> None:
    """Degrees are indexed by ``id - lo``, so a gapped domain has no valid
    indexing. Build nothing rather than something the kernel could misread."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 10, 11], "kind": ["P"] * 3 + ["C"] * 2})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2], "rel": ["F", "F"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql_index_col_stats(
        node_type_column="kind", edge_type_column="rel")
    assert [k for k in get_registry(g).degrees] == []


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
