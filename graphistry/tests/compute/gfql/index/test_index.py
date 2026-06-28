"""Tests for GFQL physical indexes (pay-as-you-go seeded-traversal acceleration).

Behavioral / differential: the index fast path must return the SAME subgraph as
the scan/join path. Engine-parametrized (pandas/cudf/polars/polars-gpu) with
importorskip so GPU lanes run only where available.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward
from graphistry.compute.gfql.index import (
    CreateIndex, DropIndex, ShowIndexes, index_op_from_json, parse_index_ddl,
    get_registry,
)


def _engines():
    out = ["pandas"]
    try:
        import cudf  # noqa
        out.append("cudf")
    except Exception:
        pass
    try:
        import polars  # noqa
        out.append("polars")
        try:
            import cudf  # noqa
            out.append("polars-gpu")
        except Exception:
            pass
    except Exception:
        pass
    return out


ENGINES = _engines()


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(0)
    n_nodes, deg = 2000, 6
    m = n_nodes * deg
    edf = pd.DataFrame({"src": rng.integers(0, n_nodes, m), "dst": rng.integers(0, n_nodes, m)})
    ndf = pd.DataFrame({"id": np.arange(n_nodes), "lab": rng.integers(0, 4, n_nodes)})
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst")


def _sig(g):
    def topd(df):
        mod = type(df).__module__
        return df.to_pandas() if ("cudf" in mod or "polars" in mod) else df
    nn = topd(g._nodes); ee = topd(g._edges)
    nodes = sorted(nn["id"].tolist())
    edges = sorted(map(tuple, ee[["src", "dst"]].itertuples(index=False, name=None)))
    return nodes, edges


SCENARIOS = [
    dict(hops=1, direction="forward"),
    dict(hops=1, direction="reverse"),
    dict(hops=1, direction="undirected"),
    dict(hops=2, direction="forward"),
    dict(hops=2, direction="undirected"),
    dict(hops=1, direction="forward", return_as_wave_front=True),
    dict(hops=2, direction="forward", return_as_wave_front=True),
    dict(hops=3, direction="forward"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sc", SCENARIOS)
def test_index_parity_vs_scan(graph, engine, sc):
    seeds = pd.DataFrame({"id": [0, 1, 2, 7, 42, 100]})
    gi = graph.gfql_index_all(engine=engine)
    base = graph.hop(nodes=seeds, engine=engine, **sc)
    idx = gi.hop(nodes=seeds, engine=engine, **sc)
    assert _sig(base) == _sig(idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_show_drop(graph, engine):
    g = graph.create_index("edge_out_adj", engine=engine)
    g = g.create_index("edge_in_adj", engine=engine)
    si = g.show_indexes()
    assert set(si["kind"]) == {"edge_out_adj", "edge_in_adj"}
    g2 = g.drop_index("edge_in_adj")
    assert set(g2.show_indexes()["kind"]) == {"edge_out_adj"}
    g3 = g2.drop_index()  # drop all
    assert g3.show_indexes().shape[0] == 0


def test_cypher_ddl_recognizer():
    assert isinstance(parse_index_ddl("CREATE GFQL INDEX FOR edge_out_adj"), CreateIndex)
    assert parse_index_ddl("CREATE GFQL INDEX pk FOR node_id ON id").column == "id"
    assert isinstance(parse_index_ddl("DROP GFQL INDEX FOR edge_in_adj"), DropIndex)
    assert isinstance(parse_index_ddl("SHOW GFQL INDEXES"), ShowIndexes)
    assert parse_index_ddl("MATCH (a) RETURN a") is None
    with pytest.raises(ValueError):
        parse_index_ddl("CREATE GFQL INDEX FOR bogus_kind")


def test_cypher_ddl_via_gfql(graph):
    g = graph.gfql("CREATE GFQL INDEX FOR edge_out_adj")
    assert get_registry(g).has("edge_out_adj")
    si = g.gfql("SHOW GFQL INDEXES")
    assert si.shape[0] == 1
    g2 = g.gfql("DROP GFQL INDEX FOR edge_out_adj")
    assert g2.show_indexes().shape[0] == 0


def test_wire_roundtrip(graph):
    op = CreateIndex(kind="edge_out_adj")
    assert index_op_from_json(op.to_json()) == op
    g = graph.gfql({"type": "CreateIndex", "kind": "edge_out_adj"})
    assert get_registry(g).has("edge_out_adj")
    show = g.gfql({"type": "ShowIndexes"})
    assert show.shape[0] == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_index_policy_force_and_explain(graph, engine):
    chain = [n({"id": 0}), e_forward(hops=1)]
    rep_off = graph.gfql_explain(chain, index_policy="off", engine=engine)
    assert rep_off["used_index"] is False
    rep_force = graph.gfql_explain(chain, index_policy="force", engine=engine)
    assert rep_force["used_index"] is True
    # results identical regardless of policy
    r_scan = graph.gfql(chain, engine=engine)
    r_force = graph.gfql(chain, index_policy="force", engine=engine)
    assert _sig(r_scan) == _sig(r_force)


def test_column_mismatch_raises_not_silent(graph):
    # A custom column that doesn't match the binding must raise, not silently no-op.
    with pytest.raises(NotImplementedError):
        graph.create_index("edge_out_adj", column="not_src")
    # Matching column (the actual binding) is accepted.
    g = graph.create_index("edge_out_adj", column="src")
    assert get_registry(g).has("edge_out_adj")
    # Custom display name is honored.
    g2 = graph.create_index("node_id", name="pk")
    assert "pk" in g2.show_indexes()["name"].tolist()


def test_show_indexes_valid_and_nbytes(graph):
    g = graph.create_index("edge_out_adj")
    si = g.show_indexes()
    assert {"valid", "nbytes"}.issubset(si.columns)
    assert bool(si.iloc[0]["valid"]) is True
    assert int(si.iloc[0]["nbytes"]) > 0
    # After an edges rebind, the index is stale -> valid=False (and auto-skipped).
    rng = np.random.default_rng(2)
    new_edf = pd.DataFrame({"src": rng.integers(0, 2000, 50), "dst": rng.integers(0, 2000, 50)})
    g2 = g.edges(new_edf, "src", "dst")
    assert bool(g2.show_indexes().iloc[0]["valid"]) is False


def test_fingerprint_invalidation_is_safe(graph):
    # An index built over one edge frame must not be used after .edges() rebind.
    gi = graph.create_index("edge_out_adj")
    rng = np.random.default_rng(1)
    new_edf = pd.DataFrame({"src": rng.integers(0, 2000, 100), "dst": rng.integers(0, 2000, 100)})
    g2 = gi.edges(new_edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0, 1, 2]})
    # registry attr carries over but fingerprint (frame id) no longer matches ->
    # treated as absent -> correct result computed by scan path.
    base = graph.edges(new_edf, "src", "dst").hop(nodes=seeds, hops=1)
    got = g2.hop(nodes=seeds, hops=1)
    assert _sig(base) == _sig(got)


# ---------------------------------------------------------------------------
# Adversarial-review regression tests (takeover 2026-06-28): each reproduces a
# CONFIRMED wrong-answer the original parity scenarios missed. The index MUST
# equal the scan oracle, or fall back to scan — never a wrong answer.
# ---------------------------------------------------------------------------

def _force(g, engine):
    """Index-all + force the index path past the cost gate (so we actually test it)."""
    gi = g.gfql_index_all(engine=engine)
    gi._gfql_index_policy = "force"
    return gi


def _nodeset(g):
    df = g._nodes
    mod = type(df).__module__
    df = df.to_pandas() if ("cudf" in mod or "polars" in mod) else df
    return df


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hop_kw", [
    dict(hops=1, max_hops=3),   # B1: max_hops must win over hops (was silently ignored)
    dict(hops=5, max_hops=2),   # B1: hops ignored when max_hops set
])
def test_index_max_hops_honored(engine, hop_kw):
    edf = pd.DataFrame({"src": [0, 1, 2, 3, 4], "dst": [1, 2, 3, 4, 5]})
    g = graphistry.edges(edf, "src", "dst").materialize_nodes()
    seeds = pd.DataFrame({"id": [0]})
    base = g.hop(nodes=seeds, engine=engine, **hop_kw)
    idx = _force(g, engine).hop(nodes=seeds, engine=engine, **hop_kw)
    assert _sig(base) == _sig(idx), f"max_hops divergence {hop_kw}"


@pytest.mark.parametrize("engine", ENGINES)
def test_index_duplicate_node_ids(engine):
    # B2: duplicate node ids corrupted the unique-key node_id index. gfql_index_all
    # must skip node_id (build adjacency only) and stay scan-equal.
    ndf = pd.DataFrame({"id": [0, 1, 2, 3, 5, 5]})  # id 5 duplicated
    edf = pd.DataFrame({"src": [0, 1, 3], "dst": [1, 2, 5]})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0, 1, 3]})
    base = g.hop(nodes=seeds, engine=engine, hops=1)
    idx = _force(g, engine).hop(nodes=seeds, engine=engine, hops=1)
    assert _sig(base) == _sig(idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_index_missing_endpoint(engine):
    # B3: an edge endpoint absent from the node table — the scan synthesizes it as a
    # node row; the index dropped it. Must match (index falls back if it can't).
    ndf = pd.DataFrame({"id": [0, 1, 2, 3]})  # 99 absent
    edf = pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 99]})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0, 1, 2]})
    base = g.hop(nodes=seeds, engine=engine, hops=1)
    idx = _force(g, engine).hop(nodes=seeds, engine=engine, hops=1)
    assert _sig(base) == _sig(idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_index_preserves_node_table_order(engine):
    # I4: a node_id-index materialization must keep the original .nodes row order
    # (not searchsorted/sorted-by-id order).
    ndf = pd.DataFrame({"id": [5, 3, 1, 0, 2, 4]})  # NOT sorted
    edf = pd.DataFrame({"src": [5, 3, 1], "dst": [3, 1, 0]})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [5, 3, 1]})
    base = g.hop(nodes=seeds, engine=engine, hops=1)
    idx = _force(g, engine).hop(nodes=seeds, engine=engine, hops=1)
    assert _nodeset(base)["id"].tolist() == _nodeset(idx)["id"].tolist(), "node order diverged"


@pytest.mark.parametrize("engine", ENGINES)
def test_index_mixed_id_dtypes(engine):
    # I6: int64 node-id seed vs int32 edge endpoints — narrowing the seed to the key
    # dtype could wrap/false-match; promotion keeps it correct.
    ndf = pd.DataFrame({"id": np.array([0, 1, 2, 3], dtype=np.int64)})
    edf = pd.DataFrame({"src": np.array([0, 1, 2], dtype=np.int32),
                        "dst": np.array([1, 2, 3], dtype=np.int32)})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0, 1, 2]})
    base = g.hop(nodes=seeds, engine=engine, hops=1)
    idx = _force(g, engine).hop(nodes=seeds, engine=engine, hops=1)
    assert _sig(base) == _sig(idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_index_stale_rebind_same_shape(engine):
    # I5: rebinding .edges() to a NEW same-shape frame must invalidate the index by
    # object identity (not id(), which can be recycled) -> scan fallback, never stale.
    g = graphistry.edges(pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]}), "src", "dst").materialize_nodes()
    gi = g.gfql_index_all(engine=engine)
    new_edf = pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 0]})  # same shape, different values
    g2 = gi.edges(new_edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0]})
    base = g.edges(new_edf, "src", "dst").hop(nodes=seeds, engine=engine, hops=1)
    got = g2.hop(nodes=seeds, engine=engine, hops=1)
    assert _sig(base) == _sig(got)


def test_drop_index_by_custom_name():
    # B2 (review): DROP GFQL INDEX <custom-name> must resolve via the index name, not
    # silently no-op (the old code split the name on ':' and only matched default names).
    g = graphistry.edges(pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]}), "src", "dst").materialize_nodes()
    g = g.create_index("edge_out_adj", name="myidx")
    assert "edge_out_adj" in g.show_indexes()["kind"].tolist()
    g2 = g.gfql("DROP GFQL INDEX myidx")
    assert g2.show_indexes().shape[0] == 0, "custom-named index was not dropped"
    # unresolvable name raises (not silent)
    with pytest.raises(ValueError):
        g.gfql("DROP GFQL INDEX nonexistent_name")
