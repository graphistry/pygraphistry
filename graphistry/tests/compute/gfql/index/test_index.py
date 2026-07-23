"""Tests for GFQL physical indexes (pay-as-you-go seeded-traversal acceleration).

Behavioral / differential: the index fast path must return the SAME subgraph as
the scan/join path. Engine-parametrized (pandas/cudf/polars/polars-gpu) with
importorskip so GPU lanes run only where available.
"""
import importlib

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
    nn = topd(g._nodes)
    ee = topd(g._edges)
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


def test_cypher_ddl_regexes_are_lazy(monkeypatch):
    import graphistry.compute.gfql.index.cypher_ddl as ddl

    real_compile = ddl.re.compile
    calls = []

    def tracking_compile(*args, **kwargs):
        calls.append(args[0])
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(ddl.re, "compile", tracking_compile)
    ddl = importlib.reload(ddl)

    assert calls == []
    assert ddl.looks_like_index_ddl("CREATE GFQL INDEX FOR edge_out_adj")
    assert len(calls) == 1
    assert ddl.parse_index_ddl("SHOW GFQL INDEXES") == ShowIndexes()
    assert len(calls) == 5
    assert isinstance(ddl.parse_index_ddl("DROP GFQL INDEX FOR edge_in_adj"), DropIndex)
    assert len(calls) == 5


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

def test_create_rebuilds_stale_resident_index():
    g = graphistry.edges(pd.DataFrame({"src": [0, 1], "dst": [1, 2]}), "src", "dst").materialize_nodes()
    gi = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")
    assert bool(gi.show_indexes().iloc[0]["valid"]) is True

    g2 = gi.edges(pd.DataFrame({"src": [2, 3, 4], "dst": [3, 4, 5]}), "src", "dst")
    assert bool(g2.show_indexes().iloc[0]["valid"]) is False

    g3 = g2.gfql("CREATE GFQL INDEX FOR edge_out_adj")
    row = g3.show_indexes().iloc[0]
    assert bool(row["valid"]) is True
    assert int(row["n_rows"]) == 3

    g4 = g2.gfql({"type": "CreateIndex", "kind": "edge_out_adj"})
    row = g4.show_indexes().iloc[0]
    assert bool(row["valid"]) is True
    assert int(row["n_rows"]) == 3


def test_invalid_index_policy_raises(graph):
    chain = [n({"id": 0}), e_forward(hops=1)]
    with pytest.raises(ValueError, match="index_policy"):
        graph.gfql(chain, index_policy="bogus")
    with pytest.raises(ValueError, match="index_policy"):
        graph.gfql(chain, index_policy="")
    with pytest.raises(ValueError, match="index_policy"):
        graph.gfql_explain(chain, index_policy="bogus")

    # Documented values remain accepted.
    graph.gfql(chain, index_policy="off")
    graph.gfql_explain(chain, index_policy="use")


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


@pytest.mark.parametrize("engine", ENGINES)
def test_explain_exposes_planner_diagnostics(graph, engine):
    """LP1: gfql_explain surfaces the planner's cost signal — seed cardinality, the
    free Σ-degree fanout estimate (from CSR group_offsets), chosen direction, the
    cost-gate threshold, and a human-readable decision_reason — not just a used-index
    yes/no. This is the EXPLAIN enrichment the indexing roadmap calls for."""
    chain = [n({"id": 0}), e_forward(hops=1)]
    # force → index path taken → per-step + top-level diagnostics populated
    rep = graph.gfql_explain(chain, index_policy="force", engine=engine)
    assert rep["used_index"] is True, (engine, rep)
    assert rep["chosen_direction"] == "forward"
    assert isinstance(rep["est_result_rows"], int) and rep["est_result_rows"] >= 0
    assert "index" in (rep["decision_reason"] or ""), rep["decision_reason"]
    step = next(s for s in rep["steps"] if s.get("path") == "index")
    for k in ("frontier_n", "n_keys", "seed_deg_sum", "est_result_rows", "threshold_frac"):
        assert k in step, (k, step)
    assert step["est_result_rows"] == step["seed_deg_sum"]  # est == free Σ-degree
    assert step["seed_deg_sum"] >= 0

    # off (index resident) → the planner records *why* it scanned, not just that it did
    gi = graph.gfql_index_all(engine=engine)
    rep_off = gi.gfql_explain(chain, index_policy="off", engine=engine)
    assert rep_off["used_index"] is False
    assert rep_off["decision_reason"] == "policy=off", rep_off


def test_seed_diagnostic_helpers_are_robust():
    """LP1 helpers degrade to None instead of crashing on odd inputs (they run under
    the explain trace and must never take down a real query)."""
    from graphistry.compute.gfql.index.api import _seed_id_array, _seed_deg_sum

    class _Col:  # a seed column with .values but no .to_numpy() (fallback branch)
        values = np.array([1, 2, 3])

    class _Frame:
        def __getitem__(self, k):
            return _Col()

    assert list(_seed_id_array(_Frame(), "id")) == [1, 2, 3]
    assert _seed_id_array(None, "id") is None  # nodes[None] raises → None, not a crash

    class _BadIdx:  # missing keys_sorted/group_offsets → None, not AttributeError
        pass

    assert _seed_deg_sum(_BadIdx(), np.array([0, 1])) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_explain_decision_reasons_for_scan_fallbacks(engine):
    """LP1: when the planner declines the index it records *why*, so a silent scan is
    diagnosable. Covers the two fall-back branches: (a) a frontier past the cost gate,
    (b) a query the index doesn't cover (min_hops>1)."""
    from graphistry.compute.gfql.index import index_trace
    rng = np.random.default_rng(2)
    N, deg = 1000, 6
    edf = pd.DataFrame({"src": rng.integers(0, N, N * deg), "dst": rng.integers(0, N, N * deg)})
    ndf = pd.DataFrame({"id": np.arange(N)})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    gi = g.gfql_index_all(engine=engine)

    # (a) cost-gate fallback: frontier = all keys >> frac*n_keys → scan, with a reason
    allseeds = pd.DataFrame({"id": np.arange(N, dtype=np.int64)})
    with index_trace() as steps:
        gi.hop(nodes=allseeds, engine=engine, hops=1, direction="forward")
    assert any("scan cheaper" in (s.get("decision_reason") or "") for s in steps), (engine, steps)
    assert not any(s.get("path") == "index" for s in steps), (engine, steps)

    # (b) not-coverable fallback: a feature outside the index fast path (zero-hop seed).
    # pandas-only: the Phase-1 polars hop rejects these features at its own engine layer
    # before the index planner is consulted, so the index "not-coverable" bail is
    # reachable via the pandas hop path here.
    if engine == "pandas":
        few = pd.DataFrame({"id": np.arange(4, dtype=np.int64)})
        with index_trace() as steps2:
            gi.hop(nodes=few, engine=engine, hops=1, direction="forward", include_zero_hop_seed=True)
        assert any(s.get("decision_reason") == "query not index-coverable" for s in steps2), (engine, steps2)


@pytest.mark.parametrize("engine", ENGINES)
def test_cost_gate_engine_aware_never_loses_to_scan(engine):
    """F1: the index-vs-scan crossover depends on scan speed, so the cost gate
    (maybe_index_hop) is engine-aware. A mid-frontier seeded hop (~frac 0.25 of source
    keys) still beats the slow pandas scan (→ index path) but would lose to the fast
    vectorized scan on polars/cudf/GPU (→ scan fallback), so a resident index never
    runs slower than the un-indexed path. Result is identical on either path."""
    from graphistry.compute.gfql.index import index_trace
    rng = np.random.default_rng(1)
    N, deg = 4000, 8
    m = N * deg
    edf = pd.DataFrame({"src": rng.integers(0, N, m), "dst": rng.integers(0, N, m)})
    ndf = pd.DataFrame({"id": np.arange(N)})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": np.arange(N // 4, dtype=np.int64)})  # frac ~0.25 of n_keys (~N)
    gi = g.gfql_index_all(engine=engine)
    with index_trace() as steps:
        out = gi.hop(nodes=seeds, engine=engine, hops=1, direction="forward")
    took_index = any(s.get("path") == "index" for s in steps)
    assert took_index is (engine == "pandas"), (engine, steps)  # vectorized engines → scan
    scan_out = g.hop(nodes=seeds, engine=engine, hops=1, direction="forward")  # no resident index
    assert _sig(out) == _sig(scan_out), engine  # correctness path-independent


def test_cost_gate_frac_tuning(monkeypatch):
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index import (
        cost_gate_frac, reset_cost_gate_frac, set_cost_gate_frac,
    )

    reset_cost_gate_frac()
    monkeypatch.delenv("GFQL_INDEX_COST_GATE_FRAC", raising=False)
    monkeypatch.delenv("GFQL_INDEX_COST_GATE_FRAC_PANDAS", raising=False)
    assert cost_gate_frac(Engine.PANDAS) == 0.5

    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC", "0.11")
    assert cost_gate_frac(Engine.PANDAS) == 0.11

    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_PANDAS", "0.22")
    assert cost_gate_frac(Engine.PANDAS) == 0.22

    set_cost_gate_frac(Engine.PANDAS, 0.33)
    assert cost_gate_frac(Engine.PANDAS) == 0.33

    set_cost_gate_frac(Engine.PANDAS, None)
    assert cost_gate_frac(Engine.PANDAS) == 0.22

    with pytest.raises(ValueError, match="cost gate fraction"):
        set_cost_gate_frac(Engine.PANDAS, 0)
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_PANDAS", "2")
    with pytest.raises(ValueError, match="GFQL_INDEX_COST_GATE_FRAC_PANDAS"):
        cost_gate_frac(Engine.PANDAS)
    reset_cost_gate_frac()


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


def test_drop_index_if_exists_semantics():
    # missing_ok / IF EXISTS: plain DROP of a missing index raises (SQL-style);
    # IF EXISTS makes it a no-op. Applies to both name- and kind-form.
    g = graphistry.edges(pd.DataFrame({"src": [0, 1], "dst": [1, 2]}), "src", "dst").materialize_nodes()

    # no-op forms succeed unchanged
    g2 = g.gfql("DROP GFQL INDEX IF EXISTS nonexistent_name")
    assert g2.show_indexes().shape[0] == 0
    g3 = g.gfql("DROP GFQL INDEX IF EXISTS FOR edge_out_adj")
    assert g3.show_indexes().shape[0] == 0

    # plain forms raise when missing
    with pytest.raises(ValueError):
        g.gfql("DROP GFQL INDEX nonexistent_name")
    with pytest.raises(ValueError):
        g.gfql("DROP GFQL INDEX FOR edge_out_adj")

    # wire JSON honors missing_ok both ways
    g4 = g.gfql({"type": "DropIndex", "name": "nope", "missing_ok": True})
    assert g4.show_indexes().shape[0] == 0
    with pytest.raises(ValueError):
        g.gfql({"type": "DropIndex", "name": "nope", "missing_ok": False})

    # and a resident index still drops through the plain form
    gi = g.create_index("edge_out_adj")
    assert gi.gfql("DROP GFQL INDEX FOR edge_out_adj").show_indexes().shape[0] == 0


# --- get_degrees index fast path (#5 degree-cache / #3 membership) ---

def _to_engine_frames(g, engine):
    if engine == "cudf":
        import cudf
        return g.nodes(cudf.from_pandas(g._nodes), g._node).edges(cudf.from_pandas(g._edges), g._source, g._destination)
    if engine in ("polars", "polars-gpu"):
        import polars as pl
        return g.nodes(pl.from_pandas(g._nodes), g._node).edges(pl.from_pandas(g._edges), g._source, g._destination)
    return g


def _degrees(g, engine):
    if engine in ("polars", "polars-gpu"):
        from graphistry.compute.gfql.lazy.engine.polars.degrees import get_degrees_polars
        return get_degrees_polars(g, engine=engine)
    return g.get_degrees()


def _degcols(g_res):
    nn = g_res._nodes
    nn = nn.to_pandas() if hasattr(nn, "to_pandas") else nn
    nn = nn.sort_values("id").reset_index(drop=True)
    return {c: nn[c].tolist() for c in ("degree", "degree_in", "degree_out")}


@pytest.mark.parametrize("engine", ENGINES)
def test_get_degrees_index_parity(graph, engine):
    g = _to_engine_frames(graph, engine)
    base = _degcols(_degrees(g, engine))                       # group_by scan path
    gi = g.gfql_index_all(engine=engine)
    idx = _degcols(_degrees(gi, engine))                       # index fast path
    assert base == idx


@pytest.mark.parametrize("engine", ENGINES)
def test_get_degrees_self_loops_and_isolated(graph, engine):
    # graph with self-loops + isolated nodes (0..N-1; some have no edge)
    import numpy as _np
    ndf = pd.DataFrame({"id": _np.arange(300)})
    edf = pd.DataFrame({"src": [0, 1, 2, 5, 5], "dst": [0, 2, 1, 5, 6]})  # self-loops at 0,5
    g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    g = _to_engine_frames(g0, engine)
    base = _degcols(_degrees(g, engine))
    gi = g.gfql_index_all(engine=engine)
    idx = _degcols(_degrees(gi, engine))
    assert base == idx
    # self-loop double-count contract: node 0 (self-loop only) degree==2
    assert idx["degree"][0] == 2


@pytest.mark.parametrize("engine", ENGINES)
def test_get_degrees_index_policy_off(graph, engine):
    g = _to_engine_frames(graph, engine)
    gi = g.gfql_index_all(engine=engine)
    setattr(gi, "_gfql_index_policy", "off")
    assert _degcols(_degrees(gi, engine)) == _degcols(_degrees(g, engine))


# --- EXISTS { (n)--() } prune-isolated via #3 membership (polars fast path) ---

@pytest.mark.parametrize("engine", [e for e in ENGINES if e in ("polars", "polars-gpu")])
def test_exists_prune_membership_parity(engine):
    import polars as pl
    rng = np.random.default_rng(4)
    N, E = 5000, 20000
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    dst[:50] = src[:50]  # self-loops (membership must still include them)
    g = graphistry.nodes(pl.DataFrame({"id": np.arange(N)}), "id").edges(
        pl.DataFrame({"src": src, "dst": dst}), "src", "dst")
    q = "MATCH (n) WHERE EXISTS { (n)--() } RETURN n.id AS id"
    def ids(gg):
        nn = gg.gfql(q, engine=engine)._nodes
        nn = nn.to_pandas() if hasattr(nn, "to_pandas") else nn
        return sorted(nn["id"].tolist())
    base = ids(g)
    gi = g.gfql_index_all(engine=engine)
    assert ids(gi) == base
    # policy='off' still correct
    gi2 = g.gfql_index_all(engine=engine)
    setattr(gi2, "_gfql_index_policy", "off")
    assert ids(gi2) == base


def test_get_degrees_polars_gpu_uses_matching_index_engine(monkeypatch):
    pl = pytest.importorskip("polars")
    from graphistry.Engine import Engine
    import graphistry.compute.gfql.index.degrees as index_degrees
    from graphistry.compute.gfql.lazy.engine.polars.degrees import get_degrees_polars

    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2]}), "id").edges(
        pl.DataFrame({"src": [0, 1], "dst": [1, 2]}), "src", "dst")
    gi = g.gfql_index_all(engine="polars-gpu")

    seen = []
    orig = index_degrees.degrees_from_index

    def wrapped(registry, nodes_df, node_col, edges_df, cols, engine):
        seen.append(engine)
        return orig(registry, nodes_df, node_col, edges_df, cols, engine)

    monkeypatch.setattr(index_degrees, "degrees_from_index", wrapped)
    out = get_degrees_polars(gi, engine="polars-gpu")

    assert Engine.POLARS_GPU in seen
    assert "degree" in out._nodes.columns


def test_exists_prune_membership_polars_gpu_uses_matching_index_engine(monkeypatch):
    pl = pytest.importorskip("polars")
    from graphistry.Engine import Engine
    from graphistry.compute.ast import n, e_undirected, serialize_binding_ops
    import graphistry.compute.gfql.index.degrees as index_degrees
    from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode
    from graphistry.compute.gfql.lazy.engine.polars.pattern_apply import _pattern_alias_keys_polars

    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2]}), "id").edges(
        pl.DataFrame({"src": [0, 1], "dst": [1, 2]}), "src", "dst")
    gi = g.gfql_index_all(engine="polars-gpu")
    binding_ops = serialize_binding_ops([n(name="n"), e_undirected(), n(name="m")])

    seen = []
    orig = index_degrees.adjacency_membership_keys

    def wrapped(registry, direction, edges_df, cols, engine):
        seen.append(engine)
        return orig(registry, direction, edges_df, cols, engine)

    monkeypatch.setattr(index_degrees, "adjacency_membership_keys", wrapped)
    with target_mode(ExecutionTarget.GPU):
        keys = _pattern_alias_keys_polars(gi, binding_ops, "n")

    assert Engine.POLARS_GPU in seen
    assert keys is not None
    assert sorted(keys.get_column("id").to_list()) == [0, 1, 2]


# ---- chain / Cypher index engagement (#1658 through gfql(), incl. typed edges) -------
# Regression: a seeded hop expressed as a gfql()/Cypher CHAIN (not a direct g.hop())
# used to ALWAYS scan on pandas/cuDF, because _chain_impl attaches a synthetic per-edge
# id column -> a fresh edge frame -> the index's `source_ref is df` identity guard missed.
# rebind_edges (chain.py) re-points the adjacency index at the augmented frame; typed
# edges additionally route a simple scalar-equality edge_match through the index.

@pytest.fixture(scope="module")
def typed_graph():
    rng = np.random.default_rng(1)
    n_nodes, deg = 2000, 6
    m = n_nodes * deg
    edf = pd.DataFrame({
        "src": rng.integers(0, n_nodes, m),
        "dst": rng.integers(0, n_nodes, m),
        "etype": rng.integers(0, 3, m),
    })
    ndf = pd.DataFrame({"id": np.arange(n_nodes)})
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst")


def _sig_typed(g):
    def topd(df):
        mod = type(df).__module__
        return df.to_pandas() if ("cudf" in mod or "polars" in mod) else df
    nn = topd(g._nodes)
    ee = topd(g._edges)
    nodes = sorted(nn["id"].tolist())
    edges = sorted(map(tuple, ee[["src", "dst", "etype"]].itertuples(index=False, name=None)))
    return nodes, edges


_TYPED_CHAIN = [n({"id": 100}), e_forward({"etype": 1}, hops=1)]
_TYPED_2HOP = [n({"id": 100}), e_forward({"etype": 1}, hops=1), n(),
               e_forward({"etype": 2}, hops=1)]
_UNTYPED_CHAIN = [n({"id": 100}), e_forward(hops=1)]
_MEMBER_CHAIN = [n({"id": 100}), e_forward({"etype": [0, 1]}, hops=1)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chain", [_TYPED_CHAIN, _TYPED_2HOP, _UNTYPED_CHAIN, _MEMBER_CHAIN])
def test_chain_index_parity_vs_scan(typed_graph, engine, chain):
    """Every chain shape returns the SAME subgraph via the index as via the scan,
    on every engine. Correctness is never traded for the fast path."""
    gi = typed_graph.gfql_index_all(engine=engine)
    base = typed_graph.gfql(chain, index_policy="off", engine=engine)
    idx = gi.gfql(chain, index_policy="use", engine=engine)
    assert _sig_typed(base) == _sig_typed(idx)


@pytest.mark.parametrize("engine", ENGINES)
def test_chain_typed_edge_engages_index(typed_graph, engine):
    """All four engines: a typed-edge (simple-equality edge_match) seeded chain hop
    ENGAGES the resident #1658 index instead of scanning (pandas/cuDF via the eager
    chain rebind; polars/polars-gpu via the native lazy chain executor rebind)."""
    gi = typed_graph.gfql_index_all(engine=engine)
    rep = gi.gfql_explain(_TYPED_CHAIN, index_policy="use", engine=engine)
    assert rep["used_index"] is True, (engine, rep)


@pytest.mark.parametrize("engine", ENGINES)
def test_chain_untyped_engages_index(typed_graph, engine):
    """An untyped seeded chain hop engages the index on every engine (pandas/cuDF via
    the rebind fix; polars/polars-gpu via their native lazy executor)."""
    gi = typed_graph.gfql_index_all(engine=engine)
    rep = gi.gfql_explain(_UNTYPED_CHAIN, index_policy="use", engine=engine)
    assert rep["used_index"] is True, (engine, rep)


@pytest.mark.parametrize("engine", ENGINES)
def test_chain_membership_edge_match_stays_on_scan(typed_graph, engine):
    """A membership-list edge_match is NOT simple-equality, so it deliberately stays on
    the scan path (parity preserved; no over-reach of the index coverage)."""
    gi = typed_graph.gfql_index_all(engine=engine)
    rep = gi.gfql_explain(_MEMBER_CHAIN, index_policy="use", engine=engine)
    assert rep["used_index"] is False, (engine, rep)


def test_rebind_edges_revalidates_after_shallow_augmentation():
    """rebind_edges re-points the edge adjacency index at a shallow-copied frame that
    merely ADDS a column (the chain's synthetic edge id), so get_valid recognizes it."""
    from graphistry.compute.gfql.index import get_registry
    from graphistry.compute.gfql.index.registry import EDGE_OUT_ADJ
    rng = np.random.default_rng(2)
    edf = pd.DataFrame({"src": rng.integers(0, 500, 3000), "dst": rng.integers(0, 500, 3000)})
    ndf = pd.DataFrame({"id": np.arange(500)})
    gi = graphistry.nodes(ndf, "id").edges(edf, "src", "dst").gfql_index_all(engine="pandas")
    reg = get_registry(gi)
    from graphistry.Engine import Engine as _E
    # augment like the chain does: shallow copy + an extra column -> a NEW frame object
    aug = gi._edges.copy(deep=False)
    aug["__synthetic_id__"] = aug.index
    assert reg.get_valid(EDGE_OUT_ADJ, aug, ("src", "dst"), _E.PANDAS) is None  # identity miss
    reg2 = reg.rebind_edges(aug)
    assert reg2.get_valid(EDGE_OUT_ADJ, aug, ("src", "dst"), _E.PANDAS) is not None  # now valid


def test_rebind_edges_drops_index_on_fingerprint_mismatch():
    """rebind_edges ENFORCES its contract structurally: a frame with a different row
    count (or missing indexed columns) cannot inherit the index — it is dropped
    (safe miss -> scan) instead of re-pointed at a frame it wasn't built over."""
    from graphistry.compute.gfql.index import get_registry
    from graphistry.compute.gfql.index.registry import EDGE_IN_ADJ, EDGE_OUT_ADJ
    rng = np.random.default_rng(3)
    edf = pd.DataFrame({"src": rng.integers(0, 100, 500), "dst": rng.integers(0, 100, 500)})
    ndf = pd.DataFrame({"id": np.arange(100)})
    gi = graphistry.nodes(ndf, "id").edges(edf, "src", "dst").gfql_index_all(engine="pandas")
    reg = get_registry(gi)
    assert reg.has(EDGE_OUT_ADJ) and reg.has(EDGE_IN_ADJ)

    # row count changed -> both edge indexes dropped, never mis-bound
    fewer = gi._edges.iloc[:-1].copy(deep=False)
    reg2 = reg.rebind_edges(fewer)
    assert not reg2.has(EDGE_OUT_ADJ) and not reg2.has(EDGE_IN_ADJ)

    # indexed column renamed away -> dropped
    renamed = gi._edges.rename(columns={"dst": "dst2"})
    reg3 = reg.rebind_edges(renamed)
    assert not reg3.has(EDGE_OUT_ADJ) and not reg3.has(EDGE_IN_ADJ)

    # same-shape shallow augmentation still rebinds (the intended use)
    aug = gi._edges.copy(deep=False)
    aug["__synthetic_id__"] = aug.index
    reg4 = reg.rebind_edges(aug)
    assert reg4.has(EDGE_OUT_ADJ) and reg4.has(EDGE_IN_ADJ)


# --- review F1/F2/F3 regressions: the guard's value/dtype domain must equal the scan's ----

def _cpu_engines():
    return [e for e in ENGINES if e in ("pandas", "polars")]


@pytest.mark.parametrize("engine", _cpu_engines())
def test_hop_frozenset_edge_match_parity_with_scan(typed_graph, engine):
    """frozenset is membership (isin) on the scan path; the index path must not treat
    it as scalar equality (a bare == is silently all-False). Parity, not zero rows."""
    from graphistry.Engine import Engine as _E, df_to_engine
    g = typed_graph
    if engine == "polars":
        g = g.edges(df_to_engine(g._edges, _E.POLARS), "src", "dst").nodes(
            df_to_engine(g._nodes, _E.POLARS), "id")
    gi = g.gfql_index_all(engine=engine)
    seeds = g._nodes[:5] if engine == "pandas" else g._nodes.head(5)
    kwargs = dict(hops=1, return_as_wave_front=True,
                  edge_match={"etype": frozenset({0, 1})}, engine=engine)
    base = g.hop(nodes=seeds, **kwargs)
    idx = gi.hop(nodes=seeds, **kwargs)

    def n_edges(h):
        return int(h._edges.shape[0])
    assert n_edges(base) > 0  # membership filter genuinely selects rows
    assert n_edges(idx) == n_edges(base)


@pytest.mark.parametrize("engine", _cpu_engines())
def test_hop_null_carrying_match_column_no_crash_and_parity(engine):
    """Null-carrying match columns (pandas nullable Int64 / polars nulls — common after
    NaN->null coercion) must not blow up mask indexing; null == val drops like scan."""
    from graphistry.Engine import Engine as _E, df_to_engine
    edf = pd.DataFrame({
        "src": [0, 0, 1, 1, 2, 2],
        "dst": [1, 2, 2, 3, 3, 0],
        "etype": pd.array([0, None, 0, 1, None, 0], dtype="Int64"),
    })
    ndf = pd.DataFrame({"id": np.arange(4)})
    g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    if engine == "polars":
        g = g.edges(df_to_engine(g._edges, _E.POLARS), "src", "dst").nodes(
            df_to_engine(g._nodes, _E.POLARS), "id")
    gi = g.gfql_index_all(engine=engine)
    seeds = g._nodes[:4] if engine == "pandas" else g._nodes.head(4)
    kwargs = dict(hops=1, return_as_wave_front=True, edge_match={"etype": 0}, engine=engine)
    base = g.hop(nodes=seeds, **kwargs)
    idx = gi.hop(nodes=seeds, **kwargs)  # was: IndexError from object-dtype mask
    assert int(idx._edges.shape[0]) == int(base._edges.shape[0]) == 3


@pytest.mark.parametrize("engine", _cpu_engines())
def test_hop_dtype_mismatch_edge_match_matches_scan_error(typed_graph, engine):
    """Numeric column vs string value: the scan raises (GFQLSchemaError E302 on pandas;
    polars raises its own ComputeError). The index path must decline (fall back to
    scan) so users get the SAME error as their engine's scan — never a silent empty
    subgraph."""
    from graphistry.Engine import Engine as _E, df_to_engine
    g = typed_graph
    if engine == "polars":
        g = g.edges(df_to_engine(g._edges, _E.POLARS), "src", "dst").nodes(
            df_to_engine(g._nodes, _E.POLARS), "id")
    gi = g.gfql_index_all(engine=engine)
    seeds = g._nodes[:5] if engine == "pandas" else g._nodes.head(5)
    kwargs = dict(hops=1, return_as_wave_front=True,
                  edge_match={"etype": "zero"}, engine=engine)
    with pytest.raises(Exception) as base_err:
        g.hop(nodes=seeds, **kwargs)
    with pytest.raises(Exception) as idx_err:
        gi.hop(nodes=seeds, **kwargs)
    assert type(idx_err.value) is type(base_err.value)
    if engine == "pandas":
        from graphistry.compute.exceptions import GFQLSchemaError
        assert isinstance(base_err.value, GFQLSchemaError)


class TestIndexAutoPreservesPolarsFrames:
    """gfql_index_all(engine='auto') on a polars graph must index in place, NOT
    coerce-and-replace the frames with pandas. Regression pin for the C3 "polars
    hop is O(E)" mystery: resolve_engine(AUTO) maps polars->PANDAS, so create_index
    used to swap the frames to pandas, and every later hop(engine='polars') paid a
    full-frame pandas->polars conversion per call (~220ms at 4M edges vs a ~1ms
    indexed hop) while the pandas-engine index could never fingerprint-match."""

    def _pl_graph(self):
        pl = pytest.importorskip("polars")
        ndf = pl.DataFrame({"id": [0, 1, 2, 3], "type": ["a", "b", "a", "b"]})
        edf = pl.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]})
        return graphistry.nodes(ndf, "id").edges(edf, "src", "dst")

    def test_auto_keeps_polars_frames_and_polars_index(self):
        from graphistry.Engine import Engine
        g = self._pl_graph()
        gi = g.gfql_index_all()  # AUTO
        assert "polars" in type(gi._nodes).__module__
        assert "polars" in type(gi._edges).__module__
        # frames indexed in place: identity preserved so get_valid's `is` check holds
        assert gi._edges is g._edges
        from graphistry.compute.gfql.index import show_indexes
        idx = show_indexes(gi)
        assert set(idx["engine"]) == {Engine.POLARS.value}
        assert idx["valid"].all()

    def test_auto_polars_hop_engages_index(self, monkeypatch):
        # big enough that one seed passes the frontier-fraction cost gate
        pl = pytest.importorskip("polars")
        import graphistry.compute.gfql.index.api as index_api
        rng = np.random.default_rng(0)
        n_nodes, m = 2000, 12000
        ndf = pl.DataFrame({"id": np.arange(n_nodes)})
        edf = pl.DataFrame({"src": rng.integers(0, n_nodes, m), "dst": rng.integers(0, n_nodes, m)})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst").gfql_index_all()
        hits = []
        orig = index_api.index_seeded_hop

        def spy(*a, **k):
            out = orig(*a, **k)
            hits.append(out is not None)  # None = scan fallback; only a real serve counts
            return out
        monkeypatch.setattr(index_api, "index_seeded_hop", spy)
        r = g.hop(nodes=pl.DataFrame({"id": [7]}), hops=1, direction="forward", engine="polars")
        assert any(hits), "resident polars index did not SERVE the polars hop (call alone is not service)"
        # parity vs the pandas indexed oracle on the same data
        gp = graphistry.nodes(ndf.to_pandas(), "id").edges(edf.to_pandas(), "src", "dst").gfql_index_all()
        rp = gp.hop(nodes=pd.DataFrame({"id": [7]}), hops=1, direction="forward", engine="pandas")
        assert sorted(r._nodes.get_column("id").to_list()) == sorted(rp._nodes["id"].tolist())

    def test_explicit_engine_still_coerces(self):
        g = self._pl_graph()
        gi = g.gfql_index_all(engine="pandas")
        assert isinstance(gi._edges, pd.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}

    def test_edges_only_polars_keeps_legacy_pandas_path(self):
        pl = pytest.importorskip("polars")
        g = graphistry.edges(
            pl.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]}), "src", "dst"
        )
        gi = g.gfql_index_all()
        assert isinstance(gi._nodes, pd.DataFrame)
        assert isinstance(gi._edges, pd.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}

    def test_nodes_only_polars_keeps_legacy_pandas_path(self):
        pl = pytest.importorskip("polars")
        gi = (
            graphistry.nodes(pl.DataFrame({"id": [0, 1, 2]}), "id")
            .create_index("node_id")
        )
        assert isinstance(gi._nodes, pd.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}

    def test_lazyframe_auto_keeps_legacy_pandas_path(self):
        # M1 pin: LazyFrame frames under AUTO must coerce to pandas like master
        # (the eager-polars-only gate), never crash in a polars index build.
        pl = pytest.importorskip("polars")
        g = graphistry.nodes(pl.LazyFrame({"id": [0, 1, 2]}), "id").edges(
            pl.LazyFrame({"src": [0, 1], "dst": [1, 2]}), "src", "dst")
        gi = g.gfql_index_all()
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}

    def test_pandas_auto_stays_pandas(self):
        ndf = pd.DataFrame({"id": [0, 1, 2]})
        edf = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})
        gi = (
            graphistry.nodes(ndf, "id")
            .edges(edf, "src", "dst")
            .gfql_index_all()
        )
        assert isinstance(gi._nodes, pd.DataFrame)
        assert isinstance(gi._edges, pd.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}

    def test_cudf_auto_stays_cudf(self):
        cudf = pytest.importorskip("cudf")
        ndf = cudf.DataFrame({"id": [0, 1, 2]})
        edf = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
        gi = (
            graphistry.nodes(ndf, "id")
            .edges(edf, "src", "dst")
            .gfql_index_all()
        )
        assert isinstance(gi._nodes, cudf.DataFrame)
        assert isinstance(gi._edges, cudf.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"cudf"}

    @pytest.mark.parametrize("polars_side", ["nodes", "edges"])
    def test_mixed_eager_frames_keep_legacy_pandas_path(self, polars_side):
        pl = pytest.importorskip("polars")
        if polars_side == "nodes":
            ndf = pl.DataFrame({"id": [0, 1, 2]})
            edf = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})
        else:
            ndf = pd.DataFrame({"id": [0, 1, 2]})
            edf = pl.DataFrame({"src": [0, 1], "dst": [1, 2]})
        gi = (
            graphistry.nodes(ndf, "id")
            .edges(edf, "src", "dst")
            .gfql_index_all()
        )
        assert isinstance(gi._nodes, pd.DataFrame)
        assert isinstance(gi._edges, pd.DataFrame)
        from graphistry.compute.gfql.index import show_indexes
        assert set(show_indexes(gi)["engine"]) == {"pandas"}
