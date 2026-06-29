"""Differential parity: native polars chain() == pandas chain().

Phase 1 of the GFQL polars engine. Pandas is the oracle; polars must produce
identical node/edge sets and alias columns. See plans/gfql-polars-engine.
"""
import random

import pandas as pd
import pytest

import graphistry
from graphistry import gt, lt, ge, le, eq, ne, between, is_in, contains, startswith, endswith
from graphistry.compute.ast import n, e, e_forward, e_reverse, e_undirected

pl = pytest.importorskip("polars")


def _nset(g):
    df = g._nodes
    if df is None:
        return set()
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    return set(df[g._node].tolist())


def _eset(g):
    df = g._edges
    if df is None or len(df) == 0:
        return set()
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    return set(zip(df[g._source].tolist(), df[g._destination].tolist()))


def _named(g, col):
    df = g._nodes if col in (g._nodes.columns if g._nodes is not None else []) else g._edges
    if df is None or col not in df.columns:
        return None
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    key = g._node if g._node in df.columns else g._source
    return set(df[df[col].fillna(False).astype(bool)][key].tolist())


NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e", "f", "g"],
    "kind": ["x", "y", "y", "z", "x", "z", "y"],
    "name": ["alice", "bob", "carol", "dave", "erin", "frank", "grace"],
    "score": [10, 20, 30, 40, 50, 60, 70],
})
EDGES = pd.DataFrame({
    "s": ["a", "a", "b", "c", "d", "e", "b", "g", "c"],
    "d": ["b", "c", "c", "d", "e", "f", "d", "a", "g"],
    "rel": ["r1", "r2", "r1", "r2", "r1", "r2", "r1", "r2", "r1"],
    "w": [1, 2, 3, 4, 5, 6, 7, 8, 9],
})
BASE = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")

CHAINS = {
    "n-e-n": [n(), e_forward(), n()],
    "filter src": [n({"kind": "x"}), e_forward(), n()],
    "filter dst": [n(), e_forward(), n({"kind": "z"})],
    "edge_match": [n(), e_forward({"rel": "r1"}), n()],
    "pred gt": [n({"score": gt(15)}), e_forward(), n()],
    "pred lt": [n(), e_forward(), n({"score": lt(45)})],
    "pred ge": [n({"score": ge(30)}), e_forward(), n()],
    "pred le": [n(), e_forward(), n({"score": le(30)})],
    "pred eq": [n({"score": eq(30)}), e_forward(), n()],
    "pred ne": [n({"score": ne(30)}), e_forward(), n()],
    "pred between": [n({"score": between(20, 50)}), e_forward(), n()],
    "pred is_in": [n({"kind": is_in(["x", "z"])}), e_forward(), n()],
    "pred contains": [n({"name": contains("a")}), e_forward(), n()],
    "pred startswith": [n({"name": startswith("a")}), e_forward(), n()],
    "pred endswith": [n({"name": endswith("e")}), e_forward(), n()],
    "reverse": [n(), e_reverse(), n()],
    "undirected": [n({"kind": "y"}), e_undirected(), n()],
    "two-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n()],
    "three-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n(), e_forward(), n()],
    "edge+dst": [n(), e_forward({"rel": "r1"}), n({"kind": "y"})],
    "src+edge+dst": [n({"kind": "x"}), e_forward({"rel": "r2"}), n({"kind": "y"})],
    "named nodes": [n({"kind": "x"}, name="srcs"), e_forward(), n(name="dsts")],
    "named edge": [n(), e_forward(name="hop1"), n()],
    "named both": [n({"kind": "y"}, name="ys"), e_forward(name="h"), n(name="tgt")],
    "named reverse mid-filter": [n(name="a"), e_reverse(name="e1"), n({"kind": "y"}, name="b")],
    "node only": [n({"kind": "y"})],
    "reverse two-hop": [n({"kind": "z"}), e_reverse(), n(), e_reverse(), n()],
    "undirected filter": [n(), e_undirected({"rel": "r1"}), n({"score": gt(25)})],
    "empty result": [n({"kind": "x"}), e_forward({"rel": "nope"}), n()],
}


@pytest.mark.parametrize("cname", list(CHAINS))
def test_polars_chain_parity(cname):
    ch = CHAINS[cname]
    gp = BASE.chain(ch, engine="pandas")
    gl = BASE.chain(ch, engine="polars")
    assert "polars" in type(gl._nodes).__module__
    assert _nset(gp) == _nset(gl), f"node mismatch [{cname}]"
    assert _eset(gp) == _eset(gl), f"edge mismatch [{cname}]"
    for op in ch:
        nm = getattr(op, "_name", None)
        if nm:
            assert _named(gp, nm) == _named(gl, nm), f"alias[{nm}] mismatch [{cname}]"


@pytest.mark.parametrize("label,pred", [
    ("literal dot",    contains("a.c", regex=False)),               # literal: no metachar
    ("regex dot",      contains("a.c", regex=True)),                # '.' is a wildcard
    ("ci literal",     contains("A.C", regex=False, case=False)),   # literal + case-insensitive
    ("ci regex",       contains("A.C", regex=True, case=False)),    # regex + case-insensitive
])
def test_polars_contains_regex_and_case_parity(label, pred):
    """B1: polars Contains must honor ``regex=``/``flags=``/``case=`` like pandas — a
    literal contains with a regex metacharacter must NOT over-match (before the fix the
    polars lowering always used ``literal=False``, so ``contains('a.c', regex=False)``
    matched 'abc'). Differential parity vs the pandas oracle."""
    nodes = pd.DataFrame({"id": ["p", "q", "r", "s", "t"],
                          "name": ["a.c", "abc", "AxC", "a.c.d", "zzz"]})
    edges = pd.DataFrame({"s": ["p", "q", "r", "s"], "d": ["q", "r", "s", "t"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    ch = [n({"name": pred})]
    gp = g.chain(ch, engine="pandas")
    gl = g.chain(ch, engine="polars")
    assert "polars" in type(gl._nodes).__module__
    assert _nset(gp) == _nset(gl), f"[{label}] pandas {_nset(gp)} != polars {_nset(gl)}"


# ---- Randomized differential fuzzer (the CHANGELOG-advertised fuzzer) ----

def _rand_node(rng):
    r = rng.random()
    if r < 0.4:
        return n()
    if r < 0.6:
        return n({"kind": rng.choice(["x", "y", "z"])})
    if r < 0.8:
        return n({"score": gt(rng.randint(0, 80))})
    return n({"score": lt(rng.randint(20, 100))})


def _rand_edge(rng):
    ctor = rng.choice([e_forward, e_reverse, e_undirected])
    if rng.random() < 0.5:
        return ctor({"rel": rng.choice(["r1", "r2", "r3"])})
    return ctor()


def _rand_chain(rng):
    ops = [_rand_node(rng)]
    for _ in range(rng.randint(1, 3)):
        ops.append(_rand_edge(rng))
        ops.append(_rand_node(rng))
    return ops


def _rand_graph(rng):
    nn = rng.randint(4, 12)
    ids = [f"n{i}" for i in range(nn)]
    nodes = pd.DataFrame({
        "id": ids,
        "kind": [rng.choice(["x", "y", "z"]) for _ in ids],
        "score": [rng.randint(0, 100) for _ in ids],
    })
    ne = rng.randint(3, 24)
    edges = pd.DataFrame({
        "s": [rng.choice(ids) for _ in range(ne)],
        "d": [rng.choice(ids) for _ in range(ne)],
        "rel": [rng.choice(["r1", "r2", "r3"]) for _ in range(ne)],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("seed", range(60))
def test_polars_chain_fuzz_parity(seed):
    rng = random.Random(seed)
    g = _rand_graph(rng)
    ch = _rand_chain(rng)
    gp = g.chain(ch, engine="pandas")
    try:
        gl = g.chain(ch, engine="polars")
    except NotImplementedError:
        pytest.skip("deferred surface")
    assert _nset(gp) == _nset(gl), f"node mismatch seed={seed} chain={ch}"
    assert _eset(gp) == _eset(gl), f"edge mismatch seed={seed} chain={ch}"


# ---- Deferred-surface guards (must raise, never silently wrong) ----

@pytest.mark.parametrize("ch", [
    [n(), e(hops=2), n()],
    [n(), e(to_fixed_point=True), n()],
    [n(), e_undirected(), n(), e_undirected(), n()],  # undirected multi-edge
])
def test_polars_chain_deferred_raises(ch):
    with pytest.raises(NotImplementedError):
        BASE.chain(ch, engine="polars")


def test_polars_chain_node_query_raises():
    with pytest.raises(NotImplementedError):
        BASE.chain([n(query="score > 10"), e_forward(), n()], engine="polars")


# ---- Edge cases: empty graph, duplicate edges (multiplicity), edges-only ----

def test_polars_chain_empty_graph():
    g = graphistry.nodes(pd.DataFrame({"id": []}), "id").edges(
        pd.DataFrame({"s": [], "d": []}), "s", "d")
    gp = g.chain([n(), e_forward(), n()], engine="pandas")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    assert _nset(gp) == _nset(gl) == set()
    assert _eset(gp) == _eset(gl) == set()


def test_polars_chain_duplicate_edges_multiplicity():
    edges = pd.DataFrame({"s": ["a", "a", "a", "b"], "d": ["b", "b", "c", "c"]})  # (a,b) x2
    g = graphistry.nodes(pd.DataFrame({"id": ["a", "b", "c"]}), "id").edges(edges, "s", "d")
    gp = g.chain([n(), e_forward(), n()], engine="pandas")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    # assert on edge COUNT (set(zip) would hide a dropped parallel edge)
    assert len(gl._edges) == len(gp._edges)


def test_polars_chain_edges_only_runs():
    # No node table / binding: this used to crash the polars engine (the prior
    # BLOCKER). The pandas engine is itself degenerate on this input (it raises
    # in pandas concat internals on newer pandas, or returns a nan node), so we
    # do NOT compare to pandas here — we assert the polars engine runs and
    # returns the sensible materialized result.
    g = graphistry.edges(pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]}), "s", "d")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    assert _nset(gl) == {0, 1, 2}
    assert _eset(gl) == {(0, 1), (1, 2), (2, 0)}


def test_polars_chain_pandas_start_nodes():
    sn = pd.DataFrame({"id": ["a"]})
    gp = BASE.chain([n(), e_forward(), n()], engine="pandas", start_nodes=sn)
    gl = BASE.chain([n(), e_forward(), n()], engine="polars", start_nodes=sn)
    assert _nset(gp) == _nset(gl)
    assert _eset(gp) == _eset(gl)


def test_lazy_collect_cpu_and_engine_polars_helpers():
    """Cover the CPU lazy-collect path + the POLARS branches of the engine helpers
    (df_concat/df_cons/s_cons/df_to_engine) — exercised by the polars engine but not
    otherwise hit by the coverage suites. The GPU-target collect branches are
    pragma-no-cover (need a device CI lacks)."""
    import polars as pl
    from graphistry.compute.gfql.lazy import collect, collect_all
    from graphistry.Engine import Engine, df_concat, df_cons, s_cons, df_to_engine

    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    assert collect(lf).shape[0] == 3                 # CPU target -> eng is None -> collect() CPU branch
    out = collect_all([lf, lf])                       # collect_all CPU branch
    assert len(out) == 2 and all(o.shape[0] == 3 for o in out)

    assert df_cons(Engine.POLARS) is pl.DataFrame     # df_cons POLARS branch
    assert s_cons(Engine.POLARS) is pl.Series         # s_cons POLARS branch
    pdf = pl.DataFrame({"x": [1, 2]})
    assert df_concat(Engine.POLARS)([pdf, pdf]).shape[0] == 4   # df_concat POLARS branch
    assert df_to_engine(pdf, Engine.POLARS).shape[0] == 2       # df_to_engine POLARS branch


def test_gpu_target_raises_not_silent_cpu_fallback():
    """NO-CHEATING for the GPU target: a plan node that isn't GPU-executable must
    RAISE (NotImplementedError pointing at engine='polars'), never silently run on
    CPU and get reported as a GPU result. We can't exercise a real GPU in CI, so we
    drive the GPU collect path with a fake LazyFrame whose collect() fails and assert
    the failure is translated, not swallowed."""
    import pytest
    import polars as pl
    from graphistry.compute.gfql.lazy import (
        collect, target_mode, ExecutionTarget, _gpu_raise,
    )

    # pure translation: any GPU-exec failure -> NotImplementedError naming the CPU escape hatch
    err = _gpu_raise(ValueError("node X has no GPU implementation"))
    assert isinstance(err, NotImplementedError)
    assert "polars-gpu" in str(err) and "engine='polars'" in str(err)
    assert "node X has no GPU implementation" in str(err)

    if not hasattr(pl, "GPUEngine"):
        pytest.skip("polars build lacks GPUEngine; collect() GPU-target path needs it")

    class _FakeLF:
        def collect(self, engine=None):  # signature matches pl.LazyFrame.collect(engine=...)
            assert engine is not None     # GPU target must pass a GPUEngine, not None
            raise RuntimeError("GPU executor cannot run this node")

    with target_mode(ExecutionTarget.GPU):
        with pytest.raises(NotImplementedError) as ei:
            collect(_FakeLF())
    assert "engine='polars'" in str(ei.value)


def test_engine_polars_clean_dependency_errors():
    """engine='polars'/'polars-gpu' raise a CLEAN, actionable install error when the
    required library is missing — not a cryptic ImportError deep in coercion / the lazy
    engine, and not mislabeled as a not-GPU-capable plan. Guards live at the chain dispatch
    (compute/chain.py), pre-coercion."""
    import builtins
    import importlib.util
    import pytest
    import pandas as pd
    import graphistry
    from graphistry.compute.ast import n

    g = (graphistry.nodes(pd.DataFrame({"id": [0, 1]}), "id")
         .edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d"))

    # (1) polars not installed -> clean "requires the 'polars' package" (simulate the import failing)
    _orig_import = builtins.__import__

    def _block_polars(name, *a, **k):
        if name == "polars" or name.startswith("polars."):
            raise ImportError("No module named 'polars'")
        return _orig_import(name, *a, **k)

    builtins.__import__ = _block_polars
    try:
        with pytest.raises(ImportError, match=r"requires the 'polars' package"):
            g.gfql([n()], engine="polars")
    finally:
        builtins.__import__ = _orig_import

    # (2) polars-gpu without the RAPIDS cudf_polars stack -> clean RAPIDS install message
    # (distinct from the genuine not-GPU-capable signal). Skip where cudf_polars IS present.
    if importlib.util.find_spec("cudf_polars") is None:
        with pytest.raises(ImportError, match=r"cudf_polars"):
            g.gfql([n()], engine="polars-gpu")
