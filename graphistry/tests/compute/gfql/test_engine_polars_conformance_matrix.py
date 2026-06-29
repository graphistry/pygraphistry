"""Generative differential-conformance matrix for the polars engine (Phase 0 strategy).

THE CORE INVARIANT: for any query, on a non-pandas engine the result is EITHER parity-equal to
the pandas oracle OR an honest NotImplementedError — NEVER silently different, NEVER a silent
bridge, NEVER a non-NIE crash. This is checked across the cross-product of SURFACES
(native chain / Cypher string / let() DAG / call() on both) × OPS/PREDICATES, which is exactly
what the prior chain-only gates missed (the DAG silent-bridge bug et al.).

CPU lane (pandas-vs-polars) runs everywhere; a GPU lane (cudf / polars-gpu) is added on the dgx.
"""
import datetime
import numpy as np
import pandas as pd
import pytest
import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, call, let


# ---- graph with diverse dtypes (int, float w/ null, str, bool) ----
def _graph(seed=0):
    rng = np.random.default_rng(seed)
    nn = 30
    nd = pd.DataFrame({
        "id": np.arange(nn),
        "num": rng.integers(0, 100, nn).astype("int64"),
        "f": rng.normal(0, 1, nn),
        "name": [f"node.{i % 7}" for i in range(nn)],  # has a literal '.' (Contains-regex trap)
        "flag": rng.integers(0, 2, nn).astype(bool),
    })
    ne = 80
    ed = pd.DataFrame({
        "s": rng.integers(0, nn, ne),
        "d": rng.integers(0, nn, ne),
        "eid": np.arange(ne),
        "w": rng.integers(0, 50, ne),
    })
    return graphistry.nodes(nd, "id").edges(ed, "s", "d").bind(edge="eid")


def _sig(g):
    """Order-insensitive signature of a graph/row result (node-id set + edge-id set + shape)."""
    no, ed = g._nodes, g._edges
    if no is not None and "polars" in type(no).__module__:
        no = no.to_pandas()
    if ed is not None and "polars" in type(ed).__module__:
        ed = ed.to_pandas()
    nids = frozenset(no["id"].tolist()) if (no is not None and "id" in no.columns) else None
    eids = frozenset(ed["eid"].tolist()) if (ed is not None and "eid" in ed.columns) else None
    nshape = None if no is None else (no.shape[0], tuple(sorted(no.columns)))
    return (nids, eids, nshape)


def _run(g, query, engine):
    """('ok', sig) | ('nie',) | ('err', ExcTypeName)."""
    try:
        return ("ok", _sig(g.gfql(query, engine=engine)))
    except NotImplementedError:
        return ("nie",)
    except Exception as ex:  # any non-NIE error is itself a conformance failure to surface
        return ("err", type(ex).__name__)


def _assert_invariant(g, query, label):
    """polars result == pandas oracle, OR polars honestly NIEs. Never silent-divergence / crash."""
    base = _run(g, query, "pandas")
    pol = _run(g, query, "polars")
    if base[0] == "err":
        pytest.skip(f"{label}: pandas oracle itself errored ({base[1]})")
    if pol[0] == "nie":
        return  # honest decline — allowed
    assert pol[0] != "err", f"{label}: polars raised non-NIE {pol[1]} where pandas={base[0]}"
    assert pol == base, f"{label}: SILENT DIVERGENCE polars{pol} != pandas{base}"


# ---- predicate cross-product (the area that had 2 wrong-answer bugs) ----
def _predicate_queries():
    from graphistry.compute.predicates.numeric import GT, LT, GE, LE, Between
    from graphistry.compute.predicates.is_in import IsIn
    from graphistry.compute.predicates.str import Contains, Startswith, Endswith
    from graphistry.compute.predicates.numeric import IsNA, NotNA
    out = []
    for P, col, kw in [
        (GT, "num", {"val": 50}), (LT, "num", {"val": 50}), (GE, "num", {"val": 50}),
        (LE, "num", {"val": 50}),
        (Between, "num", {"lower": 20, "upper": 80}),
        (IsIn, "num", {"options": [1, 2, 3, 50, 51, 52]}),
        (Contains, "name", {"pat": "e.1", "regex": False}),   # literal metachar trap
        (Contains, "name", {"pat": "e.1", "regex": True}),
        (Contains, "name", {"pat": "ODE", "regex": False, "case": False}),
        (Startswith, "name", {"pat": "node"}),
        (Endswith, "name", {"pat": ".3"}),
        (IsNA, "name", {}), (NotNA, "name", {}),
    ]:
        out.append((f"pred:{P.__name__}({col},{kw})", [n({col: P(**kw)})]))
    return out


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_chain(label, query):
    _assert_invariant(_graph(1), query, f"chain {label}")


@pytest.mark.parametrize("label,query", _predicate_queries())
def test_conformance_predicates_dag(label, query):
    # SAME predicate via a let() DAG binding — must agree with the chain surface (parity or NIE).
    g = _graph(1)
    _assert_invariant(g, let({"a": query}), f"dag {label}")


# ---- traversal cross-product (single-hop parity; multi-hop/undirected-multi-edge NIE) ----
@pytest.mark.parametrize("label,query", [
    ("fwd1", [n({"id": [0]}), e_forward()]),
    ("rev1", [n({"id": [0]}), e_reverse()]),
    ("und1", [n({"id": [0]}), e_undirected()]),
    ("n-e-n", [n(), e_forward(), n({"num": 50})]),
    ("fwd-fwd", [n({"id": [0]}), e_forward(), e_forward()]),
    ("multihop", [n({"id": [0]}), e_forward(hops=2)]),                 # NIE expected
    ("und-multi", [n({"id": [0]}), e_undirected(), e_undirected()]),   # NIE expected
])
def test_conformance_traversals(label, query):
    _assert_invariant(_graph(2), query, f"traversal {label}")


# ---- cross-surface call() consistency (the silent-bridge bug class) ----
@pytest.mark.parametrize("fn", ["get_degrees", "hypergraph", "limit"])
def test_conformance_call_chain_vs_dag_consistent(fn):
    """A call must behave the SAME (parity or NIE) on the chain and the DAG surfaces — no surface
    may silently bridge where the other declines."""
    g = _graph(3)
    params = {"value": 2} if fn == "limit" else {}
    chain_q = [call(fn, params)] if params else [call(fn)]
    chain = _run(g, chain_q, "polars")
    dag = _run(g, let({"a": (call(fn, params) if params else call(fn))}), "polars")
    # Both honest-NIE, or both ok-with-equal-sig. Never one ok and the other NIE (the bug).
    if chain[0] == "nie":
        assert dag[0] == "nie", f"call '{fn}': chain NIE but DAG {dag} (silent-bridge regression)"
    elif chain[0] == "ok":
        assert dag[0] == "ok" and dag[1] == chain[1], f"call '{fn}': chain ok but DAG {dag}"


# ---- generative predicate fuzz across surfaces (verify-not-trust: broad, seeded) ----
def test_conformance_predicate_fuzz():
    from graphistry.compute.predicates.numeric import GT, LT, Between
    from graphistry.compute.predicates.is_in import IsIn
    from graphistry.compute.predicates.str import Contains, Startswith, Endswith
    cols_num = ["num", "w_unused"]  # w is on edges; keep node cols
    fails = []
    for t in range(60):
        rng = np.random.default_rng(7000 + t)
        g = _graph(7000 + t)
        choice = int(rng.integers(0, 6))
        if choice == 0:
            q = [n({"num": GT(val=int(rng.integers(0, 100)))})]
        elif choice == 1:
            lo = int(rng.integers(0, 50)); q = [n({"num": Between(lower=lo, upper=lo + 30)})]
        elif choice == 2:
            q = [n({"num": IsIn(options=[int(x) for x in rng.integers(0, 100, 5)])})]
        elif choice == 3:
            q = [n({"name": Contains(pat="e.%d" % int(rng.integers(0, 7)),
                                     regex=bool(rng.integers(0, 2)))})]
        elif choice == 4:
            q = [n({"name": Startswith(pat="node")})]
        else:
            q = [n({"name": Endswith(pat=".%d" % int(rng.integers(0, 7)))})]
        try:
            _assert_invariant(g, q, f"fuzz {t}")
        except AssertionError as e:
            fails.append(str(e)[:120])
    assert not fails, f"predicate-conformance fuzz failures:\n" + "\n".join(fails)
