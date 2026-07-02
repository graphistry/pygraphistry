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


def _emult(g):
    """Edge MULTISET (Counter) — catches a dropped parallel/self-loop copy or an edge SWAP
    that preserves count, which `len()` and `_eset` (a set) both miss. The min_hops
    recompute-all combine (fuzz seeds 24/48) diverged exactly here."""
    from collections import Counter
    df = g._edges
    if df is None or len(df) == 0:
        return Counter()
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    return Counter(zip(df[g._source].tolist(), df[g._destination].tolist()))


def _node_attrs(g):
    """Null-aware per-node ATTRIBUTE map — catches a node present in BOTH outputs but with a
    different (or NULL) attribute cell, which `_nset` (id-set) cannot see. The min_hops
    null-attr-on-source-side-endpoint rule (fuzz seed-48 n5/n7: kind=y but carried as NaN,
    so a downstream `kind=y` filter rejects them) lives in exactly this dimension. Normalizes
    NaN/None→None and int/float→float (pandas upcasts an int col to float once a NaN-stub row
    is concatenated, while polars keeps Int64+null — without this you get spurious 5 != 5.0)."""
    df = g._nodes
    if df is None:
        return {}
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    key = g._node
    # Exclude internal engine columns (`__gfql_*`, e.g. the pandas auto hop-label
    # `__gfql_output_node_hop__` that ASTEdge.execute leaks into the min_hops result but polars
    # does not) — they are implementation detail, not user-facing data, so not a parity concern.
    cols = sorted(c for c in df.columns if c != key and not c.startswith("__gfql_"))

    def norm(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    return {row[key]: tuple(norm(row[c]) for c in cols) for _, row in df.iterrows()}


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
    match = {"rel": rng.choice(["r1", "r2", "r3"])} if rng.random() < 0.5 else None
    # Multi-hop ONLY for directed edges. e_undirected stays single-hop on purpose:
    # undirected-in-a-multi-edge-chain is a SEPARATE deferred defect (chain.py guard),
    # and a multi-HOP undirected edge would skip-mask it. Forward/reverse multi-hop
    # NIEs at the chain guard TODAY (is_simple_single_hop -> NotImplementedError), so
    # the fuzz SKIPs cleanly via the except below; once Stage 1 narrows the guard these
    # become live parity cases — no test edit needed.
    if rng.random() < 0.4:
        r = rng.random()
        if r < 0.3:
            hops = rng.randint(2, 3)                       # exact count: hops in {2,3}
            return ctor(match, hops=hops) if match else ctor(hops=hops)
        if r < 0.55:
            mx = rng.randint(2, 4)                         # variable-length 1..max_hops
            return ctor(match, max_hops=mx) if match else ctor(max_hops=mx)
        if r < 0.75 and ctor is not e_undirected:
            # min_hops>1 lower bound — DIRECTED + finite max_hops only (undirected min_hops and
            # min_hops+to_fixed_point stay NIE -> skip-masked via the except).
            lo = rng.randint(2, 3)
            hi = lo + rng.randint(0, 2)
            return ctor(match, min_hops=lo, max_hops=hi) if match else ctor(min_hops=lo, max_hops=hi)
        # to_fixed_point (unbounded) — forward/reverse only. UNDIRECTED to_fixed_point stays NIE.
        if ctor is e_undirected:
            return ctor(match, hops=2) if match else ctor(hops=2)
        return ctor(match, to_fixed_point=True) if match else ctor(to_fixed_point=True)
    return ctor(match) if match else ctor()


def _rand_chain(rng):
    ops = [_rand_node(rng)]
    for _ in range(rng.randint(1, 3)):
        ops.append(_rand_edge(rng))
        ops.append(_rand_node(rng))
    return ops


def _rand_graph(rng):
    # Denser-than-nodes topology so multi-hop chains actually traverse >1 edge:
    # a backbone DIRECTED CYCLE (guarantees A->B->...->A multi-hop reachability),
    # a self-loop (hops>=2 self-reach + multiplicity), a parallel DUPLICATE edge
    # (multiplicity under multi-hop), plus random chords. Deterministic per seed.
    nn = rng.randint(4, 8)                                  # fewer nodes => denser
    ids = [f"n{i}" for i in range(nn)]
    nodes = pd.DataFrame({
        "id": ids,
        "kind": [rng.choice(["x", "y", "z"]) for _ in ids],
        "score": [rng.randint(0, 100) for _ in ids],
    })
    es = []                                                # (src, dst, rel) tuples
    for i in range(nn):                                    # backbone cycle n0->n1->...->n0
        es.append((ids[i], ids[(i + 1) % nn], "r1"))
    sl = rng.choice(ids)                                   # self-loop
    es.append((sl, sl, "r2"))
    es.append((ids[0], ids[1 % nn], "r1"))                 # parallel dup of n0->n1
    for _ in range(rng.randint(nn, 3 * nn)):               # random chords; |E| >> |V|
        es.append((rng.choice(ids), rng.choice(ids), rng.choice(["r1", "r2", "r3"])))
    edges = pd.DataFrame({"s": [e[0] for e in es], "d": [e[1] for e in es], "rel": [e[2] for e in es]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("seed", range(0, 500))
def test_polars_chain_fuzz_parity(seed):
    rng = random.Random(seed)
    g = _rand_graph(rng)
    ch = _rand_chain(rng)
    gp = g.chain(ch, engine="pandas")
    try:
        gl = g.chain(ch, engine="polars")
    except NotImplementedError:
        pytest.skip("deferred surface")   # multi-hop fwd/rev NIEs at chain guard today
    assert _nset(gp) == _nset(gl), f"node mismatch seed={seed} chain={ch}"
    assert _eset(gp) == _eset(gl), f"edge mismatch seed={seed} chain={ch}"
    # Stricter than id/endpoint sets: edge MULTIPLICITY (Counter) and null-aware node ATTRIBUTES
    # are the two dimensions the min_hops bugs (24/404/48) lived in and that set-equality missed.
    assert _emult(gp) == _emult(gl), f"edge-multiplicity mismatch seed={seed} chain={ch}"
    assert _node_attrs(gp) == _node_attrs(gl), f"node-attr mismatch seed={seed} chain={ch}"


# ---- AMPLIFIED min_hops fuzz: every edge min_hops>1 ALWAYS followed by an attribute filter
# (seed-48 shape on EVERY seed), multiple min_hops steps (the recompute-all combine that broke
# seeds 24/48), and a sparse-graph variant (exercises the max_reached<min empty gate that the
# always-dense _rand_graph under-tests). The base fuzz hits this shape only ~3% of seeds. ----

def _rand_node_attr(rng):
    """Always attribute-bearing (never bare n()) so a min_hops null-attr carry is CONSUMED by a
    downstream filter — the only way the seed-48 class becomes observable through the chain.
    Uses ONLY predicates that agree pandas-vs-polars on NULL (eq/gt/between all EXCLUDE a null
    cell); ne() is deliberately excluded because pandas `!= x` KEEPS a NaN cell while polars
    (cypher 3-valued `null<>x`→null) drops it — a SEPARATE pre-existing predicate divergence
    (see test_ne_on_null_is_three_valued_logic), not a min_hops concern."""
    r = rng.random()
    if r < 0.45:
        return n({"kind": rng.choice(["x", "y", "z"])})
    if r < 0.75:
        return n({"score": gt(rng.randint(0, 80))})
    return n({"score": between(rng.randint(0, 40), rng.randint(50, 100))})


def _rand_minhops_edge(rng):
    ctor = rng.choice([e_forward, e_reverse])                      # directed only (undirected = NIE)
    match = {"rel": rng.choice(["r1", "r2", "r3"])} if rng.random() < 0.4 else None
    lo = rng.randint(2, 3)
    hi = lo + rng.randint(0, 2)
    return ctor(match, min_hops=lo, max_hops=hi) if match else ctor(min_hops=lo, max_hops=hi)


def _rand_graph_sparse(rng):
    nn = rng.randint(4, 9)
    ids = [f"n{i}" for i in range(nn)]
    nodes = pd.DataFrame({
        "id": ids,
        "kind": [rng.choice(["x", "y", "z"]) for _ in ids],
        "score": [rng.randint(0, 100) for _ in ids],
    })
    es = [(ids[i], ids[i + 1], "r1") for i in range(nn - 1)]        # path: NO backbone cycle
    for _ in range(rng.randint(0, nn // 2)):                        # few chords, stays sparse
        es.append((rng.choice(ids), rng.choice(ids), rng.choice(["r1", "r2"])))
    edges = pd.DataFrame({"s": [e[0] for e in es], "d": [e[1] for e in es], "rel": [e[2] for e in es]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("seed", range(0, 400))
def test_polars_chain_fuzz_minhops_attrfilter_parity(seed):
    rng = random.Random(seed)
    g = _rand_graph_sparse(rng) if rng.random() < 0.35 else _rand_graph(rng)
    ops = [_rand_node(rng)]
    for _ in range(rng.randint(1, 3)):
        ops.append(_rand_minhops_edge(rng))
        ops.append(_rand_node_attr(rng))
    gp = g.chain(ops, engine="pandas")
    try:
        gl = g.chain(ops, engine="polars")
    except NotImplementedError:
        pytest.skip("deferred surface")
    assert _nset(gp) == _nset(gl), f"node mismatch seed={seed} chain={ops}"
    assert _eset(gp) == _eset(gl), f"edge mismatch seed={seed} chain={ops}"
    assert _emult(gp) == _emult(gl), f"edge-multiplicity mismatch seed={seed} chain={ops}"
    assert _node_attrs(gp) == _node_attrs(gl), f"node-attr mismatch seed={seed} chain={ops}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_polars_chain_minhops1_equiv_default(k):
    # Metamorphic (no oracle): min_hops=1 is the default lower bound, so e(min_hops=1, max_hops=k)
    # must be identical to e(max_hops=k) on the SAME engine.
    rng = random.Random(900 + k)
    g = _rand_graph(rng)
    a = g.chain([n(), e_forward(min_hops=1, max_hops=k), n()], engine="polars")
    b = g.chain([n(), e_forward(max_hops=k), n()], engine="polars")
    assert _nset(a) == _nset(b) and _emult(a) == _emult(b), f"min_hops=1 != default @k={k}"


def test_ne_on_null_is_three_valued_logic():
    # openCypher/SQL 3-valued logic: `null <> x` is NULL -> a null cell is NOT a match (you cannot
    # prove an unknown value is unequal to x), so a null-kind node is EXCLUDED by ne() — same as
    # eq/gt and as `NOT a.kind = x`. pandas used to KEEP it (NaN != x -> True); now fixed to match
    # cudf + the polars engine. n2 has kind=NULL and must be absent from BOTH engines' results.
    nodes = pd.DataFrame({"id": ["n0", "n1", "n2", "n3"], "kind": ["x", "y", None, "z"]})
    edges = pd.DataFrame({"s": ["n0", "n1", "n2", "n3"], "d": ["n1", "n2", "n3", "n0"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    gp = g.chain([n({"kind": ne("y")})], engine="pandas")
    gl = g.chain([n({"kind": ne("y")})], engine="polars")
    assert _nset(gp) == {"n0", "n3"} == _nset(gl)   # n1 fails eq, n2 (null) excluded by 3VL


def test_membership_on_null_is_three_valued_logic():
    # openCypher/SQL 3VL: `null IN [...]` is null -> a NULL cell is never a list member (and a null
    # in the list cannot make a null cell match). n2 has kind=NULL and must be excluded by the
    # membership filter on BOTH engines (cuDF used to keep it; now fixed in filter_by_dict).
    nodes = pd.DataFrame({"id": ["n0", "n1", "n2", "n3"], "kind": ["x", "y", None, "z"]})
    edges = pd.DataFrame({"s": ["n0", "n1", "n2", "n3"], "d": ["n1", "n2", "n3", "n0"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    gp = g.chain([n({"kind": ["x", "z", None]})], engine="pandas")
    gl = g.chain([n({"kind": ["x", "z", None]})], engine="polars")
    assert _nset(gp) == {"n0", "n3"} == _nset(gl)   # x,z match; null excluded despite None in list


@pytest.mark.parametrize("k", [2, 3, 4])
def test_polars_chain_maxhops_monotone(k):
    # Metamorphic (no oracle): the forward-reachable node set is non-decreasing in max_hops.
    rng = random.Random(950 + k)
    g = _rand_graph(rng)
    lo = g.chain([n(), e_forward(max_hops=k), n()], engine="polars")
    hi = g.chain([n(), e_forward(max_hops=k + 1), n()], engine="polars")
    assert _nset(lo) <= _nset(hi), f"max_hops node set not monotone @k={k}"


# ---- Deferred-surface guards (must raise, never silently wrong) ----

@pytest.mark.parametrize("ch", [
    [n(), e(to_fixed_point=True), n()],          # undirected to_fixed_point (components/2-core)
])
def test_polars_chain_deferred_raises(ch):
    with pytest.raises(NotImplementedError):
        BASE.chain(ch, engine="polars")


@pytest.mark.parametrize("ch", [
    [n(), e_undirected(to_fixed_point=True), n()],         # undirected to_fixed_point (components/2-core)
    [n(), e_undirected(min_hops=2, max_hops=3), n()],      # UNDIRECTED min_hops>1 (stays NIE)
])
def test_polars_chain_multihop_deferred_raises(ch):
    # These multi-hop surfaces STAY NIE after native fixed-length hops=N (Stage 1), native
    # single-hop AND fixed multi-hop undirected (Stage 3 + undirected-multihop), native forward/reverse
    # to_fixed_point (Stage 4), and native forward/reverse min_hops>1 (Stage 5, the layered backward-tree
    # walk + endpoint/label/seed-strip node rule): only UNDIRECTED min_hops>1 and undirected
    # to_fixed_point remain (both need pandas connected-components + 2-core seed retention,
    # hop.py:817-887, with no vectorized polars analogue).
    with pytest.raises(NotImplementedError):
        BASE.chain(ch, engine="polars")


def test_polars_chain_node_query_raises():
    with pytest.raises(NotImplementedError):
        BASE.chain([n(query="score > 10"), e_forward(), n()], engine="polars")


# ---- Adversarial multi-hop parity (explicit oracle cases; NIE-skip until Stage 1) ----
# Small deterministic graph: a<->b 2-cycle, b->c->d path, d->d self-loop, a->b dup, f isolated.
ADV_NODES = pd.DataFrame({
    "id":   ["a", "b", "c", "d", "e", "f"],
    "kind": ["x", "y", "y", "z", "x", "z"],
    "score": [1, 2, 3, 4, 5, 6],
})
ADV_EDGES = pd.DataFrame({
    "s":   ["a", "b", "b", "c", "d", "a"],
    "d":   ["b", "a", "c", "d", "d", "b"],
    "rel": ["r1", "r1", "r2", "r1", "r2", "r1"],
})
ADV = graphistry.nodes(ADV_NODES, "id").edges(ADV_EDGES, "s", "d")

ADV_CHAINS = {
    # 1. A->B->A cycle @hops2: from a, depth-2 forward reaches a (a->b->a) and c (a->b->c)
    "cycle-hops2":        [n({"id": ["a"]}), e_forward(hops=2), n()],
    # 2. self-loop @hops2: d->d->d stays {d}; edge (d,d) retained
    "selfloop-hops2":     [n({"id": ["d"]}), e_forward(hops=2), n()],
    # 3. parallel dup multiplicity: a->b exists twice; depth-2 must keep BOTH copies (count, not set)
    "parallel-dup-hops2": [n({"id": ["a"]}), e_forward(hops=2), n()],
    # 4. unreachable-within-N -> empty: isolated f has no out-edges; any depth -> empty result
    "isolated-empty":     [n({"id": ["f"]}), e_forward(hops=3), n()],
    # 5. reverse multi-hop: predecessors of d within 2 hops back = {c (c->d), b (b->c->d), d (self)}
    "reverse-hops2":      [n({"id": ["d"]}), e_reverse(hops=2), n()],
    # 6. both-endpoint-filtered multi-hop: x-seed {a,e} -> depth<=3 -> z-end {d,f}; only a->b->c->d reaches d
    "both-filtered-hops3": [n({"kind": "x"}), e_forward(max_hops=3), n({"kind": "z"})],
    # 7. hops > diameter: bounded BFS depth 10 == forward closure of a = {a,b,c,d}
    "hops-gt-diameter":   [n({"id": ["a"]}), e_forward(hops=10), n()],
    # 8. sandwiched single+multi+single: 1-hop, then 2-hop, then 1-hop
    "sandwiched":         [n({"id": ["a"]}), e_forward(), n(), e_forward(hops=2), n(), e_forward(), n()],
}


@pytest.mark.parametrize("cname", list(ADV_CHAINS))
def test_polars_chain_adversarial_multihop_parity(cname):
    ch = ADV_CHAINS[cname]
    gp = ADV.chain(ch, engine="pandas")
    try:
        gl = ADV.chain(ch, engine="polars")
    except NotImplementedError:
        pytest.skip("multi-hop deferred (chain guard) — activates after Stage 1")
    assert _nset(gp) == _nset(gl), f"node mismatch [{cname}]"
    assert _eset(gp) == _eset(gl), f"edge mismatch [{cname}]"
    # multiplicity: the parallel-dup / self-loop cases must not silently drop a duplicate edge
    gpe, gle = gp._edges, gl._edges
    assert (0 if gpe is None else len(gpe)) == (0 if gle is None else len(gle)), f"edge-count [{cname}]"


# ---- Adversarial UNDIRECTED-multi-edge parity (Stage 3: native single-hop undirected in
# multi-edge chains; the seed-18 defect class — backward pass must thread BOTH endpoints) ----
UND_CHAINS = {
    # seed-18 class: >1 undirected edge + intermediate node filter (the original drop-a-node bug)
    "und-und-midfilter":  [n({"id": ["a"]}), e_undirected(), n({"kind": "y"}), e_undirected(), n()],
    "und-und-und":        [n({"id": ["a"]}), e_undirected(), n(), e_undirected(), n(), e_undirected(), n()],
    "und-alt-filter":     [n(), e_undirected(), n({"kind": "z"}), e_undirected(), n({"kind": "x"})],
    # mixed directed + undirected single-hop (newly enabled by Stage 3)
    "fwd-then-und":       [n({"id": ["a"]}), e_forward(), n(), e_undirected(), n()],
    "und-then-rev":       [n({"id": ["d"]}), e_undirected(), n(), e_reverse(), n()],
    # named node/edge across an undirected boundary (exercises _apply_node_names + named edge combine)
    "und-named":          [n(name="s"), e_undirected(name="ue"), n({"kind": "y"}, name="m"), e_undirected(), n(name="t")],
    # self-loop / parallel-dup adjacent to undirected edges (multiplicity)
    "und-selfloop":       [n({"id": ["d"]}), e_undirected(), n(), e_undirected(), n()],
    # fixed-length UNDIRECTED multi-hop (uses generic backward hop + path-aware recompute)
    "und-hops2":          [n({"id": ["a"]}), e_undirected(hops=2), n()],
    "und-maxhops3":       [n({"id": ["a"]}), e_undirected(max_hops=3), n()],
    "und-hops2-midfilter": [n({"id": ["a"]}), e_undirected(hops=2), n({"kind": "z"}), e_undirected(), n()],
    "und-hops2-mixed":    [n({"id": ["a"]}), e_undirected(hops=2), n(), e_forward(), n()],
    # undirected SINGLE-hop + directed FIXED multi-hop (the recompute must not collide with the
    # undirected both-endpoint backward override on the shared middle wavefront)
    "und-then-fwdhops2":  [n({"id": ["a"]}), e_undirected(), n(), e_forward(hops=2), n()],
    "fwdhops2-then-und":  [n({"id": ["a"]}), e_forward(hops=2), n(), e_undirected(), n()],
    "und-mid-fwdhops2":   [n({"id": ["a"]}), e_undirected(), n({"kind": "y"}), e_forward(hops=2), n({"kind": "z"})],
}


# ---- Adversarial to_fixed_point parity (Stage 4: unbounded variable-length forward/reverse;
# the cyclic ADV graph stresses termination — a<->b cycle, d->d self-loop, parallel a->b) ----
TOFP_CHAINS = {
    "tofp-fwd-from-a":   [n({"id": ["a"]}), e_forward(to_fixed_point=True), n()],
    "tofp-rev-from-d":   [n({"id": ["d"]}), e_reverse(to_fixed_point=True), n()],
    "tofp-fwd-selfloop": [n({"id": ["d"]}), e_forward(to_fixed_point=True), n()],
    "tofp-midfilter":    [n({"id": ["a"]}), e_forward(to_fixed_point=True), n({"kind": "z"})],
    "tofp-isolated":     [n({"id": ["f"]}), e_forward(to_fixed_point=True), n()],
    "tofp-sandwiched":   [n({"id": ["a"]}), e_forward(), n(), e_forward(to_fixed_point=True), n()],
    "tofp-named":        [n({"id": ["a"]}, name="s"), e_forward(to_fixed_point=True, name="te"), n(name="t")],
}


@pytest.mark.parametrize("cname", list(TOFP_CHAINS))
def test_polars_chain_to_fixed_point_parity(cname):
    ch = TOFP_CHAINS[cname]
    gp = ADV.chain(ch, engine="pandas")
    gl = ADV.chain(ch, engine="polars")  # Stage 4: native, must NOT raise nor hang
    assert "polars" in type(gl._nodes).__module__, f"[{cname}] not native polars (silent bridge!)"
    assert _nset(gp) == _nset(gl), f"node mismatch [{cname}]"
    assert _eset(gp) == _eset(gl), f"edge mismatch [{cname}]"
    gpe, gle = gp._edges, gl._edges
    assert (0 if gpe is None else len(gpe)) == (0 if gle is None else len(gle)), f"edge-count [{cname}]"
    for op in ch:
        nm = getattr(op, "_name", None)
        if nm:
            assert _named(gp, nm) == _named(gl, nm), f"alias[{nm}] mismatch [{cname}]"


@pytest.mark.parametrize("cname", list(UND_CHAINS))
def test_polars_chain_undirected_multiedge_parity(cname):
    ch = UND_CHAINS[cname]
    gp = ADV.chain(ch, engine="pandas")
    gl = ADV.chain(ch, engine="polars")  # Stage 3: native, must NOT raise
    assert "polars" in type(gl._nodes).__module__, f"[{cname}] not native polars (silent bridge!)"
    assert _nset(gp) == _nset(gl), f"node mismatch [{cname}]"
    assert _eset(gp) == _eset(gl), f"edge mismatch [{cname}]"
    gpe, gle = gp._edges, gl._edges
    assert (0 if gpe is None else len(gpe)) == (0 if gle is None else len(gle)), f"edge-count [{cname}]"
    for op in ch:
        nm = getattr(op, "_name", None)
        if nm:
            assert _named(gp, nm) == _named(gl, nm), f"alias[{nm}] mismatch [{cname}]"


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


def test_gpu_executor_mode_flag(monkeypatch):
    """GFQL_POLARS_GPU_EXECUTOR selects the cudf-polars executor: default 'in-memory',
    opt-in 'streaming' (larger-than-device escape hatch), invalid -> in-memory. raise_on_fail
    stays True (NO-CHEATING) regardless. Mocks pl.GPUEngine so no GPU is needed."""
    import polars as pl
    from graphistry.compute.gfql import lazy
    from graphistry.compute.gfql.lazy import _engine_for, ExecutionTarget

    captured = {}

    class _FakeGPUEngine:
        def __init__(self, **kw):
            captured.clear()
            captured.update(kw)

    monkeypatch.setattr(pl, "GPUEngine", _FakeGPUEngine)

    monkeypatch.setattr(lazy, "_GPU_EXECUTOR", "in-memory")
    _engine_for(ExecutionTarget.GPU)
    assert captured == {"executor": "in-memory", "raise_on_fail": True}

    monkeypatch.setattr(lazy, "_GPU_EXECUTOR", "streaming")
    _engine_for(ExecutionTarget.GPU)
    assert captured["executor"] == "streaming" and captured["raise_on_fail"] is True

    monkeypatch.setattr(lazy, "_GPU_EXECUTOR", "bogus")
    _engine_for(ExecutionTarget.GPU)
    assert captured["executor"] == "in-memory"  # invalid value falls back

    # CPU target unaffected (returns None, no engine)
    assert _engine_for(ExecutionTarget.CPU) is None


def test_engine_polars_no_silent_call_bridge():
    """NO-CHEATING: a DAG let() binding of a not-yet-native Plottable-method call (here
    hypergraph) raises NotImplementedError under engine='polars' (matching the chain
    surface) instead of silently running on pandas and coercing the result back.
    engine='pandas' is unaffected. (get_degrees / get_indegrees / get_outdegrees are now
    lowered natively — see the conformance matrix — so hypergraph, an architecturally
    pandas-only entity-transform with no native polars impl, stands in as the still-declined
    Plottable-method call: it runs on pandas internally and the no-bridge guard declines
    coercing its pandas result back to polars.)"""
    import pytest
    import pandas as pd
    import graphistry
    from graphistry.compute.ast import call, let

    g = (graphistry.nodes(pd.DataFrame({"id": [0, 1, 2]}), "id")
         .edges(pd.DataFrame({"s": [0, 1], "d": [1, 2]}), "s", "d"))
    # chain surface already declines; the DAG surface must too (was a silent bridge).
    with pytest.raises(NotImplementedError):
        g.gfql([call("hypergraph")], engine="polars")
    with pytest.raises(NotImplementedError):
        g.gfql(let({"d": call("hypergraph")}), engine="polars")
    # pandas DAG path still works.
    assert g.gfql(let({"d": call("hypergraph")}), engine="pandas") is not None


def test_engine_polars_predicate_correctness_fixes():
    """Contains honors the regex flag (a literal pattern with regex metachars matches literally,
    not as a regex); temporal comparison declines cleanly (no broken expr leak)."""
    import datetime
    import operator
    import polars as pl
    import pandas as pd
    import graphistry
    from graphistry.compute.ast import n
    from graphistry.compute.predicates.str import Contains
    from graphistry.compute.gfql.lazy.engine.polars.predicates import _cmp_expr

    nd = pd.DataFrame({"id": [0, 1, 2, 3], "name": ["a.b", "axb", "a.bxx", "zz"]})
    g = graphistry.nodes(nd, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")

    def ids(gg):
        no = gg._nodes
        no = no.to_pandas() if "polars" in type(no).__module__ else no
        return sorted(no["id"].tolist())

    for regex in (False, True):
        q = [n({"name": Contains(pat="a.b", regex=regex)})]
        assert ids(g.gfql(q, engine="pandas")) == ids(g.gfql(q, engine="polars")), f"regex={regex}"
    # literal 'a.b' matches 'a.b' + 'a.bxx' but NOT 'axb' (the metachar bug)
    assert ids(g.gfql([n({"name": Contains(pat="a.b", regex=False)})], engine="polars")) == [0, 2]

    # temporal val -> _cmp_expr declines (None) so upstream raises honest NIE; numeric still lowers.
    assert _cmp_expr(None, operator.gt, datetime.date(2020, 1, 1)) is None
    assert _cmp_expr(pl.col("x"), operator.gt, 5) is not None
