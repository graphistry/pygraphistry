"""Parity + gating tests for the #1755 seeded typed-hop fast path.

Two layers close the seeded-Cypher abstraction tax on pandas:
  * native  — `_try_chain_fast_path` / `_seeded_typed_hop_pandas_cudf` accelerate
    a seeded typed 1-hop chain [n({id}), e(edge_match), n({type})];
  * cypher   — `_execute_seeded_typed_hop_fast_path` accelerates the lowered
    MATCH (m {id})-[:T]->(p) RETURN p string surface.

Both are value-identical to the full path by construction (same rows/columns;
row order and index may differ — comparisons canonicalize); these tests pin that
(fast-on vs fast-off, differential) and that the fast path actually ENGAGES for
the accelerated shapes and DECLINES (falls through) for everything else,
including full-path side-channels (policy hooks, same-path WHERE, OPTIONAL
null rows, WITH..MATCH carried seeds, list-`labels` columns, null ids).
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse
import graphistry.compute.chain as chain_mod
import graphistry.compute.gfql_unified as gfql_unified


def _graph(n_persons=1500, n_messages=6000, seed=0):
    """Message -> Person HAS_CREATOR graph (the #1755 probe shape). `age` is only
    on Person rows, so the concatenated node frame carries a float `age` column
    with NaN on Messages — exercising mixed dtypes through the projection."""
    rng = np.random.default_rng(seed)
    persons = pd.DataFrame({
        "id": np.arange(n_persons), "type": "Person",
        "age": rng.integers(20, 60, n_persons),
    })
    messages = pd.DataFrame({"id": np.arange(n_persons, n_persons + n_messages), "type": "Message"})
    ndf = pd.concat([persons, messages], ignore_index=True)
    edf = pd.DataFrame({
        "src": np.arange(n_persons, n_persons + n_messages),
        "dst": rng.integers(0, n_persons, n_messages),
        "type": "HAS_CREATOR",
    })
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst"), n_persons


def _canon_nodes(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    cols = sorted(str(c) for c in df.columns)
    df.columns = [str(c) for c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)[cols] if cols else df


def _engine_graph(engine):
    """Build the probe graph in the requested engine (cuDF skips if unavailable)."""
    g, P = _graph()
    if engine == "pandas":
        return g, P
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(
            cudf.from_pandas(pd.DataFrame(g._nodes)), "id"
        ).edges(cudf.from_pandas(pd.DataFrame(g._edges)), "src", "dst"), P
    raise ValueError(engine)


def _run_diff(g, engine, query, fast):
    """Run `query` with BOTH seeded fast paths (native + cypher) either live or
    forced-off, so a single differential asserts fast==full for any shape/engine."""
    real_native = chain_mod._try_chain_fast_path
    real_cyp = gfql_unified._execute_seeded_typed_hop_fast_path
    try:
        if not fast:
            chain_mod._try_chain_fast_path = lambda *a, **k: None
            gfql_unified._execute_seeded_typed_hop_fast_path = lambda *a, **k: None
        return g.gfql(query, engine=engine)
    finally:
        chain_mod._try_chain_fast_path = real_native
        gfql_unified._execute_seeded_typed_hop_fast_path = real_cyp


def _oracle_creators(g, seed):
    """INDEPENDENT ground truth (NOT via the query engine): the Person ids that
    are HAS_CREATOR destinations of the seeded Message. Computed directly from
    the raw edge/node frames with plain pandas boolean masks, so it cannot share
    a bug with either the fast path or the full chain path."""
    edf = g._edges
    ndf = g._nodes
    creator_ids = edf.loc[
        (edf["src"] == seed) & (edf["type"] == "HAS_CREATOR"), "dst"
    ].tolist()
    # restrict to Person-typed endpoints (the typed n({type:Person}) filter)
    person_ids = set(ndf.loc[ndf["type"] == "Person", "id"].tolist())
    return sorted(cid for cid in creator_ids if cid in person_ids)


# ---------------------------------------------------------------------------
# native seeded typed hop  ([n({id}), e(edge_match), n({type})])
# ---------------------------------------------------------------------------

class TestNativeSeededTypedHop:
    def _run(self, g, ops, force_full):
        real = chain_mod._try_chain_fast_path
        try:
            if force_full:
                chain_mod._try_chain_fast_path = lambda *a, **k: None
            return g.gfql(ops, engine="pandas")
        finally:
            chain_mod._try_chain_fast_path = real

    @pytest.mark.parametrize("ops_name", ["typed_creator", "typed_person", "typed_reverse"])
    def test_parity_fast_vs_full(self, ops_name):
        g, P = _graph()
        seed = P + 42
        ops = {
            "typed_creator": [n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n()],
            "typed_person": [n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"type": "Person"})],
            "typed_reverse": [n({"type": "Person"}), e_reverse(edge_match={"type": "HAS_CREATOR"}), n({"id": seed})],
        }[ops_name]
        fast = self._run(g, ops, force_full=False)
        full = self._run(g, ops, force_full=True)
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))

    def test_fast_path_engages_on_typed_hop(self, monkeypatch):
        g, P = _graph()
        seed = P + 42
        hits = {"n": 0}
        real = chain_mod._seeded_typed_hop_pandas_cudf

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        monkeypatch.setattr(chain_mod, "_seeded_typed_hop_pandas_cudf", spy)
        g.gfql([n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"type": "Person"})], engine="pandas")
        assert hits["n"] >= 1

    def test_independent_oracle_values(self):
        """Falsifiability (a): the returned Person must be the ACTUAL creator,
        hand-computed from raw frames — not merely fast==full (both could share
        a bug). Pins VALUES, not just self-consistency."""
        g, P = _graph()
        seed = P + 42
        oracle = _oracle_creators(g, seed)
        assert len(oracle) >= 1, "probe seed must have >=1 Person creator"
        res = g.gfql(
            [n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"type": "Person"})],
            engine="pandas",
        )
        got = sorted(pd.DataFrame(res._nodes)["id"].tolist())
        # native chain _nodes = the UNION of all bound path endpoints: the seed
        # Message (n0) AND its Person creators (n2). The oracle is that union.
        expected = sorted(set(oracle) | {seed})
        assert got == expected, f"fast path returned {got}, oracle says {expected}"

    def test_numpy_helper_declines_predicate_and_undirected(self):
        from graphistry.compute.predicates.numeric import gt
        g, P = _graph()
        node, src, dst = "id", "src", "dst"
        # predicate (non-scalar) edge filter -> decline
        assert chain_mod._seeded_typed_hop_pandas_cudf(
            g.materialize_nodes(), n({"id": P + 1}), n(), e_forward(edge_match={"type": gt(0)}),
            src, dst, node, "forward") is None
        # undirected -> decline
        assert chain_mod._seeded_typed_hop_pandas_cudf(
            g.materialize_nodes(), n({"id": P + 1}), n(), e_forward(),
            src, dst, node, "undirected") is None

    @pytest.mark.parametrize("ops_name,reason", [
        ("two_hop", "multi-hop chain, not a single hop"),
        ("node_predicate", "non-scalar (predicate) node filter"),
    ])
    def test_native_declines_and_stays_correct(self, ops_name, reason, monkeypatch):
        from graphistry.compute.predicates.numeric import gt
        g, P = _graph()
        seed = P + 42
        ops = {
            "two_hop": [n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n(), e_reverse(), n()],
            "node_predicate": [n({"id": seed}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"age": gt(0)})],
        }[ops_name]
        hits = {"n": 0}
        real = chain_mod._seeded_typed_hop_pandas_cudf

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        monkeypatch.setattr(chain_mod, "_seeded_typed_hop_pandas_cudf", spy)
        fast = g.gfql(ops, engine="pandas")
        monkeypatch.undo()
        chain_mod._try_chain_fast_path, saved = (lambda *a, **k: None), chain_mod._try_chain_fast_path
        try:
            full = g.gfql(ops, engine="pandas")
        finally:
            chain_mod._try_chain_fast_path = saved
        assert hits["n"] == 0, f"numpy fast path must decline {reason}"
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))


# ---------------------------------------------------------------------------
# cypher surface  (MATCH (m {id})-[:T]->(p) RETURN p)
# ---------------------------------------------------------------------------

class TestCypherSeededTypedHop:
    def _run(self, g, cy, force_full):
        real = gfql_unified._execute_seeded_typed_hop_fast_path
        try:
            if force_full:
                gfql_unified._execute_seeded_typed_hop_fast_path = lambda *a, **k: None
            return g.gfql(cy, engine="pandas")
        finally:
            gfql_unified._execute_seeded_typed_hop_fast_path = real

    @pytest.mark.parametrize("label", ["p:Person", "p"])
    def test_parity_return_p(self, label):
        g, P = _graph()
        seed = P + 42
        cy = f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->({label}) RETURN p"
        fast = self._run(g, cy, force_full=False)
        full = self._run(g, cy, force_full=True)
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))

    def test_parity_no_match(self):
        g, P = _graph()
        cy = "MATCH (m:Message {id: 999999})-[:HAS_CREATOR]->(p:Person) RETURN p"
        fast = self._run(g, cy, force_full=False)
        full = self._run(g, cy, force_full=True)
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))

    def test_independent_oracle_values(self):
        """Falsifiability (a) on the cypher surface: the p-rows returned by the
        lowered MATCH ... RETURN p must be exactly the hand-computed creators."""
        g, P = _graph()
        seed = P + 42
        oracle = _oracle_creators(g, seed)
        assert len(oracle) >= 1
        res = g.gfql(
            f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->(p:Person) RETURN p",
            engine="pandas",
        )
        df = pd.DataFrame(res._nodes)
        # RETURN p prefixes projected columns with the alias; find the id column.
        id_col = "p.id" if "p.id" in df.columns else "id"
        got = sorted(df[id_col].tolist())
        assert got == oracle, f"cypher fast path returned {got}, oracle says {oracle}"

    def test_fast_path_engages_on_seeded_return(self, monkeypatch):
        g, P = _graph()
        seed = P + 42
        hits = {"n": 0}
        real = gfql_unified._execute_seeded_typed_hop_fast_path

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        monkeypatch.setattr(gfql_unified, "_execute_seeded_typed_hop_fast_path", spy)
        g.gfql(f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->(p:Person) RETURN p", engine="pandas")
        assert hits["n"] >= 1

    @pytest.mark.parametrize("cy_tmpl,reason", [
        ("MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR]->(p:Person) RETURN m, p", "multi-alias"),
        ("MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR]->(p:Person) RETURN p.age", "field projection"),
        ("MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR]->(p:Person) RETURN m", "return source"),
        ("MATCH (p:Person)<-[:HAS_CREATOR]-(m:Message {{id: {s}}}) RETURN p", "reverse (seed on return node)"),
        # variable-length edges are one ASTEdge but multiple hops — must decline or
        # the 1-hop reduction silently truncates (regressed polars varlen parity).
        ("MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR*1..2]->(p) RETURN p", "varlen range hop"),
        ("MATCH (m:Message {{id: {s}}})-[*1..2]->(p) RETURN p", "varlen untyped hop"),
    ])
    def test_declines_and_stays_correct(self, cy_tmpl, reason):
        g, P = _graph()
        seed = P + 42
        cy = cy_tmpl.format(s=seed)
        real = gfql_unified._execute_seeded_typed_hop_fast_path
        hits = {"n": 0}

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        gfql_unified._execute_seeded_typed_hop_fast_path = spy
        try:
            fast = g.gfql(cy, engine="pandas")
        finally:
            gfql_unified._execute_seeded_typed_hop_fast_path = real
        full = self._run(g, cy, force_full=True)
        assert hits["n"] == 0, f"fast path must decline {reason}"
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))


# ---------------------------------------------------------------------------
# Differential fast-vs-full sweep across shapes × engines
# ---------------------------------------------------------------------------

class TestSeededFastPathDifferentialSweep:
    """Broad fast-vs-full byte-parity across a matrix of seeded shapes × engines.

    Answers the coverage question: is every ENGAGED shape verified through BOTH
    the fast path and the full path, on every supported engine (pandas + cuDF)?
    This complements the ~1114 implicit DECLINE-path exercises the cypher
    conformance suite already drives through `_execute_seeded_typed_hop_fast_path`
    (real queries that fall through and still return correct values). Here we pin
    the ENGAGED shapes byte-identical fast-vs-full so the specialization can never
    silently diverge from the general path on any covered engine."""

    SHAPES = {
        "native_typed": lambda s: [n({"id": s}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"type": "Person"})],
        "native_untyped": lambda s: [n({"id": s}), e_forward(), n()],
        "native_reverse": lambda s: [n({"type": "Person"}), e_reverse(edge_match={"type": "HAS_CREATOR"}), n({"id": s})],
        "native_type_src": lambda s: [n({"type": "Message"}), e_forward(edge_match={"type": "HAS_CREATOR"}), n({"type": "Person"})],
        "cypher_p_labelled": lambda s: f"MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR]->(p:Person) RETURN p",
        "cypher_p_unlabelled": lambda s: f"MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR]->(p) RETURN p",
        "cypher_no_match": lambda s: "MATCH (m:Message {id: 9999999})-[:HAS_CREATOR]->(p:Person) RETURN p",
    }

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_fast_vs_full_parity(self, engine, shape):
        g, P = _engine_graph(engine)
        query = self.SHAPES[shape](P + 42)
        fast = _run_diff(g, engine, query, fast=True)
        full = _run_diff(g, engine, query, fast=False)
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))

    @pytest.mark.parametrize("engine", ["pandas", "cudf"])
    def test_engaged_shapes_match_independent_oracle(self, engine):
        """The two headline engaged shapes must equal the hand-computed creator set
        (native = {seed} ∪ creators; cypher RETURN p = creators) on each engine —
        not merely fast==full, since the full path could share a bug."""
        g, P = _engine_graph(engine)
        pd_g, _ = _graph()  # pandas twin for the independent oracle
        seed = P + 42
        oracle = _oracle_creators(pd_g, seed)
        assert len(oracle) >= 1
        native = _run_diff(g, engine, self.SHAPES["native_typed"](seed), fast=True)
        got_native = sorted(_canon_nodes(native)["id"].tolist())
        assert got_native == sorted(set(oracle) | {seed})
        cyp = _run_diff(g, engine, self.SHAPES["cypher_p_labelled"](seed), fast=True)
        cdf = _canon_nodes(cyp)
        idc = "p.id" if "p.id" in cdf.columns else "id"
        assert sorted(cdf[idc].tolist()) == oracle


# ---------------------------------------------------------------------------
# side-channel decline gates + semantics parity (review-skill blockers, wave 2026-07-21)
# ---------------------------------------------------------------------------

class TestFastPathSideChannelGates:
    """Each test pins a shape where ENGAGING the fast path was empirically proven
    to diverge from the full path (probes in plans/review-pr-1759/): the fix is
    to decline (or match semantics exactly), and these lock that in."""

    def _canon_edges(self, res):
        edges = res._edges
        df = edges.to_pandas() if hasattr(edges, "to_pandas") else pd.DataFrame(edges)
        cols = sorted(str(c) for c in df.columns)
        df.columns = [str(c) for c in df.columns]
        return df.sort_values(cols).reset_index(drop=True)[cols] if cols else df

    def test_labels_column_precedence(self):
        """label__X must follow resolve_filter_column: `labels` (list containment)
        wins over `type` — the fast path's type-equality shortcut must decline."""
        ndf = pd.DataFrame({
            "id": [0, 1],
            "type": ["Person", "Person"],
            "labels": [["Admin"], ["Person"]],  # containment says only id=1 is a Person
        })
        edf = pd.DataFrame({"src": [0], "dst": [1], "type": ["KNOWS"]})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = "MATCH (m {id: 0})-[{type:'KNOWS'}]->(p:Person) RETURN p"
        fast = _canon_nodes(_run_diff(g, "pandas", q, fast=True))
        full = _canon_nodes(_run_diff(g, "pandas", q, fast=False))
        pd.testing.assert_frame_equal(fast, full)

    def test_null_ids_never_link(self):
        """NaN node id + NaN edge endpoint: pandas .isin matches NaN<->NaN but the
        full pipeline's joins never link null keys — nodes AND edges must agree."""
        ndf = pd.DataFrame({"id": [0.0, 1.0, np.nan],
                            "type": ["Person", "Message", "Message"]})
        edf = pd.DataFrame({"src": [1.0, np.nan], "dst": [0.0, 0.0],
                            "type": ["HAS_CREATOR", "HAS_CREATOR"]})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        ops = [n({"type": "Message"}), e_forward(edge_match={"type": "HAS_CREATOR"}),
               n({"type": "Person"})]
        fast_r = _run_diff(g, "pandas", ops, fast=True)
        full_r = _run_diff(g, "pandas", ops, fast=False)
        pd.testing.assert_frame_equal(_canon_nodes(fast_r), _canon_nodes(full_r))
        pd.testing.assert_frame_equal(self._canon_edges(fast_r), self._canon_edges(full_r))

    def test_policy_hooks_not_skipped(self):
        """A policy dict must force the full path so pre/post hooks fire."""
        g, P = _graph()
        seed = P + 7
        fired = []

        def hook(ctx):
            fired.append(ctx.get("phase", "?"))
            return None
        q = f"MATCH (m {{id: {seed}}})-[{{type:'HAS_CREATOR'}}]->(p {{type:'Person'}}) RETURN p"
        policy = {"prechain": hook, "postchain": hook}
        r_pol = g.gfql(q, engine="pandas", policy=policy)
        assert fired, "policy hooks must fire (fast path must decline under policy)"
        r_nopol = g.gfql(q, engine="pandas")
        pd.testing.assert_frame_equal(_canon_nodes(r_pol), _canon_nodes(r_nopol))

    def test_same_path_where_not_dropped(self):
        """WHERE p.id < m.id (same-path cross-alias) must not be silently dropped."""
        ndf = pd.DataFrame({"id": [0, 1, 2], "type": ["Message", "Person", "Person"],
                            "age": [np.nan, 31.0, 32.0]})
        edf = pd.DataFrame({"src": [0, 0], "dst": [1, 2], "type": ["T", "T"]})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = "MATCH (m {id: 0})-[{type:'T'}]->(p {type:'Person'}) WHERE p.id > m.id + 1 RETURN p"
        fast = _canon_nodes(_run_diff(g, "pandas", q, fast=True))
        full = _canon_nodes(_run_diff(g, "pandas", q, fast=False))
        pd.testing.assert_frame_equal(fast, full)

    def test_optional_match_null_row(self):
        """No-match OPTIONAL MATCH returns the openCypher null row, not empty."""
        ndf = pd.DataFrame({"id": [0, 1], "type": ["Message", "Person"]})
        edf = pd.DataFrame({"src": [0], "dst": [1], "type": ["OTHER"]})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = "OPTIONAL MATCH (m {id: 0})-[{type:'MISSING'}]->(p {type:'Person'}) RETURN p"
        fast = _canon_nodes(_run_diff(g, "pandas", q, fast=True))
        full = _canon_nodes(_run_diff(g, "pandas", q, fast=False))
        pd.testing.assert_frame_equal(fast, full)

    def test_with_match_reentry_carried_seeds(self):
        """WITH..MATCH reentry hands carried seeds via start_nodes; the fast path
        deriving its seed from the filter alone would silently widen the seed set.
        Node 0 is excluded by the WITH but matches the reentry filter and has a
        LIKES edge -> divergence unless the fast path declines."""
        ndf = pd.DataFrame({"id": [0, 1, 2, 10, 11, 13],
                            "kind": ["person"] * 3 + ["post"] * 3})
        edf = pd.DataFrame({"src": [0, 0, 1, 2, 0], "dst": [1, 2, 10, 11, 13],
                            "type": ["KNOWS", "KNOWS", "LIKES", "LIKES", "LIKES"]})
        g = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = ("MATCH (p {id:0})-[{type:'KNOWS'}]->(a) WITH a "
             "MATCH (a {kind:'person'})-[{type:'LIKES'}]->(b) RETURN b")
        fast = _canon_nodes(_run_diff(g, "pandas", q, fast=True))
        full = _canon_nodes(_run_diff(g, "pandas", q, fast=False))
        pd.testing.assert_frame_equal(fast, full)
        assert sorted(fast["b.id"].tolist()) == [10, 11]


# ---------------------------------------------------------------------------
# polars cypher surface  (#1755 generalization: MATCH (m {id})-[:T]->(p) RETURN p)
# ---------------------------------------------------------------------------

class TestCypherSeededTypedHopPolars:
    """The seeded cypher fast path also covers polars via _seeded_typed_return_dst_polars
    (dispatched from _execute_seeded_typed_hop_fast_path for Engine.POLARS/POLARS_GPU).
    Same differential-parity + independent-oracle + decline bar as the pandas class."""

    def _pl_graph(self):
        pl = pytest.importorskip("polars")
        g, P = _graph()
        gp = graphistry.nodes(pl.from_pandas(pd.DataFrame(g._nodes)), "id").edges(
            pl.from_pandas(pd.DataFrame(g._edges)), "src", "dst")
        return gp, P, g  # g (pandas) reused for the independent oracle

    def _run(self, g, cy, force_full):
        real = gfql_unified._execute_seeded_typed_hop_fast_path
        try:
            if force_full:
                gfql_unified._execute_seeded_typed_hop_fast_path = lambda *a, **k: None
            return g.gfql(cy, engine="polars")
        finally:
            gfql_unified._execute_seeded_typed_hop_fast_path = real

    @pytest.mark.parametrize("label", ["p:Person", "p"])
    def test_parity_return_p(self, label):
        gp, P, _ = self._pl_graph()
        seed = P + 42
        cy = f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->({label}) RETURN p"
        fast = self._run(gp, cy, force_full=False)
        full = self._run(gp, cy, force_full=True)
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))

    def test_independent_oracle_values(self):
        gp, P, g_pd = self._pl_graph()
        seed = P + 42
        oracle = _oracle_creators(g_pd, seed)
        assert len(oracle) >= 1
        res = self._run(gp, f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->(p:Person) RETURN p", force_full=False)
        df = _canon_nodes(res)
        id_col = "p.id" if "p.id" in df.columns else "id"
        assert sorted(df[id_col].tolist()) == oracle

    def test_fast_path_engages(self, monkeypatch):
        gp, P, _ = self._pl_graph()
        seed = P + 42
        hits = {"n": 0}
        real = gfql_unified._execute_seeded_typed_hop_fast_path

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        monkeypatch.setattr(gfql_unified, "_execute_seeded_typed_hop_fast_path", spy)
        gp.gfql(f"MATCH (m:Message {{id: {seed}}})-[:HAS_CREATOR]->(p:Person) RETURN p", engine="polars")
        assert hits["n"] >= 1

    # Only shapes the full polars path can itself render (polars declines
    # multi-entity/field whole-row RETURN by design — those aren't a parity
    # target). Variable-length is the shape the fast path must decline (the bug
    # this generalization inherited from the pandas path).
    @pytest.mark.parametrize("cy_tmpl,reason", [
        ("MATCH (m:Message {{id: {s}}})-[:HAS_CREATOR*1..2]->(p) RETURN p", "varlen range hop"),
        ("MATCH (m:Message {{id: {s}}})-[*1..2]->(p) RETURN p", "varlen untyped hop"),
    ])
    def test_declines_and_stays_correct(self, cy_tmpl, reason):
        gp, P, _ = self._pl_graph()
        seed = P + 42
        cy = cy_tmpl.format(s=seed)
        real = gfql_unified._execute_seeded_typed_hop_fast_path
        hits = {"n": 0}

        def spy(*a, **k):
            r = real(*a, **k)
            if r is not None:
                hits["n"] += 1
            return r

        gfql_unified._execute_seeded_typed_hop_fast_path = spy
        try:
            fast = gp.gfql(cy, engine="polars")
        finally:
            gfql_unified._execute_seeded_typed_hop_fast_path = real
        full = self._run(gp, cy, force_full=True)
        assert hits["n"] == 0, f"polars fast path must decline {reason}"
        pd.testing.assert_frame_equal(_canon_nodes(fast), _canon_nodes(full))


# ---------------------------------------------------------------------------
# polars-specific decline gates (review-skill wave 2026-07-21, #1760)
# ---------------------------------------------------------------------------

class TestPolarsFastPathGates:
    def _q(self, seed):
        return f"MATCH (m {{id: {seed}}})-[{{type:'HAS_CREATOR'}}]->(p {{type:'Person'}}) RETURN p"

    def test_lazyframe_declines_not_crashes(self):
        """A LazyFrame-backed graph must decline (full path), not AttributeError."""
        pl = pytest.importorskip("polars")
        from graphistry.compute.chain_fast_paths import _seeded_typed_return_dst_polars
        from graphistry.compute.ast import ASTNode, ASTEdge
        g, P = _graph()
        lazy_g = graphistry.nodes(
            pl.from_pandas(pd.DataFrame(g._nodes)).lazy(), "id"
        ).edges(pl.from_pandas(pd.DataFrame(g._edges)).lazy(), "src", "dst")
        r = _seeded_typed_return_dst_polars(
            lazy_g, n({"id": P + 1}), n({"type": "Person"}),
            e_forward(edge_match={"type": "HAS_CREATOR"}),
            "src", "dst", "id", "forward")
        assert r is None

    def test_mixed_engine_frames_decline(self):
        """polars nodes + pandas edges (or vice versa) must decline at dispatch."""
        pl = pytest.importorskip("polars")
        g, P = _graph()
        mixed = graphistry.nodes(
            pl.from_pandas(pd.DataFrame(g._nodes)), "id"
        ).edges(pd.DataFrame(g._edges), "src", "dst")
        hits = {"n": 0}
        real = gfql_unified._execute_seeded_typed_hop_fast_path

        def spy(*a, **k):
            r = real(*a, **k)
            hits["n"] += 1 if r is not None else 0
            return r
        gfql_unified._execute_seeded_typed_hop_fast_path = spy
        try:
            mixed.gfql(self._q(P + 1), engine="polars")
        except Exception:
            pass  # full path may or may not support mixed frames; no crash IN the fast path
        finally:
            gfql_unified._execute_seeded_typed_hop_fast_path = real
        assert hits["n"] == 0, "fast path must not engage on mixed-engine frames"

    def test_polars_labels_column_precedence(self):
        """label__X on a polars graph with BOTH labels+type columns: decline to the
        full path (labels list-containment wins over type equality)."""
        pl = pytest.importorskip("polars")
        ndf = pd.DataFrame({
            "id": [0, 1], "type": ["Person", "Person"],
            "labels": [["Admin"], ["Person"]],
        })
        edf = pd.DataFrame({"src": [0], "dst": [1], "type": ["KNOWS"]})
        gp = graphistry.nodes(pl.from_pandas(ndf), "id").edges(pl.from_pandas(edf), "src", "dst")
        gpd = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = "MATCH (m {id: 0})-[{type:'KNOWS'}]->(p:Person) RETURN p"
        got = _canon_nodes(_run_diff(gp, "polars", q, fast=True))
        oracle = _canon_nodes(_run_diff(gpd, "pandas", q, fast=False))
        assert got["p.id"].tolist() == oracle["p.id"].tolist()

    def test_polars_null_ids_never_link(self):
        """Null seed/node ids must not link via is_in membership on polars."""
        pl = pytest.importorskip("polars")
        ndf = pl.DataFrame({"id": [0, 1, None], "kind": ["person", "post", "post"]})
        edf = pl.DataFrame({"src": [0, None], "dst": [1, 1], "type": ["T", "T"]})
        gp = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
        q = "MATCH (m {kind:'person'})-[{type:'T'}]->(p {kind:'post'}) RETURN p"
        fast = _canon_nodes(_run_diff(gp, "polars", q, fast=True))
        full = _canon_nodes(_run_diff(gp, "polars", q, fast=False))
        pd.testing.assert_frame_equal(fast, full)
        assert fast["p.id"].tolist() == [1]
