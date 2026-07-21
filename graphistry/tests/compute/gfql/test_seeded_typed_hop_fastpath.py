"""Parity + gating tests for the #1755 seeded typed-hop fast path.

Two layers close the seeded-Cypher abstraction tax on pandas:
  * native  — `_try_chain_fast_path` / `_seeded_typed_hop_pandas_cudf` accelerate
    a seeded typed 1-hop chain [n({id}), e(edge_match), n({type})];
  * cypher   — `_execute_seeded_typed_hop_fast_path` accelerates the lowered
    MATCH (m {id})-[:T]->(p) RETURN p string surface.

Both are byte-identical to the full path by construction; these tests pin that
(fast-on vs fast-off, differential) and that the fast path actually ENGAGES for
the accelerated shapes and DECLINES (falls through) for everything else.
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
