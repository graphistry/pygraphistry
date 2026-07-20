"""Differential parity: native polars hop() == pandas hop() (Phase 1 core BFS; pandas = oracle;
identical node-id and edge sets). See plans/gfql-polars-engine."""
import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")

from graphistry.tests.compute.gfql.polars_test_utils import (  # noqa: E402
    node_id_set as _node_set,
    edge_pair_set as _edge_set,
    node_attr_map as _node_attrs_hop,
)


GRAPHS = {
    "line5": pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "e"]}),
    "cycle4": pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "a"]}),
    "branch": pd.DataFrame({"s": ["a", "a", "b", "c", "x"], "d": ["b", "c", "d", "d", "y"]}),
}

CASES = [
    dict(hops=1, direction="forward"),
    dict(hops=2, direction="forward"),
    dict(hops=3, direction="forward"),
    dict(hops=1, direction="reverse"),
    dict(hops=2, direction="reverse"),
    dict(hops=1, direction="undirected"),
    dict(hops=2, direction="undirected"),
    dict(to_fixed_point=True, direction="forward"),
    dict(to_fixed_point=True, direction="undirected"),
    dict(hops=1, direction="forward", return_as_wave_front=True),
    dict(hops=2, direction="forward", return_as_wave_front=True),
]

SEEDS = [None, ["a"], ["a", "c"], ["z"]]  # z = absent


@pytest.mark.parametrize("gname", list(GRAPHS))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_polars_hop_parity(gname, case, seed):
    base = graphistry.edges(GRAPHS[gname], "s", "d").materialize_nodes()
    nodes = None if seed is None else pd.DataFrame({base._node: seed})

    gp = base.hop(nodes=nodes, engine="pandas", **case)
    gl = base.hop(nodes=nodes, engine="polars", **case)

    assert "polars" in type(gl._nodes).__module__
    assert _node_set(gp) == _node_set(gl), f"node mismatch {gname} {case} seed={seed}"
    assert _edge_set(gp) == _edge_set(gl), f"edge mismatch {gname} {case} seed={seed}"


@pytest.mark.parametrize("kw", [
    # label_node_hops / label_seeds are NATIVE on the plain BFS as of #1741 — see
    # TestHopLabelsDifferential; only label_edge_hops and min_hops>1 labeling still decline.
    {"label_edge_hops": "h"},
    {"min_hops": 2},   # min_hops>1 is native in chain()/gfql() only; a DIRECT hop() stays NIE
    {"output_min_hops": 1},
    {"output_max_hops": 2},
    {"source_node_query": "s == 'a'"},
    {"destination_node_query": "s == 'a'"},
    {"edge_query": "s == 'a'"},
    {"include_zero_hop_seed": True},
])
def test_polars_hop_unsupported_raises(kw):
    base = graphistry.edges(GRAPHS["line5"], "s", "d").materialize_nodes()
    with pytest.raises(NotImplementedError):
        base.hop(hops=1, engine="polars", **kw)


def test_polars_hop_min_hops_1_allowed():
    # min_hops=1 is the default and must NOT raise (boundary guard)
    base = graphistry.edges(GRAPHS["line5"], "s", "d").materialize_nodes()
    base.hop(hops=1, min_hops=1, engine="polars")


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("seed_sel", [["a"], ["c"], ["a", "d"]])
def test_polars_hop_min_hops_labeled_policy_unit_parity(direction, seed_sel):
    # CHAIN-CONTEXT min_hops node-policy unit test: drive the internal polars hop with
    # min_hops_label_policy=True; compare the FULL node frame (non-id attrs, null-aware) to pandas'
    # LABELED hop (label_node_hops=..., the track_node_hops path ASTEdge.execute takes) — asserts
    # the null-attr/seed-strip/endpoint contract (seed-48 class) without a chain. Direct
    # base.hop(min_hops>1) stays NIE (test_polars_hop_unsupported_raises); only chain wires this.
    from graphistry.compute.gfql.lazy.engine.polars.hop_eager import hop_polars
    from graphistry.Engine import Engine, df_to_engine
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
                          "kind": ["x", "y", "y", "z", "x"], "score": [1, 2, 3, 4, 5]})
    edges = pd.DataFrame({"s": ["a", "b", "c", "d", "a", "c"], "d": ["b", "c", "d", "e", "c", "a"]})
    base = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    sel = base._nodes[base._nodes["id"].isin(seed_sel)]
    gp = base.hop(nodes=sel, min_hops=2, max_hops=3, direction=direction, return_as_wave_front=True,
                  label_node_hops="__gfql_output_node_hop__", label_edge_hops="__gfql_output_edge_hop__",
                  engine="pandas")
    gpol = base.nodes(df_to_engine(base._nodes, Engine.POLARS), "id").edges(
        df_to_engine(base._edges, Engine.POLARS), "s", "d")
    gl = hop_polars(gpol, nodes=df_to_engine(sel, Engine.POLARS), min_hops=2, max_hops=3,
                    direction=direction, return_as_wave_front=True, min_hops_label_policy=True)
    assert _node_set(gp) == _node_set(gl)
    assert _node_attrs_hop(gp) == _node_attrs_hop(gl), f"labeled-policy attr divergence {direction} {seed_sel}"


def test_polars_hop_edges_only_parity():
    g = graphistry.edges(pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]}), "s", "d")
    gp = g.hop(hops=1, engine="pandas")
    gl = g.hop(hops=1, engine="polars")
    assert _node_set(gp) == _node_set(gl)
    assert _edge_set(gp) == _edge_set(gl)


def test_polars_hop_dtype_mismatch_parity():
    # float node ids (with a null) vs int edge endpoints — pandas coerces; polars must align
    # join-key dtypes rather than crash
    nodes = pd.DataFrame({"id": [0.0, 1.0, 2.0, None], "k": ["x", "y", "z", "x"]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    gp = g.hop(hops=1, engine="pandas")
    gl = g.hop(hops=1, engine="polars")
    assert _node_set(gp) == _node_set(gl)
    assert _edge_set(gp) == _edge_set(gl)


def test_polars_hop_predicate_matches_parity():
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
    edges = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "rel": ["r1", "r2", "r1"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    from graphistry import gt
    # all-nodes hop (no custom seed) so node-match resolves against the full table on both engines
    for kw in (
        {"edge_match": {"rel": "r1"}},
        {"source_node_match": {"score": gt(15)}},
        {"destination_node_match": {"score": gt(15)}},
    ):
        gp = g.hop(hops=1, engine="pandas", **kw)
        gl = g.hop(hops=1, engine="polars", **kw)
        assert _node_set(gp) == _node_set(gl), kw
        assert _edge_set(gp) == _edge_set(gl), kw


def test_polars_filter_by_dict_exotic_predicate_declines():
    # NO-CHEATING: an exotic predicate with no native lowering must NIE, NOT silently evaluate
    # via pandas (the old bridge misrepresented pandas semantics as polars)
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars, predicate_to_expr

    class IsOdd(ASTPredicate):
        def __call__(self, s):
            return (s % 2) == 1

    df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "v": [1, 2, 3, 4, 5]})
    assert predicate_to_expr("v", IsOdd()) is None  # not lowered
    with pytest.raises(NotImplementedError):
        filter_by_dict_polars(df, {"v": IsOdd()})


class TestHopLabelsDifferential:
    """#1741 — native polars ``label_node_hops`` on the plain (shortest-path) BFS, pandas as oracle.

    pandas' labeling rule is direction-dependent and that asymmetry is the whole reason the
    polars chain used to alias a backtracked-to seed that pandas leaves unaliased:
      * forward/reverse — EVERY destination of a hop is labeled, first-wins, so a seed re-entered
        at hop 1 IS labeled 1;
      * undirected — destinations MINUS everything already visited, so a seed re-reached by
        walking back along the edge it arrived on stays NULL.
    """

    LABEL_GRAPHS = {
        # p0 -> p1 -> p2, plus p1 -> p0: undirected hop 2 backtracks into the seed.
        "backtrack": pd.DataFrame({"s": ["p0", "p1", "p2", "p1"], "d": ["p1", "p2", "p4", "p0"]}),
        "line5": GRAPHS["line5"],
        "cycle4": GRAPHS["cycle4"],
        "branch": GRAPHS["branch"],
    }

    @pytest.mark.parametrize("graph_name", sorted(LABEL_GRAPHS))
    @pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
    @pytest.mark.parametrize("hops", [1, 2, 3])
    @pytest.mark.parametrize("label_seeds", [False, True])
    def test_node_hop_labels_match_pandas(self, graph_name, direction, hops, label_seeds):
        edf = self.LABEL_GRAPHS[graph_name]
        seed_id = edf["s"].iloc[0]
        g_pd = graphistry.edges(edf, "s", "d").materialize_nodes()
        seeds_pd = g_pd._nodes[g_pd._nodes["id"] == seed_id]
        expected = g_pd.hop(
            nodes=seeds_pd, direction=direction, hops=hops,
            label_node_hops="_h", label_seeds=label_seeds,
        )._nodes[["id", "_h"]].sort_values("id").reset_index(drop=True)

        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        actual = g_pl.hop(
            nodes=pl.from_pandas(seeds_pd[["id"]]), direction=direction, hops=hops,
            label_node_hops="_h", label_seeds=label_seeds, engine="polars",
        )._nodes.select(["id", "_h"]).sort("id").to_pandas()

        assert list(actual["id"]) == list(expected["id"])
        assert (
            actual["_h"].astype("float").fillna(-1).tolist()
            == expected["_h"].astype("float").fillna(-1).tolist()
        )

    def test_undirected_backtracked_seed_stays_unlabeled(self):
        """The #1741 shape, pinned explicitly: p0 is re-reached at hop 2 yet keeps a NULL label."""
        edf = self.LABEL_GRAPHS["backtrack"]
        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        out = g_pl.hop(
            nodes=pl.DataFrame({"id": ["p0"]}), direction="undirected", hops=2,
            label_node_hops="_h", engine="polars",
        )._nodes.sort("id")
        assert dict(zip(out["id"].to_list(), out["_h"].to_list())) == {"p0": None, "p1": 1, "p2": 2}

    def test_forward_reentered_seed_is_labeled(self):
        """Same seed re-entry, forward: pandas DOES label it, so polars must too."""
        edf = pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"]})
        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        out = g_pl.hop(
            nodes=pl.DataFrame({"id": ["a"]}), direction="forward", hops=2,
            label_node_hops="_h", engine="polars",
        )._nodes.sort("id")
        assert dict(zip(out["id"].to_list(), out["_h"].to_list())) == {"a": 2, "b": 1}

    def test_label_seeds_writes_hop_zero(self):
        edf = self.LABEL_GRAPHS["line5"]
        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        out = g_pl.hop(
            nodes=pl.DataFrame({"id": ["a"]}), direction="forward", hops=2,
            label_node_hops="_h", label_seeds=True, engine="polars",
        )._nodes.sort("id")
        assert dict(zip(out["id"].to_list(), out["_h"].to_list())) == {"a": 0, "b": 1, "c": 2}

    def test_edge_hop_labels_still_decline(self):
        """label_edge_hops stays an honest NIE: with labels on, pandas DUPLICATES an undirected
        edge traversed in both directions, and reproducing that artifact is not parity worth
        having. Node labels are what #1741 needs."""
        edf = self.LABEL_GRAPHS["line5"]
        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        with pytest.raises(NotImplementedError, match="label_edge_hops"):
            g_pl.hop(nodes=pl.DataFrame({"id": ["a"]}), direction="forward", hops=2,
                     label_edge_hops="_eh", engine="polars")

    def test_min_hops_above_one_still_declines_labels(self):
        edf = self.LABEL_GRAPHS["line5"]
        g_pl = graphistry.edges(pl.from_pandas(edf), "s", "d").materialize_nodes(engine="polars")
        with pytest.raises(NotImplementedError, match="label_node_hops"):
            g_pl.hop(nodes=pl.DataFrame({"id": ["a"]}), direction="forward",
                     min_hops=2, max_hops=3, label_node_hops="_h", engine="polars")

    # --- Amplified axes (#1746 review): these combinations regressed before the undirected
    # seed pre-seed fix — the node/edge SET matched pandas but the LABELS diverged, because a
    # seed filtered out of the frontier (source_node_match / return_as_wave_front) was not
    # counted as "seen" and got re-labeled. The chain drives the polars hop in wavefront mode and
    # keys its alias gate off these labels, so a wrong label is a wrong query result.
    SEED_FILTER_GRAPHS = {
        "tri": (pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "a"]}),
                pd.DataFrame({"id": ["a", "b", "c"], "k": [0, 1, 1]}), ["a"]),
        "selfloop2": (pd.DataFrame({"s": ["n0", "n1", "n4"], "d": ["n1", "n0", "n4"]}),
                      pd.DataFrame({"id": ["n0", "n1", "n4"], "k": [0, 0, 1]}), ["n0", "n4"]),
    }

    @pytest.mark.parametrize("graph_name", sorted(SEED_FILTER_GRAPHS))
    @pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
    @pytest.mark.parametrize("hops", [1, 2])
    @pytest.mark.parametrize("rwf", [True, False])
    @pytest.mark.parametrize("snm", [None, {"k": 0}])
    @pytest.mark.parametrize("dnm", [None, {"k": 1}])
    def test_node_hop_labels_match_pandas_with_seed_filters(
            self, graph_name, direction, hops, rwf, snm, dnm):
        edf, ndf, seeds = self.SEED_FILTER_GRAPHS[graph_name]

        g_pd = graphistry.nodes(ndf, "id").edges(edf, "s", "d")
        seeds_pd = g_pd._nodes[g_pd._nodes["id"].isin(seeds)]
        exp = g_pd.hop(
            nodes=seeds_pd, direction=direction, hops=hops, return_as_wave_front=rwf,
            label_node_hops="_h", source_node_match=snm, destination_node_match=dnm,
            engine="pandas",
        )._nodes[["id", "_h"]].sort_values("id").reset_index(drop=True)

        g_pl = graphistry.nodes(pl.from_pandas(ndf), "id").edges(pl.from_pandas(edf), "s", "d")
        act = g_pl.hop(
            # full-column seed (carries "k") so source_node_match can filter it at hops=1, exactly
            # as the pandas oracle above receives seeds_pd with its columns.
            nodes=pl.from_pandas(seeds_pd), direction=direction, hops=hops,
            return_as_wave_front=rwf, label_node_hops="_h",
            source_node_match=snm, destination_node_match=dnm, engine="polars",
        )._nodes.select(["id", "_h"]).sort("id").to_pandas()

        assert list(act["id"]) == list(exp["id"])
        assert (act["_h"].astype("float").fillna(-1).tolist()
                == exp["_h"].astype("float").fillna(-1).tolist())

    def test_requested_label_column_collision_redirects_like_pandas(self):
        """#1746 review IMPORTANT-1: a label_node_hops name that already exists on the node table
        must be redirected to `<name>_1` (pandas resolve_label_col), not clobbered/`_right`-ed."""
        edf = self.LABEL_GRAPHS["line5"]
        ndf = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "_h": ["u0", "u1", "u2", "u3", "u4"]})
        g_pd = graphistry.nodes(ndf, "id").edges(edf, "s", "d")
        exp = g_pd.hop(nodes=g_pd._nodes[g_pd._nodes["id"] == "a"], direction="forward", hops=2,
                       label_node_hops="_h", engine="pandas")._nodes
        g_pl = graphistry.nodes(pl.from_pandas(ndf), "id").edges(pl.from_pandas(edf), "s", "d")
        act = g_pl.hop(nodes=pl.DataFrame({"id": ["a"]}), direction="forward", hops=2,
                       label_node_hops="_h", engine="polars")._nodes
        # pandas redirected the label to a non-clobbering name and kept the user's "_h" strings.
        assert "_h_1" in exp.columns and "_h_1" in act.columns
        assert set(act["_h"].to_list()) <= {"u0", "u1", "u2", "u3", "u4"}
