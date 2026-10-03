"""A native op-list seeded on a membership set takes the resident index (#2116 item 3a).

``[n({"id": is_in(seeds)}), e_forward(), n()]`` returned the right rows but ran the isin
scan with ``gfql_explain`` recording nothing (used_index False, decision_code None) while the
scalar seed ``n({"id": 5})`` and the Cypher twin (#2117) took the index. The native lanes now
resolve a membership set on the node-id key the way the indexed bindings kernel does
(``_membership_seed_ids``), look the ids up in the node-id index, and re-apply the canonical
filter on the hits.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry import e_forward, n
from graphistry.compute.chain_fast_paths import _seeded_seed_filters
from graphistry.compute.predicates.is_in import is_in

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover - depends on test env
    _HAS_CUDF = False

_ENGINES = ["pandas"] + (["cudf"] if _HAS_CUDF else [])


def _graph(engine="pandas"):
    rng = np.random.default_rng(7)
    n_nodes, n_edges = 20000, 100000
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    nodes = pd.DataFrame({"id": np.arange(n_nodes), "kind": rng.choice(["a", "b"], n_nodes)})
    seeds = sorted(rng.choice(n_nodes, 50, replace=False).tolist())
    frames = (edges, nodes)
    if engine == "cudf":  # the indexes are built for the frames they will serve
        import cudf
        frames = (cudf.from_pandas(edges), cudf.from_pandas(nodes))
    g = graphistry.edges(frames[0], "src", "dst").nodes(frames[1], "id").gfql_index_all()
    return g, edges, nodes, seeds


def _explain(g, ops, engine, policy):
    d = g.gfql_explain(ops, engine=engine, index_policy=policy)
    d = d if isinstance(d, dict) else d.__dict__
    return d.get("used_index"), d.get("decision_code"), [s.get("seam") for s in (d.get("steps") or [])]


def _n(frame):
    return int(frame.shape[0])


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("seed_form", ["is_in", "list"])
@pytest.mark.route_engaged("native-fast", "index-hop")
def test_membership_seeded_hop_takes_the_index_and_says_so(engine, seed_form):
    g, edges, nodes, seeds = _graph(engine)
    seed = is_in(seeds) if seed_form == "is_in" else list(seeds)
    ops = [n({"id": seed}), e_forward(), n()]
    truth = int(edges["src"].isin(seeds).sum())
    for policy in ("off", "auto", "use", "force"):
        assert _n(g.gfql(ops, engine=engine, index_policy=policy)._edges) == truth, policy
    assert _explain(g, ops, engine, "off")[:2] == (False, "policy_off")
    for policy in ("auto", "use", "force"):
        used, code, seams = _explain(g, ops, engine, policy)
        assert (used, code) == (True, "index_selected"), policy
        assert "native_seeded_hop" in seams, (policy, seams)


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.route_engaged("native-fast", "index-hop")
def test_residual_filters_are_still_applied_on_the_index_hits(engine):
    g, edges, nodes, seeds = _graph(engine)
    kind = nodes.set_index("id")["kind"]
    a_seeds = [s for s in seeds if kind[s] == "a"]
    ops = [n({"id": is_in(seeds), "kind": "a"}), e_forward(), n({"kind": "b"})]
    truth = int((edges["src"].isin(a_seeds) & edges["dst"].map(kind).eq("b")).sum())
    for policy in ("off", "auto", "force"):
        assert _n(g.gfql(ops, engine=engine, index_policy=policy)._edges) == truth, policy
    assert _explain(g, ops, engine, "force")[:2] == (True, "index_selected")


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.route_engaged("native-fast", "index-hop")
def test_single_node_membership_seed_is_served_by_the_node_id_index(engine):
    g, edges, nodes, seeds = _graph(engine)
    ops = [n({"id": is_in(seeds)})]
    for policy in ("off", "auto", "force"):
        assert _n(g.gfql(ops, engine=engine, index_policy=policy)._nodes) == len(seeds), policy
    used, code, seams = _explain(g, ops, engine, "auto")
    assert (used, code) == (True, "index_selected") and "native_seed_lookup" in seams


def test_non_integral_members_keep_the_scan_semantics():
    g, edges, nodes, seeds = _graph()
    truth = int(edges["src"].isin(seeds).sum())
    mixed = [n({"id": is_in(seeds + ["x"])}), e_forward(), n()]
    assert _n(g.gfql(mixed, engine="pandas", index_policy="force")._edges) == truth
    booled = [n({"id": is_in([True, seeds[0]])}), e_forward(), n()]
    scan = _n(g.gfql(booled, engine="pandas", index_policy="off")._edges)
    assert _n(g.gfql(booled, engine="pandas", index_policy="force")._edges) == scan
    # a membership set on a non-id column is not a seed: the lane declines, rows unchanged
    kinds = [n({"kind": is_in(["a"])}), e_forward(), n()]
    assert _n(g.gfql(kinds, engine="pandas", index_policy="force")._edges) == _n(g.gfql(kinds, engine="pandas", index_policy="off")._edges)


def test_seed_filter_resolver_contract():
    df = pd.DataFrame({"id": [1, 2, 3], "kind": ["a", "b", "a"]})
    assert _seeded_seed_filters({"id": is_in([3, 1, 1])}, df, "id") == {"id": (1, 3)}
    assert _seeded_seed_filters({"id": [2, 3], "kind": "a"}, df, "id") == {"id": (2, 3), "kind": "a"}
    assert _seeded_seed_filters({"id": 2}, df, "id") == {"id": 2}
    assert _seeded_seed_filters({}, df, "id") == {}
    assert _seeded_seed_filters({"id": is_in([1, "x"])}, df, "id") is None      # not all integral
    assert _seeded_seed_filters({"id": is_in([True, 1])}, df, "id") is None      # bool is not an id
    assert _seeded_seed_filters({"kind": is_in(["a"])}, df, "id") is None        # membership off the id key
    assert _seeded_seed_filters({"id": is_in([1])}, df.drop(columns=["id"]), "id") is None  # id column absent


@pytest.mark.route_engaged("native-fast", "index-hop")
def test_a_seed_without_a_usable_index_records_the_decline():
    # the lane's scan branch used to leave explain silent; now it says why the index was not used
    g, edges, nodes, seeds = _graph()
    bare = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")  # no resident index
    ops = [n({"id": is_in(seeds)}), e_forward(), n()]
    truth = int(edges["src"].isin(seeds).sum())
    assert _n(bare.gfql(ops, engine="pandas", index_policy="use")._edges) == truth
    used, code, seams = _explain(bare, ops, "pandas", "use")
    assert (used, code) == (False, "index_path_unavailable") and "native_seeded_hop" in seams
    # a scalar seed takes the same branch and says the same
    used, code, seams = _explain(bare, [n({"id": seeds[0]}), e_forward(), n()], "pandas", "use")
    assert (used, code) == (False, "index_path_unavailable") and "native_seeded_hop" in seams


@pytest.mark.route_engaged("native-fast", "index-hop")
def test_non_integral_members_decline_the_native_lane():
    # the lane's own gate: a member that is not an id falls back to the scan body (the general
    # chain's hop may still take the index when the lane is off, so this is an engagement pin)
    g, edges, nodes, seeds = _graph()
    assert _explain(g, [n({"id": is_in(seeds + ["x"])}), e_forward(), n()], "pandas", "force")[0] is False
