"""Correctness gates for the Graphalytics kernels.

Three tiers in fail-fast order:

1. Toy graphs with hand-computed answers -- catches semantics bugs (the dangling
   term, CDLP's tie-break, multiset counting, MIS isolated-vertex handling).
2. Independent references on random graphs -- networkx for WCC and Dijkstra, a
   naive Python LDBC implementation for CDLP, a plain-Python loop for PageRank.
3. Chunk invariance -- chunked results must equal unchunked ones. This is the
   property the 1B-edge runs depend on, and it is where an index-alignment bug
   hid: `gather` returns a 0-based Series while an `iloc` slice keeps its
   original index, so a misaligned `df_cons` silently produced NaN. It only
   showed up once chunking was switched on.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.algorithms import _dfops as D
from graphistry.compute.algorithms import kernels as K


def _edges(src, dst, weights=None):
    e = pd.DataFrame({"s": np.array(src, dtype="int32"), "d": np.array(dst, dtype="int32")})
    if weights is not None:
        e["w"] = np.array(weights, dtype="float32")
    return e


def _random_graph(seed: int, n: int, m: int, weighted: bool = False):
    rng = np.random.default_rng(seed)
    e = pd.DataFrame({"s": rng.integers(0, n, m), "d": rng.integers(0, n, m)})
    e = e[e["s"] != e["d"]]
    dense, ids, v_count = D.dense_renumber(e, "s", "d")
    if weighted:
        dense["w"] = K.make_weights(dense, "s", "d")
    return dense, ids, v_count


# --------------------------------------------------------------------------
# Tier 1: hand-computed answers
# --------------------------------------------------------------------------


def test_dense_renumber_is_monotone():
    """Monotone renumbering is what makes dense min-label == min original id."""
    e = _edges([10, 10, 50, 90], [50, 90, 90, 10])
    dense, ids, v_count = D.dense_renumber(e, "s", "d")
    assert v_count == 3
    assert list(ids) == [10, 50, 90] == sorted(ids)
    assert str(dense["s"].dtype) == "int32"


def test_wcc_labels_are_min_original_id():
    e = _edges([0, 1, 0, 5], [1, 2, 2, 6])
    dense, ids, v_count = D.dense_renumber(e, "s", "d")
    lbl = K.wcc(dense, "s", "d", v_count)
    as_orig = [int(ids.iloc[int(x)]) for x in lbl]
    assert as_orig == [0, 0, 0, 5, 5]


def test_pagerank_matches_hand_computation_with_dangling():
    """Chain 0->1->2 where 2 is dangling; mass must be conserved."""
    e = _edges([0, 1], [1, 2])
    pr = K.pagerank(e, "s", "d", 3, iterations=10)

    v_count, damping, iters = 3, 0.85, 10
    out = {0: [1], 1: [2], 2: []}
    ref = [1 / v_count] * v_count
    for _ in range(iters):
        dang = sum(ref[v] for v in range(v_count) if not out[v])
        inflow = [0.0] * v_count
        for u in range(v_count):
            for w in out[u]:
                inflow[w] += ref[u] / len(out[u])
        ref = [(1 - damping) / v_count + damping * (inflow[v] + dang / v_count) for v in range(v_count)]

    assert float(pr.sum()) == pytest.approx(1.0, abs=1e-12)
    for got, want in zip(pr, ref):
        assert float(got) == pytest.approx(want, rel=1e-12)


def test_cdlp_tie_breaks_to_smallest_label_and_oscillates():
    """Star 0--{1,2,3}: vertex 0 sees a 3-way tie and must pick the smallest.

    The period-2 oscillation is why CDLP has no early exit -- 'no change' may
    never happen, so the spec is K iterations regardless.
    """
    e = _edges([0, 0, 0], [1, 2, 3])
    assert list(K.cdlp(e, "s", "d", 4, iterations=1)) == [1, 0, 0, 0]
    assert list(K.cdlp(e, "s", "d", 4, iterations=2)) == [0, 1, 1, 1]
    assert list(K.cdlp(e, "s", "d", 4, iterations=3)) == [1, 0, 0, 0]


def test_cdlp_uses_multiset_not_set_semantics():
    """Parallel edges count multiple times.

    Neighbours of 0 are the multiset {2,2,1}, so label 2 wins with count 2.
    Under set semantics it would be a {1,2} tie and resolve to 1, so this case
    discriminates the two readings of the spec.
    """
    e = _edges([0, 0, 0], [2, 2, 1])
    assert int(K.cdlp(e, "s", "d", 3, iterations=1).iloc[0]) == 2


def test_sssp_re_relaxes_a_settled_vertex():
    """0->2 costs 100 directly but 7 via 1, so vertex 2 must be improved after
    it has already been assigned -- the case a visited-set BFS gets wrong."""
    e = _edges([0, 1, 0], [1, 2, 2], [5.0, 2.0, 100.0])
    assert list(K.sssp(e, "s", "d", "w", 3, source=0)) == [0.0, 5.0, 7.0]


def test_sssp_marks_unreachable_as_inf():
    e = _edges([0], [1], [2.0])
    out = list(K.sssp(e, "s", "d", "w", 3, source=0))
    assert out[:2] == [0.0, 2.0]
    assert np.isinf(out[2])


def test_mis_on_a_path_with_an_isolated_vertex():
    e = _edges([0, 1, 2, 3], [1, 2, 3, 4])
    in_set = K.mis(e, "s", "d", 6)
    chosen = {i for i, x in enumerate(in_set) if x}
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert all(not (u in chosen and v in chosen) for u, v in edges)
    assert 5 in chosen, "isolated vertices must be in the set or maximality fails"


# --------------------------------------------------------------------------
# Tier 2: independent references
# --------------------------------------------------------------------------


def test_wcc_matches_networkx():
    nx = pytest.importorskip("networkx")
    dense, _, v_count = _random_graph(11, 500, 2500)
    g = nx.DiGraph()
    g.add_nodes_from(range(v_count))
    g.add_edges_from(zip(dense["s"].tolist(), dense["d"].tolist()))

    ref = {}
    for comp in nx.weakly_connected_components(g):
        low = min(comp)
        for v in comp:
            ref[v] = low

    mine = K.wcc(dense, "s", "d", v_count)
    assert all(int(mine.iloc[v]) == ref[v] for v in range(v_count))


def test_sssp_matches_dijkstra_exactly():
    """Integer weights in [1,255] keep every distance exactly representable in
    float32, so this is an EXACT comparison, not a tolerance one."""
    nx = pytest.importorskip("networkx")
    dense, _, v_count = _random_graph(11, 500, 2500, weighted=True)
    g = nx.DiGraph()
    g.add_nodes_from(range(v_count))
    for s, d, w in zip(dense["s"].tolist(), dense["d"].tolist(), dense["w"].tolist()):
        g.add_edge(int(s), int(d), weight=float(w))

    ref = nx.single_source_dijkstra_path_length(g, 0, weight="weight")
    mine = K.sssp(dense, "s", "d", "w", v_count, source=0)
    for v in range(v_count):
        assert float(mine.iloc[v]) == ref.get(v, float("inf"))


def test_cdlp_matches_naive_ldbc_reference():
    dense, _, v_count = _random_graph(11, 300, 1500)
    iterations = 10

    nbr: dict[int, list[int]] = {v: [] for v in range(v_count)}
    for s, d in zip(dense["s"].tolist(), dense["d"].tolist()):
        nbr[int(s)].append(int(d))
        nbr[int(d)].append(int(s))

    lbl = list(range(v_count))
    for _ in range(iterations):
        nxt = list(lbl)
        for v in range(v_count):
            if not nbr[v]:
                continue
            counts = Counter(lbl[u] for u in nbr[v])
            # most frequent, ties -> smallest label
            nxt[v] = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        lbl = nxt

    mine = K.cdlp(dense, "s", "d", v_count, iterations=iterations)
    assert [int(x) for x in mine] == lbl


def test_mis_invariants_hold_at_scale():
    dense, _, v_count = _random_graph(3, 2000, 12000)
    in_set = K.mis(dense, "s", "d", v_count).values
    src = dense["s"].values
    dst = dense["d"].values

    assert not bool((in_set[src] & in_set[dst]).any()), "not independent"

    nbr_in_set = np.zeros(v_count, dtype=bool)
    np.logical_or.at(nbr_in_set, src, in_set[dst])
    np.logical_or.at(nbr_in_set, dst, in_set[src])
    assert not bool((~in_set & ~nbr_in_set).any()), "not maximal"

    assert in_set.sum() > 0


def test_mis_is_deterministic_and_seed_sensitive():
    dense, _, v_count = _random_graph(3, 2000, 12000)
    a = K.mis(dense, "s", "d", v_count).values
    assert np.array_equal(a, K.mis(dense, "s", "d", v_count).values)
    assert not np.array_equal(a, K.mis(dense, "s", "d", v_count, seed=99).values)


# --------------------------------------------------------------------------
# Tier 3: chunk invariance -- the property the 1B-edge runs depend on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["wcc", "cdlp", "sssp", "mis"])
def test_chunked_equals_unchunked_exactly(kernel):
    dense, _, v_count = _random_graph(7, 400, 3000, weighted=True)
    run = {
        "wcc": lambda c: K.wcc(dense, "s", "d", v_count, chunks=c),
        "cdlp": lambda c: K.cdlp(dense, "s", "d", v_count, chunks=c),
        "sssp": lambda c: K.sssp(dense, "s", "d", "w", v_count, source=0, chunks=c),
        "mis": lambda c: K.mis(dense, "s", "d", v_count, chunks=c),
    }[kernel]
    assert np.array_equal(run(1).values, run(4).values)


def test_pagerank_chunking_drift_is_float_noise_only():
    """PageRank is the one kernel compared by tolerance: float64 summation is
    not associative, so chunk boundaries shift the last bits (~1e-16). Every
    other kernel is integer or exact-float32 and must match bitwise."""
    dense, _, v_count = _random_graph(7, 400, 3000)
    a = K.pagerank(dense, "s", "d", v_count, chunks=1).values
    b = K.pagerank(dense, "s", "d", v_count, chunks=4).values
    rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30))
    assert rel < 1e-12, f"chunking drift {rel:.3e} is larger than float noise"


# --------------------------------------------------------------------------
# Validators must REJECT corrupted results, not just accept good ones.
# A validator that always passes is worse than no validator, because it makes
# an unchecked number look checked.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _prepared():
    # The validators address columns as src/dst/w, matching what the runner's
    # `prepare` step produces -- so the fixture must use those names too.
    dense, _, v_count = _random_graph(5, 800, 5000, weighted=True)
    dense = dense.rename(columns={"s": "src", "d": "dst"})
    return {
        "edges": dense,
        "vertices": v_count,
        "edge_count": len(dense),
        "sssp_source": 0,
    }


def _run(kernel, prep):
    d, v = prep["edges"], prep["vertices"]
    return {
        "wcc": lambda: K.wcc(d, "src", "dst", v),
        "pagerank": lambda: K.pagerank(d, "src", "dst", v),
        "cdlp": lambda: K.cdlp(d, "src", "dst", v),
        "sssp": lambda: K.sssp(d, "src", "dst", "w", v, source=0),
        "mis": lambda: K.mis(d, "src", "dst", v),
    }[kernel]()


@pytest.mark.parametrize("kernel", ["wcc", "pagerank", "cdlp", "sssp", "mis"])
def test_validator_accepts_a_correct_result(kernel, _prepared):
    from graphistry.compute.algorithms import validate as V

    assert V.validate_result(kernel, _run(kernel, _prepared), _prepared)["status"] == "ok"


def test_wcc_validator_rejects_a_wrong_label(_prepared):
    from graphistry.compute.algorithms import validate as V

    bad = _run("wcc", _prepared).copy()
    bad.iloc[5] = 1 if int(bad.iloc[5]) != 1 else 2
    assert V.validate_result("wcc", bad, _prepared)["status"] == "fail"


def test_pagerank_validator_rejects_broken_mass(_prepared):
    from graphistry.compute.algorithms import validate as V

    bad = _run("pagerank", _prepared).copy()
    bad.iloc[0] *= 2
    assert V.validate_result("pagerank", bad, _prepared)["status"] == "fail"


def test_sssp_validator_rejects_a_triangle_inequality_violation(_prepared):
    from graphistry.compute.algorithms import validate as V

    bad = _run("sssp", _prepared).copy()
    bad.iloc[3] = bad.iloc[3] + 50
    assert V.validate_result("sssp", bad, _prepared)["status"] == "fail"


def test_mis_validator_rejects_both_failure_modes(_prepared):
    """All-in violates independence; all-out violates maximality. A validator
    that only checked one would pass one of these."""
    from graphistry.compute.algorithms import validate as V

    everything = _run("mis", _prepared).copy()
    everything.iloc[:] = True
    assert V.validate_result("mis", everything, _prepared)["status"] == "fail"

    nothing = _run("mis", _prepared).copy()
    nothing.iloc[:] = False
    assert V.validate_result("mis", nothing, _prepared)["status"] == "fail"
