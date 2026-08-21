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

import inspect
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace

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


def test_pagerank_cugraph_compatible_signature_defaults():
    signature = inspect.signature(K.pagerank)
    expected = {
        "alpha": 0.85,
        "personalization": None,
        "precomputed_vertex_out_weight": None,
        "max_iter": 100,
        "tol": 1.0e-5,
        "nstart": None,
        "dangling": None,
        "fail_on_nonconvergence": True,
    }
    assert {name: signature.parameters[name].default for name in expected} == expected
    assert signature.parameters["method"].default == "auto"


@pytest.mark.parametrize("method", ["fast", "bounded"])
def test_pagerank_weighted_personalized_matches_linear_system(method):
    edges = _edges([0, 0, 1, 2], [1, 2, 2, 0], [2.0, 1.0, 3.0, 4.0])
    personalization = pd.Series([0.7, 0.1, 0.1, 0.1])
    nstart = pd.Series([0.0, 1.0, 0.0, 0.0])
    out_weight = pd.Series([3.0, 3.0, 4.0, np.nan])

    got = K.pagerank(
        edges,
        "s",
        "d",
        4,
        alpha=0.85,
        personalization=personalization,
        precomputed_vertex_out_weight=out_weight,
        max_iter=1000,
        tol=1.0e-12,
        nstart=nstart,
        dangling={"ignored": 1.0},
        weight="w",
        method=method,
    )

    transition = np.array(
        [
            [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            list(personalization),
        ]
    )
    ref = np.linalg.solve(
        np.eye(4) - 0.85 * transition.T,
        (1.0 - 0.85) * personalization.to_numpy(),
    )
    assert [float(value) for value in got] == pytest.approx(
        list(ref), rel=1.0e-12, abs=1.0e-15
    )


@pytest.mark.skipif(__import__("os").environ.get("TEST_CUDF") != "1", reason="cuDF tests need TEST_CUDF=1")
def test_pagerank_cudf_fast_matches_pandas_full_controls():
    cudf = pytest.importorskip("cudf")
    edges = _edges([0, 0, 1, 2], [1, 2, 2, 0], [2.0, 1.0, 0.0, 4.0])
    gpu_edges = cudf.from_pandas(edges)

    expected_unweighted = K.pagerank(edges[["s", "d"]], "s", "d", 4, method="fast")
    actual_unweighted = K.pagerank(gpu_edges[["s", "d"]], "s", "d", 4, method="fast")
    assert actual_unweighted.to_pandas().to_numpy() == pytest.approx(expected_unweighted.to_numpy(), rel=1.0e-12, abs=1.0e-15)

    personalization = pd.Series([0.7, 0.1, 0.1, 0.1])
    nstart = pd.Series([0.0, 1.0, 0.0, 0.0])
    out_weight = pd.Series([3.0, 0.0, 4.0, np.nan])
    expected_weighted = K.pagerank(
        edges,
        "s",
        "d",
        4,
        alpha=0.85,
        personalization=personalization,
        precomputed_vertex_out_weight=out_weight,
        max_iter=1000,
        tol=1.0e-12,
        nstart=nstart,
        weight="w",
        method="fast",
    )
    actual_weighted = K.pagerank(
        gpu_edges,
        "s",
        "d",
        4,
        alpha=0.85,
        personalization=cudf.from_pandas(personalization),
        precomputed_vertex_out_weight=cudf.from_pandas(out_weight),
        max_iter=1000,
        tol=1.0e-12,
        nstart=cudf.from_pandas(nstart),
        weight="w",
        method="fast",
    )
    assert actual_weighted.to_pandas().to_numpy() == pytest.approx(expected_weighted.to_numpy(), rel=1.0e-12, abs=1.0e-15)


def test_pagerank_nonconvergence_policy_and_fixed_alias():
    edges = _edges([0, 1], [1, 2])
    with pytest.raises(K.ConvergenceError):
        K.pagerank(edges, "s", "d", 3, max_iter=1, tol=1.0e-30)

    ranks, converged = K.pagerank(
        edges,
        "s",
        "d",
        3,
        max_iter=1,
        tol=1.0e-30,
        fail_on_nonconvergence=False,
    )
    assert converged is False
    assert float(ranks.sum()) == pytest.approx(1.0)

    legacy = K.pagerank(edges, "s", "d", 3, iterations=10, damping=0.85)
    explicit = K.pagerank(
        edges,
        "s",
        "d",
        3,
        max_iter=10,
        alpha=0.85,
        stopping="fixed_iterations",
    )
    assert list(legacy) == pytest.approx(list(explicit), rel=1.0e-15)


def test_pagerank_tolerance_is_on_unit_mass_rank_scale():
    edges = _edges([0], [1])
    ranks, converged = K.pagerank(
        edges,
        "s",
        "d",
        100,
        max_iter=1,
        tol=0.01,
        fail_on_nonconvergence=False,
    )

    # First-step L1 delta is 0.01683: above tol, but below V * tol (=1).
    assert converged is False
    assert float(ranks.sum()) == pytest.approx(1.0)


def test_pagerank_method_validation_and_chunk_contract():
    edges = _edges([0, 1], [1, 2])
    with pytest.raises(ValueError, match="method must be"):
        K.pagerank(
            edges,
            "s",
            "d",
            3,
            method="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires chunks=1"):
        K.pagerank(edges, "s", "d", 3, method="fast", chunks=2)


def test_pagerank_auto_memory_preflight(monkeypatch):
    edges = _edges([0, 1, 2, 2], [1, 2, 0, 1])
    fixed = 64 * 1024 * 1024
    unweighted = K._pagerank_fast_estimated_bytes(edges, 3, False)
    weighted = K._pagerank_fast_estimated_bytes(edges, 3, True)
    assert unweighted == fixed + 64 * 3 + 8 * len(edges)
    assert weighted == fixed + 64 * 3 + 24 * len(edges)

    monkeypatch.setattr(K, "_pagerank_available_bytes", lambda _: 2 * unweighted)
    assert K._pagerank_auto_uses_fast(edges, 3, False, 1) is True
    assert K._pagerank_auto_uses_fast(edges, 3, False, 2) is False

    monkeypatch.setattr(K, "_pagerank_available_bytes", lambda _: 2 * unweighted - 1)
    assert K._pagerank_auto_uses_fast(edges, 3, False, 1) is False
    monkeypatch.setattr(K, "_pagerank_available_bytes", lambda _: None)
    assert K._pagerank_auto_uses_fast(edges, 3, False, 1) is False


def test_pagerank_iteration_rejects_mass_drift():
    initial = pd.Series([1.0])

    def bad_step(current):
        return current, 1.5, 0.0

    with pytest.raises(AssertionError, match="mass not conserved"):
        K._pagerank_iterations(initial, bad_step, 1, 1.0e-5, "convergence")


def test_pagerank_gpu_memory_estimate_without_gpu(monkeypatch):
    edges = _edges([0, 1, 2, 2], [1, 2, 0, 1])
    monkeypatch.setattr(K, "is_cudf", lambda _: True)

    fixed = 64 * 1024 * 1024
    assert K._pagerank_fast_estimated_bytes(edges, 3, False) == fixed + 96 * 3
    assert K._pagerank_fast_estimated_bytes(edges, 3, True) == fixed + 96 * 3 + 16 * len(edges)


def test_pagerank_cgroup_available_bytes_contract(tmp_path):
    limit_path = tmp_path / "limit"
    current_path = tmp_path / "current"

    limit_path.write_text("1000", encoding="utf-8")
    current_path.write_text("250", encoding="utf-8")
    assert K._pagerank_cgroup_available_bytes(str(limit_path), str(current_path)) == 750

    limit_path.write_text("max", encoding="utf-8")
    assert K._pagerank_cgroup_available_bytes(str(limit_path), str(current_path)) is None

    limit_path.write_text("invalid", encoding="utf-8")
    assert K._pagerank_cgroup_available_bytes(str(limit_path), str(current_path)) is None

    limit_path.write_text(str(1 << 60), encoding="utf-8")
    current_path.write_text("0", encoding="utf-8")
    assert K._pagerank_cgroup_available_bytes(str(limit_path), str(current_path)) is None

    limit_path.write_text("100", encoding="utf-8")
    current_path.write_text("-1", encoding="utf-8")
    assert K._pagerank_cgroup_available_bytes(str(limit_path), str(current_path)) is None


def test_pagerank_host_available_bytes_contract(monkeypatch):
    values = {"SC_AVPHYS_PAGES": 10, "SC_PAGE_SIZE": 100}
    monkeypatch.setattr(K.os, "sysconf", lambda name: values[name])
    monkeypatch.setattr(
        K,
        "_pagerank_cgroup_available_bytes",
        lambda limit_path, _current_path: 300 if limit_path.endswith("memory.max") else None,
    )
    assert K._pagerank_host_available_bytes() == 300

    def unavailable(_name):
        raise OSError("sysconf unavailable")

    monkeypatch.setattr(K.os, "sysconf", unavailable)
    monkeypatch.setattr(K, "_pagerank_cgroup_available_bytes", lambda *_: None)
    assert K._pagerank_host_available_bytes() is None


def test_pagerank_gpu_available_bytes_probe_without_gpu(monkeypatch):
    fake_cupy = ModuleType("cupy")
    fake_cupy.__dict__["cuda"] = SimpleNamespace(
        Device=lambda: SimpleNamespace(mem_info=(123, 456))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(K, "is_cudf", lambda _: True)
    assert K._pagerank_available_bytes(object()) == 123

    class BrokenDevice:
        def __init__(self):
            raise RuntimeError("probe failed")

    fake_cupy.__dict__["cuda"] = SimpleNamespace(Device=BrokenDevice)
    assert K._pagerank_available_bytes(object()) is None


def test_dfops_lazy_gpu_dispatch_without_gpu(monkeypatch):
    fake_cupy = ModuleType("cupy")
    fake_cupy.__dict__.update(
        {"stack": lambda values: values, "asnumpy": lambda values: values}
    )
    fake_cudf = ModuleType("cudf")
    fake_cudf.__dict__["Series"] = lambda values: ("cudf-series", values)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cudf", fake_cudf)
    monkeypatch.setattr(D, "is_cudf", lambda _: True)

    values = np.array([1.0, 2.0])

    class FakeSeries:
        pass

    series = FakeSeries()
    series.values = values
    assert D.array_namespace(object()) is fake_cupy
    assert D.series_to_array(series) is values
    assert D.series_from_array(object(), values) == ("cudf-series", values)
    assert D.to_host_floats([]) == ()

    fake_scalar = type("FakeCupyScalar", (float,), {"__module__": "cupy"})
    assert D.to_host_floats([fake_scalar(1.25), fake_scalar(2.5)]) == (1.25, 2.5)


def test_pagerank_fast_dispatches_cudf_backend_without_gpu(monkeypatch):
    edges = _edges([0], [1])
    vectors = [pd.Series([0.5, 0.5]) for _ in range(4)]
    expected = (pd.Series([0.25, 0.75]), True)
    received = []

    def cudf_fast(*args):
        received.append(args)
        return expected

    monkeypatch.setattr(K, "is_cudf", lambda _: True)
    monkeypatch.setattr(K, "_pagerank_cudf_fast", cudf_fast)
    actual = K._pagerank_fast(
        edges,
        "s",
        "d",
        2,
        None,
        vectors[0],
        vectors[1],
        vectors[2],
        vectors[3],
        0.85,
        100,
        1.0e-5,
        "convergence",
    )

    assert actual is expected
    assert len(received) == 1
    args = received[0]
    assert args[0] is edges
    assert args[1:5] == ("s", "d", 2, None)
    assert all(
        actual_vector is expected_vector
        for actual_vector, expected_vector in zip(args[5:9], vectors)
    )
    assert args[9:] == (0.85, 100, 1.0e-5, "convergence")


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


def test_pagerank_chunking_drift_is_float_noise_only(monkeypatch):
    """Fast, bounded, and chunked reductions differ only by float noise."""
    monkeypatch.setattr(K, "_pagerank_available_bytes", lambda _: 1 << 60)
    dense, _, v_count = _random_graph(7, 400, 3000)
    auto_fast = K.pagerank(dense, "s", "d", v_count, chunks=1).values
    explicit_fast = K.pagerank(
        dense, "s", "d", v_count, method="fast"
    ).values
    bounded = K.pagerank(
        dense, "s", "d", v_count, method="bounded"
    ).values
    auto_chunked = K.pagerank(dense, "s", "d", v_count, chunks=4).values
    assert np.array_equal(auto_fast, explicit_fast)
    for candidate in (bounded, auto_chunked):
        rel = np.max(
            np.abs(auto_fast - candidate) / np.maximum(np.abs(auto_fast), 1e-30)
        )
        assert rel < 1e-12, f"PageRank drift {rel:.3e} exceeds float noise"


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
