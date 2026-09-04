"""#2023: the undirected seed-rediscovery rule in numpy stays equal to its reference.

The rule (components holding >1 seed, self-loops, the multigraph 2-core) is documented on
``undirected_rediscovered_seed_ids``; these pins hold the numpy form to the pure-Python
reference on random multigraphs (parallel edges, self-loops, isolated seeds, string ids)
and bound its wall time on a large traversal.
"""
import random
import time

import numpy as np
import pytest

from graphistry.compute.hop import (
    _undirected_rediscovered_seed_ids_reference as reference,
    undirected_rediscovered_seed_ids as vectorized,
)


def _random_multigraph(rng, n_nodes, n_edges, ids):
    src = [ids[rng.randrange(n_nodes)] for _ in range(n_edges)]
    dst = [ids[rng.randrange(n_nodes)] for _ in range(n_edges)]
    k = rng.randint(1, max(1, n_nodes // 2))
    seeds = rng.sample(ids[:n_nodes], k)
    return src, dst, seeds


@pytest.mark.parametrize("ids", [list(range(12)), [f"n{i}" for i in range(12)]])
def test_vectorized_equals_reference_on_random_multigraphs(ids):
    rng = random.Random(2023)
    for _ in range(3000):
        n_nodes = rng.randint(1, len(ids))
        n_edges = rng.randint(0, 14)
        src, dst, seeds = _random_multigraph(rng, n_nodes, n_edges, ids)
        # seeds may include ids absent from every edge: never rediscovered, never crash
        if rng.random() < 0.2 and n_nodes < len(ids):
            seeds = seeds + [ids[n_nodes]]
        assert vectorized(src, dst, seeds) == reference(src, dst, seeds), (src, dst, seeds)


def test_long_pendant_paths_take_the_csr_frontier_path_and_agree():
    """A path far longer than the scan-round budget exercises the CSR branches of both rules."""
    from graphistry.compute import hop as hop_module
    k = 40 * hop_module._SCAN_ROUNDS
    src = list(range(k))
    dst = list(range(1, k + 1))
    # seeds at both ends share the component (rule A over > _SCAN_ROUNDS BFS levels);
    # peeling the path is > _SCAN_ROUNDS rounds (rule B); a triangle hung on the far end
    # keeps one seed on a cycle.
    src += [k, k + 1, k + 2]
    dst += [k + 1, k + 2, k]
    seeds = [0, k // 2, k + 1]
    assert vectorized(src, dst, seeds) == reference(src, dst, seeds) == {0, k // 2, k + 1}
    lone = vectorized(src[:k], dst[:k], [5])
    assert lone == reference(src[:k], dst[:k], [5]) == set()


def test_sparse_integer_ids_are_factorized_not_indexed_directly():
    big = 10 ** 12
    src = [big, big + 1, big + 2]
    dst = [big + 1, big + 2, big]
    assert vectorized(src, dst, [big + 2]) == {big + 2}
    assert vectorized(np.array(src), np.array(dst), np.array([big])) == {big}


def test_float_ids_take_the_reference_path():
    src = [0.5, 1.5, 2.5]
    dst = [1.5, 2.5, 0.5]
    assert vectorized(src, dst, [0.5]) == reference(src, dst, [0.5]) == {0.5}
    assert vectorized(np.array(src), np.array(dst), np.array([9.0])) == set()


def test_seeds_only_in_the_seed_list_never_survive():
    assert vectorized([1, 2], [2, 1], [7]) == set()
    assert vectorized(np.array([1, 2]), np.array([2, 1]), np.array([7, 1])) == {1}


def test_accepts_numpy_arrays_and_returns_plain_ids():
    src = np.array([0, 1, 2, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 6], dtype=np.int64)
    out = vectorized(src, dst, np.array([0, 5]))
    assert out == {0}
    assert all(type(x) is int for x in out)


def test_unorderable_ids_fall_back_to_the_reference():
    src = [0, "a", 1]
    dst = ["a", 1, 0]
    assert vectorized(src, dst, [0]) == reference(src, dst, [0]) == {0}


def test_empty_inputs():
    assert vectorized([], [], [1]) == set()
    assert vectorized([1], [2], []) == set()


def test_no_per_edge_python_loop_on_a_large_traversal():
    """A 2M-edge traversal from 50 seeds stays under 10 s (#2023 repro was 60 s for one query)."""
    rng = np.random.default_rng(2023)
    n = 400_000
    src = rng.integers(0, n, 2_000_000)
    dst = rng.integers(0, n, 2_000_000)
    seeds = rng.choice(n, 50, replace=False)
    t0 = time.perf_counter()
    out = vectorized(src, dst, seeds)
    assert time.perf_counter() - t0 < 10.0
    assert out  # a random graph this dense is one giant cyclic component: seeds survive
