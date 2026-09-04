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
