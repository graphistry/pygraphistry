"""#2023: the undirected seed-rediscovery rule, engine-native, on every engine.

Oracle: a pure-Python transcription of the rule (components holding >1 seed, self-loops,
the multigraph 2-core), itself checked against brute-force edge-disjoint-walk enumeration
when the rule was introduced (#1918). Each case below pins BOTH sides of a boundary: the
seed that is kept and the seed that is dropped for the same reason.
"""
import random
from typing import Dict, Hashable, List, Set, Tuple

import pandas as pd
import pytest

from graphistry.compute.gfql.seed_rediscovery import rediscovered_seed_ids as pandas_rule

try:
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.seed_rediscovery import (
        rediscovered_seed_ids as polars_rule,
    )
    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

ENGINES = [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")),
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
]


def reference(src: List[Hashable], dst: List[Hashable], seed_ids: List[Hashable]) -> Set[Hashable]:
    seeds = set(seed_ids)
    if not seeds or not src:
        return set()
    adjacency: Dict[Hashable, Set[Hashable]] = {}
    for u, v in zip(src, dst):
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    keep: Set[Hashable] = set()
    seen: Set[Hashable] = set()
    for seed in seeds:
        if seed in seen:
            continue
        stack, component = [seed], set()  # type: List[Hashable], Set[Hashable]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            seen.add(current)
            stack.extend(n for n in adjacency.get(current, set()) if n not in component)
        if len(component & seeds) > 1:
            keep.update(component & seeds)
    loop_nodes = {u for u, v in zip(src, dst) if u == v}
    simple = [(u, v) for u, v in zip(src, dst) if u != v]
    incident: Dict[Hashable, Set[int]] = {}
    endpoints: Dict[int, Tuple[Hashable, Hashable]] = {}
    for eid, (u, v) in enumerate(simple):
        endpoints[eid] = (u, v)
        incident.setdefault(u, set()).add(eid)
        incident.setdefault(v, set()).add(eid)
    removed: Set[Hashable] = set()
    queue = [node for node, eids in incident.items() if len(eids) <= 1]
    while queue:
        current = queue.pop()
        if current in removed or len(incident.get(current, set())) > 1:
            continue
        removed.add(current)
        for eid in list(incident.get(current, set())):
            u, v = endpoints[eid]
            other = v if u == current else u
            if other in removed:
                continue
            incident[other].discard(eid)
            if len(incident[other]) <= 1:
                queue.append(other)
        incident[current] = set()
    cycle_nodes = loop_nodes | {n for n, eids in incident.items() if eids and n not in removed}
    return (keep | cycle_nodes) & seeds


def run(engine: str, src, dst, seeds) -> Set[Hashable]:
    edges = pd.DataFrame({"s": src, "d": dst})
    seed_df = pd.DataFrame({"id": seeds})
    if engine == "pandas":
        out = pandas_rule(edges, "s", "d", seed_df, "id")
        return set(out["id"].tolist())
    if engine == "cudf":
        out = pandas_rule(cudf.DataFrame.from_pandas(edges), "s", "d", cudf.DataFrame.from_pandas(seed_df), "id")
        assert out.__class__.__module__.startswith("cudf"), "the rule must stay on the GPU frame"
        return set(out.to_pandas()["id"].tolist())
    out = polars_rule(pl.from_pandas(edges), "s", "d", pl.from_pandas(seed_df), "id")
    assert isinstance(out, pl.DataFrame)
    return set(out["id"].to_list())


# Boundary cases: (name, src, dst, seeds, kept). Every case names a kept AND a dropped seed
# for the same rule, or pairs with the case right after it.
CASES = [
    # rule A boundary: alone in an acyclic component -> dropped; two seeds in it -> both kept
    ("path_one_seed", [0, 1, 2, 3], [1, 2, 3, 4], [0], set()),
    ("path_middle_seed", [0, 1, 2, 3], [1, 2, 3, 4], [2], set()),
    ("path_two_seeds", [0, 1, 2, 3], [1, 2, 3, 4], [0, 4], {0, 4}),
    ("path_two_seeds_one_isolated", [0, 1, 2, 3, 7], [1, 2, 3, 4, 8], [0, 4, 7], {0, 4}),
    ("two_components_one_seed_each", [0, 5], [1, 6], [0, 5], set()),
    # rule B boundary: on a cycle -> kept; hanging off the cycle -> dropped
    ("triangle_seed", [0, 1, 2], [1, 2, 0], [0], {0}),
    ("triangle_pendant_seed", [0, 1, 2, 2], [1, 2, 0, 9], [9], set()),
    ("parallel_edges_two_cycle", [0, 0], [1, 1], [0], {0}),
    ("single_edge_no_cycle", [0], [1], [0], set()),
    ("self_loop", [0, 1], [0, 2], [0, 1], {0}),
    ("self_loop_only_other_seed_dropped", [0, 1], [0, 2], [1], set()),
    ("star_hub", [0, 0, 0], [1, 2, 3], [0], set()),
    ("star_leaf_and_hub", [0, 0, 0], [1, 2, 3], [0, 3], {0, 3}),
    # seeds absent from every edge never survive
    ("seed_absent", [0, 1, 2], [1, 2, 0], [42], set()),
    ("seed_absent_beside_kept", [0, 1, 2], [1, 2, 0], [42, 1], {1}),
    # a NULL endpoint is not an identity: the edge is ignored, not a bridge
    ("null_endpoint_ignored", [0, 1, None], [1, None, 2], [0, 2], set()),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,src,dst,seeds,kept", CASES, ids=[c[0] for c in CASES])
def test_boundary_cases(engine, name, src, dst, seeds, kept):
    src_f = pd.Series(src, dtype="Int64" if None in src else "int64")
    dst_f = pd.Series(dst, dtype="Int64" if None in dst else "int64")
    assert run(engine, src_f, dst_f, seeds) == kept
    if None not in src and None not in dst:
        assert reference(src, dst, seeds) == kept, "the oracle and the pinned expectation disagree"


@pytest.mark.parametrize("engine", ENGINES)
def test_string_ids(engine):
    src = ["a", "b", "c", "c"]
    dst = ["b", "c", "a", "z"]
    assert run(engine, src, dst, ["a", "z"]) == {"a", "z"}, "z shares a component with seed a"
    assert run(engine, src, dst, ["z"]) == set(), "alone, the pendant z has no way back"
    assert run(engine, src, dst, ["a"]) == {"a"}, "a sits on the triangle"


@pytest.mark.parametrize("engine", ENGINES)
def test_seed_dtype_narrower_than_edges(engine):
    src = pd.Series([0, 1, 2], dtype="int64")
    dst = pd.Series([1, 2, 0], dtype="int64")
    assert run(engine, src, dst, pd.Series([0], dtype="int32")) == {0}


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_inputs(engine):
    assert run(engine, [], [], [1]) == set()
    assert run(engine, [1], [2], []) == set()


@pytest.mark.parametrize("engine", ENGINES)
def test_long_pendant_path_and_far_seeds(engine):
    k = 300
    src = list(range(k)) + [k, k + 1, k + 2]
    dst = list(range(1, k + 1)) + [k + 1, k + 2, k]
    seeds = [0, k // 2, k + 1]
    assert run(engine, src, dst, seeds) == reference(src, dst, seeds) == {0, k // 2, k + 1}
    assert run(engine, src[:k], dst[:k], [5]) == set()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ids", [list(range(12)), [f"n{i}" for i in range(12)]], ids=["int", "str"])
def test_random_multigraphs_equal_the_reference(engine, ids):
    rng = random.Random(2023)
    n_graphs = 600 if engine == "pandas" else 200
    for _ in range(n_graphs):
        n_nodes = rng.randint(1, len(ids))
        n_edges = rng.randint(0, 14)
        src = [ids[rng.randrange(n_nodes)] for _ in range(n_edges)]
        dst = [ids[rng.randrange(n_nodes)] for _ in range(n_edges)]
        seeds = rng.sample(ids[:n_nodes], rng.randint(1, max(1, n_nodes // 2)))
        if rng.random() < 0.2 and n_nodes < len(ids):
            seeds.append(ids[n_nodes])
        assert run(engine, src, dst, seeds) == reference(src, dst, seeds), (src, dst, seeds)
