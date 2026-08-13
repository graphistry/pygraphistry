"""Precomputed in/out degree facts.

The two-hop count kernel spends O(E) per query on a bincount plus gather. With
degrees precomputed the same answer is ``dot(indeg, outdeg)`` -- O(N). These pin
the equivalence and the declines.

STALENESS IS A WRONG ANSWER for this fact kind, unlike min/max where a stale fact
merely costs a scan, so the validity guards get their own pins.
"""
import numpy as np
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.index.build import build_degree_fact
from graphistry.compute.gfql.index.registry import GfqlIndexRegistry


def _edges() -> pd.DataFrame:
    return pd.DataFrame({"s": [0, 1, 2, 0, 0], "d": [1, 2, 0, 3, 4]})


def _oracle(e: pd.DataFrame, n: int) -> int:
    """What the kernel computes today: bincount + gather + sum."""
    return int(np.bincount(e["d"].values, minlength=n)[e["s"].values].sum())


def test_dot_of_degrees_equals_the_gather_sum_oracle() -> None:
    """The whole premise: O(N) dot == O(E) gather-sum, exactly."""
    e = _edges()
    f = build_degree_fact(e, "s", "d", 0, 4, Engine.PANDAS)
    assert f is not None
    assert int(np.dot(f.indeg, f.outdeg)) == _oracle(e, 5)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_equivalence_holds_on_random_graphs(seed: int) -> None:
    """Not a single hand case: the identity must hold for arbitrary degree
    distributions, including isolated nodes and self-loops."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 40))
    m = int(rng.integers(1, 200))
    e = pd.DataFrame({"s": rng.integers(0, n, m), "d": rng.integers(0, n, m)})
    f = build_degree_fact(e, "s", "d", 0, n - 1, Engine.PANDAS)
    assert f is not None
    assert int(np.dot(f.indeg, f.outdeg)) == _oracle(e, n)


def test_endpoint_outside_the_interval_declines_rather_than_clamps() -> None:
    """A clamp would silently MISCOUNT. Refusing costs only the scan."""
    assert build_degree_fact(_edges(), "s", "d", 0, 2, Engine.PANDAS) is None


def test_explicit_preconditions_decline() -> None:
    e = _edges()
    assert build_degree_fact(e, "s", "nope", 0, 4, Engine.PANDAS) is None
    assert build_degree_fact(e, "s", "d", 0, 10 ** 9, Engine.PANDAS) is None  # span cap
    assert build_degree_fact(e, "s", "d", 4, 0, Engine.PANDAS) is None        # hi < lo
    floats = pd.DataFrame({"s": [0.5], "d": [1.5]})
    assert build_degree_fact(floats, "s", "d", 0, 4, Engine.PANDAS) is None


def test_a_rebound_frame_invalidates_the_fact() -> None:
    """Staleness here is a WRONG COUNT, not a lost optimization. A frame that is
    equal-but-not-identical must miss, because the degrees may no longer describe
    it -- identity, not value, is the guard."""
    e = _edges()
    f = build_degree_fact(e, "s", "d", 0, 4, Engine.PANDAS)
    assert f is not None
    reg = GfqlIndexRegistry().with_degrees(f)
    assert reg.get_degree_valid("s", "d", e, Engine.PANDAS) is f
    assert reg.get_degree_valid("s", "d", e.copy(), Engine.PANDAS) is None
    assert reg.get_degree_valid("s", "d", e, Engine.CUDF) is None
    assert reg.get_degree_valid("s", "d", None, Engine.PANDAS) is None


def test_typed_degrees_are_keyed_separately() -> None:
    """Degrees must be per relationship type: q8 counts over one rel only, so a
    global array would be the wrong denominator for it."""
    e = pd.DataFrame({"s": [0, 1, 0], "d": [1, 2, 3], "rel": ["F", "F", "X"]})
    only_f = e[e["rel"] == "F"]
    f = build_degree_fact(only_f, "s", "d", 0, 3, Engine.PANDAS,
                          type_column="rel", type_value="F")
    assert f is not None
    reg = GfqlIndexRegistry().with_degrees(f)
    assert reg.get_degree_valid("s", "d", only_f, Engine.PANDAS, "rel", "F") is f
    assert reg.get_degree_valid("s", "d", only_f, Engine.PANDAS) is None  # global != typed
    assert int(np.dot(f.indeg, f.outdeg)) == _oracle(only_f.reset_index(drop=True), 4)
