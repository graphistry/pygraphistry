"""``sorted_unique`` returns exactly ``xp.unique`` for every dtype the index layer sees."""
import numpy as np
import pytest

from graphistry.compute.gfql.index.engine_arrays import sorted_unique, union1d

rng = np.random.default_rng(0)

CASES = [
    np.array([], dtype=np.int64),
    np.array([7], dtype=np.int64),
    np.array([3, 1, 3, 2, 1], dtype=np.int32),
    (rng.integers(0, 3_000_000, 5_000) * 1024 + 2**42).astype(np.int64),
    np.array([np.iinfo(np.uint64).max, 0, np.iinfo(np.uint64).max, 1], dtype=np.uint64),
    np.array([True, False, True], dtype=bool),
    np.array([1.5, np.nan, 1.5, np.nan, -0.0, 0.0]),
    np.array(["b", "a", "b"], dtype=object),
    np.array(["b", "a", "b"]),
]


@pytest.mark.parametrize("values", CASES, ids=[f"{c.dtype}-{c.shape[0]}" for c in CASES])
def test_sorted_unique_matches_numpy_unique(values: np.ndarray) -> None:
    got = sorted_unique(np, values)
    expected = np.unique(values)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_union1d_matches_numpy_union1d() -> None:
    a = rng.integers(0, 50, 200).astype(np.int64)
    b = rng.integers(25, 75, 200).astype(np.int64)
    np.testing.assert_array_equal(union1d(a, b, np), np.union1d(a, b))
    np.testing.assert_array_equal(union1d(a[:0], b, np), np.unique(b))
    np.testing.assert_array_equal(union1d(a, b[:0], np), np.unique(a))


def test_sorted_unique_cupy_matches_cupy_unique() -> None:
    cp = pytest.importorskip("cupy")
    try:
        values = cp.asarray(CASES[3])
    except Exception:  # no usable GPU
        pytest.skip("no GPU")
    assert bool((sorted_unique(cp, values) == cp.unique(values)).all())
