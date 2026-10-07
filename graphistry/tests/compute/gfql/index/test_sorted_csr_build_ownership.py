"""CSR buckets preserve rows and independent sidecars across sorted and general inputs."""
import numpy as np
import pytest

from graphistry.compute.gfql.index.build import _csr_from_keys


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    return np if request.param == "numpy" else pytest.importorskip("cupy")


def host(values, backend):
    return np.asarray(values) if backend is np else backend.asnumpy(values)


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint64", "float64"])
@pytest.mark.parametrize("values", [[], [1], [0, 1, 2], [2, 1, 0], [1, 1, 2], [2, 1, 2, 1], [0, 0, 0]])
def test_csr_buckets_preserve_every_original_row(values, dtype, backend):
    keys = backend.asarray(values, dtype=dtype)
    source = host(keys, backend).copy()
    if backend is np:
        keys.flags.writeable = False
    unique, offsets, positions = _csr_from_keys(keys, backend)
    unique, offsets, positions = (host(array, backend) for array in (unique, offsets, positions))
    np.testing.assert_array_equal(unique, np.unique(source))
    assert offsets.dtype == positions.dtype == np.dtype("int64")
    assert unique.dtype == source.dtype
    assert len(offsets) == len(unique) + 1
    assert offsets[0] == 0 and offsets[-1] == len(source)
    np.testing.assert_array_equal(np.sort(positions), np.arange(len(source)))
    for group, value in enumerate(unique):
        rows = positions[offsets[group]:offsets[group + 1]]
        np.testing.assert_array_equal(np.sort(rows), np.flatnonzero(source == value))
    np.testing.assert_array_equal(host(keys, backend), source)


@pytest.mark.parametrize("values", [[], [1], [0, 1, 2], [2, 1, 2, 1]])
@pytest.mark.parametrize("mutated", [0, 1, 2])
def test_csr_arrays_have_independent_buffers(values, mutated, backend):
    keys = backend.asarray(values, dtype=backend.int64)
    before = host(keys, backend).copy()
    arrays = _csr_from_keys(keys, backend)
    saved = [host(array, backend).copy() for array in arrays]
    if arrays[mutated].size:
        arrays[mutated][0] = -999
    np.testing.assert_array_equal(host(keys, backend), before)
    for number, array in enumerate(arrays):
        if number != mutated:
            np.testing.assert_array_equal(host(array, backend), saved[number])


@pytest.mark.parametrize("dtype,values", [
    ("int64", [-(2**63), 0, 2**63 - 1]),
    ("uint64", [0, 2**63, 2**64 - 1]),
    ("float64", [float("-inf"), -0.0, float("inf")]),
])
def test_csr_extreme_keys_keep_native_order_and_values(dtype, values, backend):
    keys = backend.asarray(values, dtype=dtype)
    unique, offsets, positions = _csr_from_keys(keys, backend)
    np.testing.assert_array_equal(host(unique, backend), np.asarray(values, dtype=dtype))
    np.testing.assert_array_equal(host(offsets, backend), np.arange(len(values) + 1))
    np.testing.assert_array_equal(host(positions, backend), np.arange(len(values)))


def test_custom_numpy_comparisons_keep_sort_semantics():
    class AlwaysGreater(np.ndarray):
        def __gt__(self, other):
            return np.ones(self.shape, dtype=bool)

    keys = np.asarray([2, 0, 1]).view(AlwaysGreater)
    unique, offsets, positions = _csr_from_keys(keys, np)
    np.testing.assert_array_equal(unique, [0, 1, 2])
    np.testing.assert_array_equal(offsets, [0, 1, 2, 3])
    np.testing.assert_array_equal(positions, [1, 2, 0])


def test_object_keys_keep_original_comparison_contract():
    class OrderedKey:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

        def __gt__(self, other):
            raise AssertionError("Object keys must retain the original sort comparator")

    values = [OrderedKey(2), OrderedKey(0), OrderedKey(1)]
    keys = np.asarray(values, dtype=object)
    # NumPy sorting also invokes __gt__; preserve its existing error behavior.
    with pytest.raises(AssertionError):
        _csr_from_keys(keys, np)
    assert list(keys) == values


@pytest.mark.parametrize("polars_engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("columns", [("a",), ("a", "b")])
@pytest.mark.parametrize("data", [
    {"a": [], "b": []},
    {"a": [0., 1., 2.], "b": [2., 1., 0.]},
    {"a": [0., None, 2.], "b": [None, 1., 0.]},
    {"a": [0., 1., 2.], "b": [None, 1., 0.]},
    {"a": [None, None], "b": [None, None]},
    {"a": [0., float("nan"), 2.], "b": [2., 1., 0.]},
])
def test_polars_non_null_build_keeps_rows_and_original_positions(data, columns, polars_engine):
    pl = pytest.importorskip("polars")
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.build import _non_null_id_rows

    frame = pl.DataFrame(data, schema={"a": pl.Float64, "b": pl.Float64})
    before = frame.clone()
    expected = frame.drop_nulls(subset=columns)
    valid, positions = _non_null_id_rows(frame, columns, Engine(polars_engine), np)
    assert valid.equals(expected)
    assert frame.equals(before)
    if len(expected) == len(frame):
        assert valid is frame and positions is None
    else:
        expected_positions = frame.with_row_index("row").drop_nulls(subset=columns).get_column("row").to_numpy()
        np.testing.assert_array_equal(positions, expected_positions)
        assert positions.dtype == np.dtype("int64")


@pytest.mark.parametrize("polars_engine", ["polars", "polars-gpu"])
def test_polars_non_null_build_preserves_missing_column_error(polars_engine):
    pl = pytest.importorskip("polars")
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.build import _non_null_id_rows

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        _non_null_id_rows(pl.DataFrame({"a": [1]}), ("missing",), Engine(polars_engine), np)
