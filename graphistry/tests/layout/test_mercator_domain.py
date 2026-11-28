"""mercator_layout latitude domain (#1967).

Mercator is undefined at the poles and outside [-90, 90]. In-range latitudes are clamped to
the Web-Mercator limit (+-85.051129), out-of-domain or non-numeric coordinates raise a
ValueError, and missing values stay missing. Positions are never infinite.
"""
import builtins
import os

import numpy as np
import pandas as pd
import pytest

import graphistry

# Latitude at which Web Mercator becomes a square map, ~85.051129
WEB_MERCATOR_MAX_LAT = float(np.degrees(np.arctan(np.sinh(np.pi))))
test_cudf = os.environ.get("TEST_CUDF") == "1"
EDGES = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})


def _run(lat, lon, frame=pd.DataFrame, edges=EDGES, **kwargs):
    nodes = frame({"id": [0, 1, 2], "lat": lat, "lon": lon})
    if frame is not pd.DataFrame:
        edges = frame(edges)
    g = graphistry.edges(edges, "s", "d").nodes(nodes, "id").bind(point_latitude="lat", point_longitude="lon")
    return g.mercator_layout(**kwargs)


def _y(out):
    nodes = out._nodes
    return np.asarray(nodes["y"].to_pandas() if hasattr(nodes["y"], "to_pandas") else nodes["y"], dtype=float)


def test_poles_are_finite_mirror_images_at_the_web_mercator_limit():
    y = _y(_run([90.0, -90.0, 0.0], [0.0, 0.0, 0.0]))
    limit = _y(_run([WEB_MERCATOR_MAX_LAT, -WEB_MERCATOR_MAX_LAT, 0.0], [0.0, 0.0, 0.0]))
    assert np.isfinite(y).all()
    assert np.isclose(y[0], -y[1], rtol=1e-12)
    assert np.allclose(y, limit)
    # The Web-Mercator limit is the latitude whose projection is a square: |y| == pi * R
    assert np.isclose(y[0], np.pi * 637.8137, rtol=1e-6)


def test_latitudes_beyond_the_limit_clamp_monotonically():
    y = _y(_run([85.0, 89.0, 90.0], [0.0, 0.0, 0.0]))
    assert np.isfinite(y).all()
    assert y[0] < y[1] == y[2]


@pytest.mark.parametrize("bad", [200.0, -90.5, 999.0, float("inf"), float("-inf")])
def test_out_of_domain_latitude_raises_naming_the_column(bad):
    with pytest.raises(ValueError, match=r"lat.*\[-90, 90\]"):
        _run([bad, 0.0, 0.0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_infinite_longitude_raises_naming_the_column(bad):
    with pytest.raises(ValueError, match=r"lon.*finite"):
        _run([0.0, 0.0, 0.0], [bad, 0.0, 0.0])


def test_missing_coordinates_stay_missing_without_failing():
    out = _run([37.7, float("nan"), 51.5], [-122.4, -74.0, float("nan")])
    x = out._nodes["x"].to_numpy()
    y = out._nodes["y"].to_numpy()
    assert np.isnan(y[1]) and np.isnan(x[2])
    assert np.isfinite(x[[0, 1]]).all() and np.isfinite(y[[0, 2]]).all()


def test_nullable_dtype_missing_values_stay_missing():
    lat = pd.array([37.7, None, 51.5], dtype="Float64")
    out = _run(lat, [-122.4, -74.0, -0.1])
    y = out._nodes["y"].to_numpy(dtype=float, na_value=np.nan)
    assert np.isnan(y[1]) and np.isfinite(y[[0, 2]]).all()


def test_non_numeric_coordinates_raise_a_value_error():
    with pytest.raises(ValueError, match=r"lat.*numeric"):
        _run(["37.7", "40.7", "51.5"], ["-122.4", "-74.0", "-0.1"])


def test_valid_input_is_unchanged():
    lat = np.array([37.7749, 40.7128, -33.8688])
    lon = np.array([-122.4194, -74.0060, 151.2093])
    out = _run(lat, lon, scale_for_graphistry=False)
    R = 6378137.0
    assert np.allclose(out._nodes["x"].to_numpy(), R * np.radians(lon), rtol=0, atol=1e-9)
    assert np.allclose(out._nodes["y"].to_numpy(), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)), rtol=0, atol=1e-6)


@pytest.mark.skipif(not test_cudf, reason="cudf tests need TEST_CUDF=1")
class TestCudf:

    def test_cudf_poles_clamp_and_out_of_domain_raises(self):
        import cudf
        y = _y(_run([90.0, -90.0, 0.0], [0.0, 0.0, 0.0], frame=cudf.DataFrame))
        assert np.isfinite(y).all() and np.isclose(y[0], -y[1], rtol=1e-12)
        with pytest.raises(ValueError, match=r"lat.*\[-90, 90\]"):
            _run([200.0, 0.0, 0.0], [0.0, 0.0, 0.0], frame=cudf.DataFrame)

    def test_cudf_matches_pandas_with_nulls(self):
        import cudf
        lat, lon = [37.7, None, 89.0], [-122.4, -74.0, -0.1]
        gpu = _y(_run(lat, lon, frame=cudf.DataFrame))
        cpu = _y(_run([37.7, float("nan"), 89.0], lon))
        assert np.allclose(gpu, cpu, equal_nan=True)

    def test_cudf_without_cupy_falls_back_to_numpy(self, monkeypatch):
        import cudf
        real_import = builtins.__import__

        def no_cupy(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("hidden for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_cupy)
        lat, lon = [37.7, 40.7, 90.0], [-122.4, -74.0, -0.1]
        gpu = _y(_run(lat, lon, frame=cudf.DataFrame))
        monkeypatch.setattr(builtins, "__import__", real_import)
        assert np.allclose(gpu, _y(_run(lat, lon)))
