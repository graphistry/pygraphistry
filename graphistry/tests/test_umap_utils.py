import sys
import types
from pathlib import Path

import pandas as pd

# Force imports to use the in-repo package rather than any preinstalled wheel
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.modules.pop("graphistry", None)

from graphistry import umap_utils  # noqa: E402


class _FakeCupyArray(list):
    """Simulates cupy.ndarray without supporting .get()."""

    def get(self):
        raise AssertionError("umap_graph_to_weighted_edges should not call device .get()")


class _DummyCoo:
    def __init__(self):
        self.row = _FakeCupyArray([0, 1])
        self.col = _FakeCupyArray([1, 0])
        self.data = _FakeCupyArray([0.5, 0.25])


class _DummyGraph:
    def tocoo(self):
        return _DummyCoo()


def test_cuml_path_avoids_get_and_uses_cudf(monkeypatch):
    """Ensure CUDA path builds cudf structures without calling deprecated COO.get() (#844)."""

    created_frames = []

    class _FakeSeries:
        def __init__(self, values):
            self.values = values

    def _dataframe(columns):
        converted = {k: _FakeSeries(v) for k, v in columns.items()}
        created_frames.append(converted)
        return {"_df": converted}

    fake_cudf = types.SimpleNamespace(Series=_FakeSeries, DataFrame=_dataframe)
    monkeypatch.setitem(sys.modules, "cudf", fake_cudf)
    fake_cupy = types.SimpleNamespace(
        ndarray=_FakeCupyArray,
        asnumpy=lambda arr: list(arr),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    result = umap_utils.umap_graph_to_weighted_edges(
        _DummyGraph(), engine="cuml", is_legacy=False, cfg=umap_utils.config
    )

    assert created_frames, "Expected cudf.DataFrame constructor to be invoked"
    cols = created_frames[0]
    assert cols[umap_utils.config.SRC].values == [0, 1]
    assert cols[umap_utils.config.DST].values == [1, 0]
    assert cols[umap_utils.config.WEIGHT].values == [0.5, 0.25]
    assert result["_df"] == cols


class _LegacyDeviceArray:
    def __init__(self, values):
        self._values = values

    def get(self):
        return self._values


class _LegacyCoo:
    """Simulates legacy cupy COO objects that still expose .get()."""

    def __init__(self):
        import numpy as np

        self.row = _LegacyDeviceArray(np.array([0, 1]))
        self.col = _LegacyDeviceArray(np.array([1, 0]))
        self.data = _LegacyDeviceArray(np.array([0.5, 0.25]))

    def get(self):
        return types.SimpleNamespace(
            row=self.row.get(),
            col=self.col.get(),
            data=self.data.get(),
        )


def test_cuml_path_supports_legacy_get(monkeypatch):
    """Verify backward compatibility when cupy.ndarray still provides .get()."""

    class _LegacyGraph:
        def tocoo(self):
            return _LegacyCoo()

    fake_cudf = types.SimpleNamespace(
        DataFrame=lambda cols: cols,
    )
    monkeypatch.setitem(sys.modules, "cudf", fake_cudf)
    # No cupy module injected so `_to_host` falls back to .get()

    result = umap_utils.umap_graph_to_weighted_edges(
        _LegacyGraph(), engine="cuml", is_legacy=False, cfg=umap_utils.config
    )

    assert list(result[umap_utils.config.SRC]) == [0, 1]
    assert list(result[umap_utils.config.DST]) == [1, 0]
    assert list(result[umap_utils.config.WEIGHT]) == [0.5, 0.25]


def test_umap_learn_path_returns_pandas():
    """engine='umap_learn' should remain pure pandas even after new compatibility code."""

    class _CpuGraph:
        def tocoo(self):
            class _CpuCoo:
                row = [0]
                col = [1]
                data = [0.1]

            return _CpuCoo()

    df = umap_utils.umap_graph_to_weighted_edges(
        _CpuGraph(), engine="umap_learn", is_legacy=False, cfg=umap_utils.config
    )
    expected = pd.DataFrame(
        {
            umap_utils.config.SRC: [0],
            umap_utils.config.DST: [1],
            umap_utils.config.WEIGHT: [0.1],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
