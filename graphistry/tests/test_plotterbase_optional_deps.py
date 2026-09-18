"""Pins the optional-dependency contract of PlotterBase's maybe_* helpers.

Engine dispatch relies on each returning None when its library is absent rather than
raising. The helpers are lru_cached, so each test clears the cache around itself.
"""

import sys

import pytest

from graphistry.PlotterBase import (
    maybe_cudf,
    maybe_dask_cudf,
    maybe_dask_dataframe,
    maybe_polars,
    maybe_spark,
)

HELPERS = [
    (maybe_cudf, "cudf"),
    (maybe_dask_cudf, "dask_cudf"),
    (maybe_dask_dataframe, "dask.dataframe"),
    (maybe_spark, "pyspark"),
    (maybe_polars, "polars"),
]


@pytest.fixture
def blocked(monkeypatch):
    """Make a module unimportable for the duration of one test."""
    def _block(name: str) -> None:
        root = name.split(".")[0]
        for key in [k for k in sys.modules if k == root or k.startswith(root + ".")]:
            monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.setitem(sys.modules, root, None)
    return _block


@pytest.mark.parametrize("helper,module", HELPERS, ids=[m for _, m in HELPERS])
def test_helper_returns_none_when_its_library_is_absent(helper, module, blocked):
    helper.cache_clear()
    try:
        blocked(module)
        assert helper() is None
    finally:
        helper.cache_clear()


@pytest.mark.parametrize("helper,module", HELPERS, ids=[m for _, m in HELPERS])
def test_helper_returns_the_module_when_its_library_is_present(helper, module):
    pytest.importorskip(module)
    helper.cache_clear()
    try:
        assert helper() is not None
    finally:
        helper.cache_clear()


def test_the_result_is_cached(monkeypatch):
    """lru_cache is load-bearing: these are called per engine-dispatch decision."""
    maybe_cudf.cache_clear()
    try:
        first = maybe_cudf()
        assert maybe_cudf() is first
        assert maybe_cudf.cache_info().hits >= 1
    finally:
        maybe_cudf.cache_clear()


class _LenRaises:
    """Stands in for an edges table whose length cannot be taken."""

    def __len__(self) -> int:
        raise RuntimeError("length unavailable")


def test_make_arrow_dataset_survives_an_edges_table_whose_length_raises():
    """The empty-graph warning is advisory; it must never be able to fail an upload."""
    import graphistry
    from graphistry.PlotterBase import PlotterBase

    g = graphistry.bind()
    assert isinstance(g, PlotterBase)
    au = g._make_arrow_dataset(
        edges=_LenRaises(),  # type: ignore[arg-type]
        nodes=None,
        name="n",
        description="d",
        metadata=None,
    )
    assert au is not None


def test_plot_dispatch_continues_when_label_inference_raises(monkeypatch):
    """infer_labels is a convenience; a failure in it must not break the upload path."""
    import pandas as pd

    import graphistry
    from graphistry.PlotterBase import PlotterBase

    calls = []

    def boom(self):
        calls.append(1)
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(PlotterBase, "infer_labels", boom, raising=True)

    edges = pd.DataFrame({"s": ["a"], "d": ["b"]})
    nodes = pd.DataFrame({"n": ["a", "b"], "lbl": ["x", "y"]})
    g = graphistry.edges(edges, "s", "d").nodes(nodes, "n")

    au = g._plot_dispatch(
        graph=edges,
        nodes=nodes,
        name="n",
        description="d",
        metadata=None,
        memoize=False,
    )

    assert calls, "the test did not reach infer_labels, so it proves nothing"
    assert au is not None
