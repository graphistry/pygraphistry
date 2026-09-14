"""Probing for an optional dependency must not silence the caller's warnings."""
import os
import warnings

import pytest

from graphistry.utils.lazy_import import lazy_cudf_import, lazy_cuml_import


skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


@pytest.mark.parametrize('probe', [lazy_cudf_import, lazy_cuml_import])
def test_gpu_lazy_import_leaves_global_warning_filters_intact(probe) -> None:
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("always")
        before = list(warnings.filters)
        probe()
        assert warnings.filters == before


@skip_gpu
def test_cudf_probe_leaves_a_user_warning_deliverable() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        has_cudf, _, _ = lazy_cudf_import()
        assert has_cudf
        warnings.warn("still audible", UserWarning)
    assert [str(w.message) for w in rec] == ["still audible"]
