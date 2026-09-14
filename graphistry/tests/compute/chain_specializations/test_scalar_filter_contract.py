"""The scalar filter contract accepts Python scalars and declines containers/nulls."""
import numpy as np
import pandas as pd
import pytest

from graphistry.compute.chain_fast_paths import _seeded_scalar_filters


class IntSubclass(int):
    pass


@pytest.mark.parametrize("value", [0, -1, 1.5, "x", True, False, IntSubclass(2)])
def test_scalar_filter_keeps_admitted_value_and_type(value):
    filters = {"value": value}
    result = _seeded_scalar_filters(filters, pd.DataFrame({"value": [1]}))
    assert result is not None
    assert result["value"] is value
    assert filters == {"value": value}


@pytest.mark.parametrize("value", [None, pd.NA, [1], (1,), {1}, {"a": 1}, np.int64(1), np.bool_(True)])
def test_scalar_filter_declines_outside_python_scalar_contract(value):
    assert _seeded_scalar_filters({"value": value}, pd.DataFrame({"value": [1]})) is None
