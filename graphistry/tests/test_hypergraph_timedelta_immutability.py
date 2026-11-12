"""GPU timedelta coercion must not rewrite the caller's event frame (#2291)."""

import os

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.hyper_dask import HyperBindings, clean_events


pytestmark = pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="requires TEST_CUDF=1")


@pytest.mark.parametrize("engine", [Engine.CUDF, Engine.DASK_CUDF])
@pytest.mark.parametrize("entrypoint", ["clean_events", "hypergraph"])
@pytest.mark.parametrize("has_event_id", [True, False])
def test_timedelta_coercion_preserves_caller(engine, entrypoint, has_event_id):
    cudf = pytest.importorskip("cudf")
    original = pd.DataFrame({"person": ["a", "b", "c", "d"], "duration": pd.to_timedelta([1, 2, 3, 4], unit="s")})
    if has_event_id:
        original = original.assign(eid=[10, 20, 30, 40])
    events = cudf.from_pandas(original)
    if engine == Engine.DASK_CUDF:
        dask_cudf = pytest.importorskip("dask_cudf")
        events = dask_cudf.from_cudf(events, npartitions=2)
    if entrypoint == "clean_events":
        result = clean_events(events, HyperBindings(EVENTID="eid"), engine)
    else:
        result = graphistry.hypergraph(events, entity_types=["person"], opts={"EVENTID": "eid"}, engine=engine)["events"]
    if engine == Engine.DASK_CUDF:
        actual_input, actual_result = events.compute(), result.compute()
    else:
        actual_input, actual_result = events, result
    pd.testing.assert_frame_equal(actual_input.to_pandas(), original)
    assert actual_result["duration"].dtype.name != "timedelta64[ns]"
    assert len(actual_result) == len(original)
