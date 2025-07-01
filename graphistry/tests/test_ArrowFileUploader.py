import concurrent.futures as cf
import itertools
import time
import types
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from graphistry.ArrowFileUploader import (
    ArrowFileUploader,
    _CACHE,
    _CACHE_LOCK,
    _hash_metadata,
    _hash_full_table,
)


@pytest.fixture
def afu():
    """Fresh ArrowFileUploader with deterministic file-id generator."""
    counter = itertools.count(1)

    stub_uploader = types.SimpleNamespace(
        token="dummy",
        server_base_path="https://example",
        certificate_validation=False,
        _post_arrow_generic=lambda *a, **k: None,  # never reached
    )

    obj = ArrowFileUploader(stub_uploader)
    obj.create_file = mock.Mock(side_effect=lambda *_, **__: f"file-{next(counter)}")
    obj._post_arrow = mock.Mock(return_value=None)

    yield obj

    # isolate global cache between tests
    with _CACHE_LOCK:
        _CACHE.clear()



def test_first_upload_returns_new_id(afu):
    arr = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

    fid = afu.create_and_post_file(arr)
    assert fid.startswith("file-")
    afu.create_file.assert_called_once()
    afu._post_arrow.assert_called_once_with(arr, fid, "erase=true")


def test_second_equal_upload_hits_cache(afu):
    arr1 = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    arr2 = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

    fid1 = afu.create_and_post_file(arr1)
    fid2 = afu.create_and_post_file(arr2)

    assert fid1 == fid2               # same cached ID
    assert afu.create_file.call_count == 1
    assert afu._post_arrow.call_count == 1


def test_metadata_collision_triggers_second_upload(afu):
    arr_a = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(3)}))
    arr_b = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(3) + 10}))

    fid_a = afu.create_and_post_file(arr_a)
    fid_b = afu.create_and_post_file(arr_b)

    assert fid_a != fid_b             # different full hash ⇒ new upload
    assert afu._post_arrow.call_count == 2



def test_parallel_hits_warmed_cache(afu):
    arr = pa.Table.from_pandas(pd.DataFrame({"x": list(range(1000))}))

    warm_id = afu.create_and_post_file(arr)
    assert afu._post_arrow.call_count == 1

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(afu.create_and_post_file, [arr] * 40))

    assert all(fid == warm_id for fid in results)
    assert afu._post_arrow.call_count == 1  # no extra uploads


def test_hash_stability():
    arr = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    assert _hash_metadata(arr) == _hash_metadata(arr)
    assert _hash_full_table(arr) == _hash_full_table(arr)
