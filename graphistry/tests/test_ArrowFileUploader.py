import concurrent.futures as cf
import itertools
import time
import types
from unittest import mock

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pytest

from graphistry.ArrowFileUploader import (
    ArrowFileUploader,
    _CACHE,
    _CACHE_LOCK,
    _hash_metadata,
    _hash_full_table,
)


# Fixture – ArrowFileUploader with mocked IO
# ------------------------------------------------------------------ #

@pytest.fixture
def afu():
    counter = itertools.count(1)

    stub = types.SimpleNamespace(
        token="dummy",
        server_base_path="https://example",
        certificate_validation=False,
        post_arrow_generic=lambda *a, **k: None,
    )

    obj = ArrowFileUploader(stub)
    obj.create_file = mock.Mock(side_effect=lambda *_, **__: f"file-{next(counter)}")
    obj.post_arrow = mock.Mock(return_value={"mock": "resp"})
    yield obj

    with _CACHE_LOCK:
        _CACHE.clear()


# Functional behaviour
# ------------------------------------------------------------------ #

def test_first_upload_returns_new_id_and_response(afu):
    arr = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    fid, resp = afu.create_and_post_file(arr)
    assert fid.startswith("file-") and resp == {"mock": "resp"}
    afu.create_file.assert_called_once()
    afu.post_arrow.assert_called_once_with(arr, fid, "erase=true")


def test_second_equal_upload_hits_cache(afu):
    df = pd.DataFrame({"x": [1, 2, 3]})
    arr1 = pa.Table.from_pandas(df)
    arr2 = pa.Table.from_pandas(df)
    
    shallow_hash_1 = _hash_metadata(arr1)
    shallow_hash_2 = _hash_metadata(arr2)
    assert shallow_hash_1 == shallow_hash_2

    full_hash_1 = _hash_full_table(arr1)
    full_hash_2 = _hash_full_table(arr2)
    assert full_hash_1 != full_hash_2
    
    r1 = afu.create_and_post_file(arr1)
    r2 = afu.create_and_post_file(arr2)


    assert r1 == r2
    assert afu.create_file.call_count == 1
    assert afu.post_arrow.call_count == 1


def test_metadata_collision_triggers_second_upload(afu):
    arr_a = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(3)}))
    arr_b = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(3) + 10}))
    fid_a, _ = afu.create_and_post_file(arr_a)
    fid_b, _ = afu.create_and_post_file(arr_b)
    assert fid_a != fid_b
    assert afu.post_arrow.call_count == 2


# Deep vs. shallow hash check
# ------------------------------------------------------------------ #

def test_midrow_change_bypasses_shallow_hash(afu):
    base_df = pd.DataFrame({"x": [0, 1, 2, 3, 4]})
    arr1 = pa.Table.from_pandas(base_df)
    fid1, _ = afu.create_and_post_file(arr1)

    mod_df = base_df.copy()
    mod_df.loc[2, "x"] = 99  # change a middle row: first/last unchanged
    arr2 = pa.Table.from_pandas(mod_df)

    # Shallow (metadata) hash identical; deep hash differs
    assert _hash_metadata(arr1) == _hash_metadata(arr2)
    assert _hash_full_table(arr1) != _hash_full_table(arr2)

    fid2, _ = afu.create_and_post_file(arr2)
    assert fid2 != fid1              # new upload required
    assert afu.post_arrow.call_count == 2


# Performance guard – large table
# ------------------------------------------------------------------ #

def test_large_table_hash_fast(afu):
    mb = 1024 ** 2
    target = 50 * mb
    rows = target // 8
    arr = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(rows, dtype="int64")}))
    before = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    fid, resp = afu.create_and_post_file(arr)
    elapsed = time.perf_counter() - t0
    after = psutil.Process().memory_info().rss
    assert fid.startswith("file-") and resp == {"mock": "resp"}
    assert elapsed < 3
    assert after - before < 5 * mb


# Hash helper idempotence
# ------------------------------------------------------------------ #

def test_hash_stability():
    arr = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    assert _hash_metadata(arr) == _hash_metadata(arr)
    assert _hash_full_table(arr) == _hash_full_table(arr)
