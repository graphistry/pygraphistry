import pandas as pd, pyarrow as pa, unittest

from graphistry.arrow_uploader import ArrowUploader
from graphistry.ArrowFileUploader import (
    ArrowFileUploader,
    MemoizedFileUpload,
    _ID_CACHE,
    _SIG_CACHE,
    _CACHE_LOCK,
    _compute_signature,
)

# TODO mock requests for testing actual effectful code


class TestArrowFileUploader_Core(unittest.TestCase):
    def test_memoization(self):

        afu = ArrowFileUploader(ArrowUploader(token="xx"))

        arr = pa.Table.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

        # Manually populate the cache with test data
        with _CACHE_LOCK:
            obj_id = id(arr)
            sig = _compute_signature(arr)
            memo = MemoizedFileUpload("a", sig, {"test": "output"})
            _ID_CACHE[obj_id] = memo
            _SIG_CACHE[sig] = memo

        assert afu.create_and_post_file(arr) == ("a", {"test": "output"})
