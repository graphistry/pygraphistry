
import mock, pandas as pd, pyarrow as pa, pytest, unittest

import graphistry
from common import NoAuthTestCase
from graphistry.arrow_uploader import ArrowUploader
from graphistry.ArrowFileUploader import ArrowFileUploader, DF_TO_FILE_ID_CACHE, MemoizedFileUpload, WrappedTable, cache_arr

#TODO mock requests for testing actual effectful code

class TestArrowFileUploader_Core(unittest.TestCase):
    def test_memoization(self):

        afu = ArrowFileUploader(ArrowUploader(token='xx'))

        arr = pa.Table.from_pandas(pd.DataFrame({'x': [1,2,3]}))

        #avoid directly holding references
        DF_TO_FILE_ID_CACHE[ cache_arr(WrappedTable(arr)) ] = MemoizedFileUpload('a', 'b')

        assert afu.create_and_post_file(arr) == ('a', 'b')