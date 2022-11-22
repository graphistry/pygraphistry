import pyarrow as pa, requests, sys
from functools import lru_cache
from typing import Any, Tuple, Optional
from weakref import WeakKeyDictionary
from .util import setup_logger
logger = setup_logger(__name__)


# WrappedTable -> {'file_id': str, 'output': dict}
DF_TO_FILE_ID_CACHE : WeakKeyDictionary = WeakKeyDictionary()
"""
NOTE: Will switch to pa.Table -> ... when RAPIDS upgrades from pyarrow, 
     which adds weakref support
"""

class ArrowFileUploader():
    """
        Implement file API with focus on Arrow support

        Memoization in this class is based on reference equality, while plotter is based on hash.
        That means the plotter resolves different-identity value matches, so by the time ArrowFileUploader compares,
        identities are unified for faster reference-based checks.

        Example: Upload files with per-session memoization
            uploader : ArrowUploader
            arr : pa.Table
            afu = ArrowFileUploader(uploader)

            file1_id = afu.create_and_post_file(arr)[0]
            file2_id = afu.create_and_post_file(arr)[0]

            assert file1_id == file2_id # memoizes by default (memory-safe: weak refs)

        Example: Explicitly create a file and upload data for it
            uploader : ArrowUploader
            arr : pa.Table
            afu = ArrowFileUploader(uploader)

            file1_id = afu.create_file()
            afu.post_arrow(arr, file_id)

            file2_id = afu.create_file()
            afu.post_arrow(arr, file_id)

            assert file1_id != file2_id

    """

    uploader : Any = None  # ArrowUploader, circular

    def __init__(self, uploader):  # ArrowUploader
        self.uploader = uploader

    ###

    def create_file(self, file_opts: dict = {}) -> str:
        """
            Creates File and returns file_id str.
            
            Defauls:
              - file_type: 'arrow'

            See File REST API for file_opts

        """

        tok = self.uploader.token

        json_extended = {
            'file_type': 'arrow',
            'agent_name': 'pygraphistry',
            'agent_version': sys.modules['graphistry'].__version__,  # type: ignore
            **file_opts
        }

        res = requests.post(
            self.uploader.server_base_path + '/api/v2/files/',
            verify=self.uploader.certificate_validation,
            headers={'Authorization': f'Bearer {tok}'},
            json=json_extended)

        try:
            out = res.json()
            logger.debug('Server create file response: %s', out)
            if res.status_code != requests.codes.ok:
                res.raise_for_status()
        except Exception as e:
            logger.error('Failed creating file: %s', res.text, exc_info=True)
            raise e
        
        return out['file_id']

    def post_arrow(self, arr: pa.Table, file_id: str, url_opts: str = 'erase=true') -> dict:
        """
            Upload new data to existing file id

            Default url_opts='erase=true' throws exceptions on parse errors and deletes upload.

            See File REST API for url_opts (file upload)
        """

        sub_path = f'api/v2/upload/files/{file_id}'
        tok = self.uploader.token

        res = self.uploader.post_arrow_generic(sub_path, tok, arr, url_opts)

        try:
            out = res.json()
            logger.debug('Server upload file response: %s', out)
            if not out['is_valid']:
                if out['is_uploaded']:
                    raise RuntimeError("Uploaded file contents but cannot parse (file_id still valid), see errors", out['errors'])
                else:
                    raise RuntimeError("Erased uploaded file contents upon failure (file_id still valid), see errors", out['errors'])
            return out
        except Exception as e:
            logger.error('Failed uploading file: %s', res.text, exc_info=True)
            raise e

    ###

    def create_and_post_file(
        self, arr: pa.Table, file_id: Optional[str] = None, file_opts: dict = {}, upload_url_opts: str = 'erase=true', memoize: bool = True
    ) -> Tuple[str, dict]:
        """
            Create file and upload data for it.

            Default upload_url_opts='erase=true' throws exceptions on parse errors and deletes upload.

            Default memoize=True skips uploading 'arr' when previously uploaded in current session

            See File REST API for file_opts (file create) and upload_url_opts (file upload)
        """

        if memoize:
            #FIXME if pa.Table was hashable, could do direct set/get map
            wrapped_table : WrappedTable
            val : MemoizedFileUpload
            for wrapped_table, val in DF_TO_FILE_ID_CACHE.items():
                if wrapped_table.arr is arr:
                    logger.debug('arrow->file_id memoization hit: %s', val.file_id)
                    return val.file_id, val.output
            logger.debug('arrow->file_id memoization miss (of %s)', len(DF_TO_FILE_ID_CACHE))

        if file_id is None:
            file_id = self.create_file(file_opts)
        
        resp = self.post_arrow(arr, file_id, upload_url_opts)
        out = MemoizedFileUpload(file_id, resp)

        if memoize:
            wrapped = WrappedTable(arr)
            cache_arr(wrapped)
            DF_TO_FILE_ID_CACHE[wrapped] = out
            logger.debug('Memoized arrow->file_id %s', file_id)
        
        return out.file_id, out.output

@lru_cache(maxsize=100)
def cache_arr(arr):
    """
        Hold reference to most recent memoization entries
        Hack until RAPIDS supports Arrow 2.0, when pa.Table becomes weakly referenceable
    """
    return arr

class WrappedTable():
    arr : pa.Table
    def __init__(self, arr: pa.Table):
        self.arr = arr

class MemoizedFileUpload():    
    file_id: str
    output: dict
    def __init__(self, file_id: str, output: dict):
        self.file_id = file_id
        self.output = output
