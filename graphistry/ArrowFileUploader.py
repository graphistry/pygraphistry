import sys, threading, weakref, hashlib
from functools import lru_cache
from typing import Any, Tuple, Optional, Dict

import pyarrow as pa
import requests

from graphistry.utils.requests import log_requests_error
from .util import setup_logger

logger = setup_logger(__name__)

class MemoizedFileUpload:
    __slots__ = ("file_id", "arrow_hash", "output")

    def __init__(self, file_id: str, arrow_hash: int, output: dict):
        self.file_id = file_id
        self.arrow_hash = arrow_hash
        self.output = output


_ID_CACHE: Dict[int, "MemoizedFileUpload"] = {}
_SIG_CACHE: Dict[int, "MemoizedFileUpload"] = {}

_CACHE_LOCK = threading.RLock()


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

    def __init__(self, uploader) -> None:
        self.uploader = uploader


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
        log_requests_error(res)

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
            logger.debug("Server upload file response: %s", out)
            if not out["is_valid"]:
                if out["is_uploaded"]:
                    raise RuntimeError(
                        "Uploaded file contents but cannot parse "
                        "(file_id still valid), see errors",
                        out["errors"],
                    )
                else:
                    raise RuntimeError(
                        "Erased uploaded file contents upon failure "
                        "(file_id still valid), see errors",
                        out["errors"],
                    )
            return out
        except Exception as e:
            logger.error("Failed uploading file: %s", res.text, exc_info=True)
            raise e

    ###

    def create_and_post_file(
        self,
        arr: pa.Table,
        file_id: Optional[str] = None,
        file_opts: dict = {},
        upload_url_opts: str = "erase=true",
        memoize: bool = True,
    ) -> Tuple[str, dict]:
        """
        Create a new file (unless `file_id` supplied) and upload `arr`.

        If `memoize` is True (default):

        * Returns a cached `(file_id, output)` when either
          * the exact same `pa.Table` object was uploaded earlier, or
          * any other `pa.Table` with identical **schema+metadata+columns**
            was uploaded earlier.
        """

        sig = 0  # default to no memoization
        if memoize:
            obj_id = id(arr)
            sig = _compute_signature(arr)
            with _CACHE_LOCK:
                cached = _ID_CACHE.get(obj_id) or _SIG_CACHE.get(sig)
                if cached:
                    logger.debug(
                        "Memoization hit (id=%s, sig=%s) → %s", obj_id, sig, cached.file_id
                    )
                    return cached.file_id, cached.output
            logger.debug("Memoization miss (cache size=%s)", len(_ID_CACHE))

        # Fresh upload
        if file_id is None:
            file_id = self.create_file(file_opts)

        resp = self.post_arrow(arr, file_id, upload_url_opts)
        memo = MemoizedFileUpload(file_id, sig, resp)

        if memoize:
            _memoize(arr, memo)

        return memo.file_id, memo.output



def _compute_signature(table: pa.Table) -> int:
    """
    Pure structural hash: schema, metadata, column order.
    Avoids storing the table itself.
    """
    schema_str = str(table.schema)
    meta_items = tuple(sorted((table.schema.metadata or {}).items()))
    col_names = tuple(table.column_names)
    # 64‑bit stable hash via sha1 → int
    sig_bytes = (
        schema_str.encode()
        + b"|"
        + str(meta_items).encode()
        + b"|"
        + str(col_names).encode()
    )
    return int.from_bytes(hashlib.sha1(sig_bytes).digest()[:8], "big", signed=False)


def _evict_id_cache(obj_id: int):
    """
    Called when the pa.Table is garbage-collected.
    Removes the cache entries to prevent collisions.
    """
    with _CACHE_LOCK:
        memo = _ID_CACHE.pop(obj_id, None)
        if memo:
            _SIG_CACHE.pop(memo.arrow_hash, None)
        


def _memoize(table: pa.Table, memo: "MemoizedFileUpload"):
    """
    Store both identity and value cache entries, and register GC evict hook.
    """
    obj_id = id(table)
    sig = _compute_signature(table)

    with _CACHE_LOCK:
        if obj_id not in _ID_CACHE:
            _ID_CACHE[obj_id] = memo
            _SIG_CACHE[sig] = memo
            weakref.finalize(table, _evict_id_cache, obj_id)
        logger.debug("Memoized: id=%s, sig=%s → %s", obj_id, sig, memo.file_id)
