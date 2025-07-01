import sys, threading, hashlib
from typing import Any, Optional, Dict, Tuple
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import requests

from graphistry.utils.requests import log_requests_error
from .util import setup_logger

logger = setup_logger(__name__)

# metadata_hash -> { full_hash -> (response, file_id) }
_CACHE: Dict[int, Dict[int, Tuple[str, dict]]] = {}
_CACHE_LOCK = threading.RLock()
_MAX_SAMPLE_COLS = 20  # cap for cheap sampling


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

        * Returns a cached `file_id` when there is a hash match for the table,
          in the file_id cache.
        """
        md_hash = _hash_metadata(arr)

        bucket: Optional[Dict[int, Tuple[str, dict]]]
        with _CACHE_LOCK:
            bucket = _CACHE.get(md_hash)

        fh: Optional[int] = None
        if memoize and bucket is not None:
            fh = _hash_full_table(arr)

            with _CACHE_LOCK:
                cached = bucket.get(fh)
                if cached:
                    logger.debug("Memoisation hit (md=%s, full=%s)", md_hash, fh)
                    return cached

        # Fresh upload
        if file_id is None:
            file_id = self.create_file(file_opts)

        # Upload
        resp = self.post_arrow(arr, file_id, upload_url_opts)

        if memoize:
            fh = _hash_full_table(arr) if fh is None else fh
            with _CACHE_LOCK:
                _CACHE.setdefault(md_hash, {})[fh] = (file_id, resp)
                logger.debug("Memoised new upload (md=%s, full=%s)", md_hash, fh)

        return file_id, resp


def _hash_metadata(table: pa.Table, max_cols: int = _MAX_SAMPLE_COLS) -> int:
    """
    Fast, approximate 64-bit digest of *shape*:
        schema + metadata + col order + bytes + rows + sampled values
    """
    digest = hashlib.sha256()

    schema_str = str(table.schema)
    meta_items = tuple(sorted((table.schema.metadata or {}).items()))
    col_names = tuple(table.column_names)
    num_rows = table.num_rows

    # total bytes – cheap property in >=1.0, fallback otherwise
    if hasattr(table, "nbytes"):
        nbytes = table.nbytes
    else:
        nbytes = 0

    digest.update(schema_str.encode())
    digest.update(str(meta_items).encode())
    digest.update(str(col_names).encode())
    digest.update(str(num_rows).encode())
    digest.update(str(nbytes).encode())

    # sample first / last row values (bulk, not scalar loop)
    if num_rows:
        ncols = min(len(col_names), max_cols)
        for i in range(ncols):
            col = table.column(i)
            try:
                first_v = col.slice(0, 1).to_pylist()[0]
                last_v = col.slice(num_rows - 1, 1).to_pylist()[0]
            except Exception:
                first_v = last_v = None
            digest.update(str(first_v).encode())
            digest.update(str(last_v).encode())

    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _hash_full_table(table: pa.Table) -> int:
    """
    Precise 64-bit digest of the *entire* table.
    """
    digest = hashlib.sha256()

    # schema (captures types, nullability, field names, etc.)
    digest.update(str(table.schema).encode())

    # stream all buffers
    for column in table.columns:
        for chunk in column.chunks:
            for buf in chunk.buffers():
                if buf:
                    digest.update(buf)  # buffer protocol, zero‑copy

    return int.from_bytes(digest.digest()[:8], "big", signed=False)
