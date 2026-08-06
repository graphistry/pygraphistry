from typing import Any, Optional, Tuple
import warnings
from graphistry .util import setup_logger
logger = setup_logger(__name__)


#TODO use new importer when it lands (this is copied from umap_utils)
def lazy_cudf_import():
    try:
        warnings.filterwarnings("ignore")
        import cudf  # type: ignore

        # cudf >= 26.02 removed DataFrame.from_pandas() and Series.from_pandas().
        # Restore them so existing call sites keep working across RAPIDS versions.
        # TODO(rapids-compat): migrate call sites to cudf.from_pandas() and remove shim
        if not hasattr(cudf.DataFrame, 'from_pandas'):
            cudf.DataFrame.from_pandas = staticmethod(cudf.from_pandas)
        if not hasattr(cudf.Series, 'from_pandas'):
            cudf.Series.from_pandas = staticmethod(cudf.from_pandas)

        return True, "ok", cudf
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn("Unexpected exn during lazy import", exc_info=e)
        return False, e, None

_CUPY_COMPUTE_OK = None


def lazy_cupy_import() -> Tuple[bool, Any, Optional[Any]]:  # hygiene-ok: explicit-any -- reason is exn-or-str; cupy module untyped
    """(available, reason, cupy) -- available ONLY when cupy can actually COMPUTE.

    cupy imports fine on a host whose CUDA install lacks NVRTC (``libnvrtc.so``),
    but then essentially EVERY compute op -- elementwise arithmetic, comparisons,
    astype, sort/search/bincount -- raises ``RuntimeError`` at first use (only
    allocation and a few cub-backed reductions survive). An ``except ImportError``
    guard cannot catch that, so callers holding a CPU fallback must gate on THIS
    probe (one tiny elementwise op, cached per process), not on importability.
    """
    global _CUPY_COMPUTE_OK
    try:
        warnings.filterwarnings("ignore")
        import cupy  # type: ignore
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn("Unexpected exn during lazy cupy import", exc_info=e)
        return False, e, None
    if _CUPY_COMPUTE_OK is None:
        try:
            (cupy.arange(2) + 1).sum()
            _CUPY_COMPUTE_OK = (True, "ok")
        except Exception as e:  # RuntimeError on NVRTC-less CUDA installs
            _CUPY_COMPUTE_OK = (False, str(e))
    ok, reason = _CUPY_COMPUTE_OK
    return ok, reason, (cupy if ok else None)


class CudfRuntimeCaps:
    """One-stop capability answer for the cudf stack (see cudf_runtime_caps)."""
    __slots__ = ("has_cudf", "cudf_reason", "cudf", "has_cupy_compute", "cupy_reason", "cupy")

    def __init__(self, has_cudf: bool, cudf_reason: Any, cudf: Optional[Any],  # hygiene-ok: explicit-any -- reasons are exn-or-str; cudf/cupy modules untyped
                 has_cupy_compute: bool, cupy_reason: Any, cupy: Optional[Any]) -> None:  # hygiene-ok: explicit-any -- reasons are exn-or-str; cudf/cupy modules untyped
        self.has_cudf = has_cudf
        self.cudf_reason = cudf_reason
        self.cudf = cudf
        self.has_cupy_compute = has_cupy_compute
        self.cupy_reason = cupy_reason
        self.cupy = cupy


def cudf_runtime_caps() -> CudfRuntimeCaps:
    """The question typical cudf-path code should ask, answered once.

    ``has_cudf``: the dataframe engine is importable (precompiled libcudf ops --
    construct/filter/merge/groupby -- run even on NVRTC-less CUDA installs).
    ``has_cupy_compute``: the cupy ARRAY sidecar can actually compute (kernel
    JIT works); False on NVRTC-less installs, where consumers holding a host
    fallback should take it. Encapsulates the cudf-vs-cupy split so call sites
    do not juggle two gates; ``lazy_cupy_import`` remains the low-level probe.
    """
    has_cudf, cudf_reason, cudf = lazy_cudf_import()
    has_cupy, cupy_reason, cupy = lazy_cupy_import()
    return CudfRuntimeCaps(has_cudf, cudf_reason, cudf, has_cupy, cupy_reason, cupy)


def lazy_cuml_import():
    try:
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import cuml  # type: ignore

        return True, "ok", cuml
    except ModuleNotFoundError as e:
        return False, e, None
    except ImportError as e:
        # Catch ImportError for broken library dependencies (e.g., RMM)
        logger.debug("cuML import failed with ImportError: %s", e)
        return False, e, None
    except Exception as e:
        logger.warn("Unexpected exn during lazy import", exc_info=e)
        return False, e, None

def lazy_dbscan_import():
    has_min_dependency = True
    DBSCAN = None
    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError:
        has_min_dependency = False
        logger.info("Please install sklearn for CPU DBSCAN")
    except Exception as e:
        logger.warn("Unexpected exn during lazy import", exc_info=e)
        return False, None, False, None

    has_cuml_dependency = True
    cuDBSCAN = None
    try:
        from cuml import DBSCAN as cuDBSCAN
    except ModuleNotFoundError:
        has_cuml_dependency = False
        logger.info("Please install cuml for GPU DBSCAN")
    except ImportError as e:
        # Catch ImportError for broken library dependencies (e.g., RMM)
        has_cuml_dependency = False
        logger.debug("cuML DBSCAN import failed with ImportError: %s", e)
    except Exception as e:
        has_cuml_dependency = False
        logger.warn("Unexpected exn during lazy import", exc_info=e)

    return has_min_dependency, DBSCAN, has_cuml_dependency, cuDBSCAN

def lazy_dgl_import():
    try:
        warnings.filterwarnings('ignore')
        import dgl  # noqa: F811
        return True, 'ok', dgl
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn("Unexpected exn during lazy import", exc_info=e)
        return False, e, None

def lazy_dirty_cat_import():
    warnings.filterwarnings("ignore")
    try:
        import dirty_cat 
        return True, 'ok', dirty_cat
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

def lazy_skrub_import():
    warnings.filterwarnings("ignore")
    try:
        import skrub 
        return True, 'ok', skrub
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

def lazy_embed_import():
    try:
        import torch
        import torch.nn as nn
        import dgl
        from dgl.dataloading import GraphDataLoader
        import torch.nn.functional as F
        from graphistry.networks import HeteroEmbed
        from tqdm import trange
        return True, torch, nn, dgl, GraphDataLoader, HeteroEmbed, F, trange
    except ModuleNotFoundError:
        return False, None, None, None, None, None, None, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, None, None, None, None, None, None, None

def lazy_networks_import():  # noqa
    try:
        import dgl
        import dgl.nn as dglnn
        import dgl.function as fn
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        Module = nn.Module
        return nn, dgl, dglnn, fn, torch, F, Module
    except ModuleNotFoundError:
        return None, None, None, None, None, None, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return None, None, None, None, None, None, None

def lazy_torch_import_has_dependency():
    try:
        warnings.filterwarnings('ignore')
        import torch  # noqa: F811
        return True, 'ok', torch
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

def lazy_umap_import():
    try:
        warnings.filterwarnings("ignore")
        import umap  # noqa

        return True, "ok", umap
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

#@check_set_memoize
def lazy_sentence_transformers_import():
    warnings.filterwarnings("ignore")
    try:
        from sentence_transformers import SentenceTransformer
        return True, 'ok', SentenceTransformer
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

def lazy_import_has_min_dependancy():
    warnings.filterwarnings("ignore")
    try:
        import scipy.sparse  # noqa
        from scipy import __version__ as scipy_version
        from sklearn import __version__ as sklearn_version
        logger.debug(f"SCIPY VERSION: {scipy_version}")
        logger.debug(f"sklearn VERSION: {sklearn_version}")
        return True, 'ok'
    except ModuleNotFoundError as e:
        return False, e
    except Exception as e:
        logger.warn('Unexpected exn during lazy import', exc_info=e)
        return False, e, None

def assert_imported_text():
    has_dependancy_text_, import_text_exn, _ = lazy_sentence_transformers_import()
    if not has_dependancy_text_:
        logger.error(  # noqa
            "AI Package sentence_transformers not found,"
            "trying running `pip install graphistry[ai]`"
        )
        raise import_text_exn

def assert_imported():
    has_min_dependancy_, import_min_exn = lazy_import_has_min_dependancy()
    if not has_min_dependancy_:
        logger.error(  # noqa
                     "AI Packages not found, trying running"  # noqa
                     "`pip install graphistry[ai]`"  # noqa
        )
        raise import_min_exn
