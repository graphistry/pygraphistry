from typing import Any
import warnings
from graphistry .util import setup_logger, check_set_memoize
logger = setup_logger(__name__)


#TODO use new importer when it lands (this is copied from umap_utils)
def lazy_cudf_import():
    try:
        warnings.filterwarnings("ignore")
        import cudf  # type: ignore

        return True, "ok", cudf
    except ModuleNotFoundError as e:
        return False, e, None
    except Exception as e:
        logger.warn("Unexpected exn during lazy import", exc_info=e)
        return False, e, None

def lazy_cuml_import():
    try:
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import cuml  # type: ignore

        return True, "ok", cuml
    except ModuleNotFoundError as e:
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
