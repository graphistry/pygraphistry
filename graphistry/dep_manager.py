import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, List, TYPE_CHECKING, Any, Tuple

### umap_utils lazy
def lazy_umap_import_has_dependancy():
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import umap  # noqa
        return True, "ok", umap
    except ModuleNotFoundError as e:
        return False, e, None

def lazy_cuml_import_has_dependancy():
    try:
        import warnings
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import cuml  # type: ignore
        return True, "ok", cuml
    except ModuleNotFoundError as e:
        return False, e, None

def lazy_cudf_import_has_dependancy():
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import cudf  # type: ignore
        return True, "ok", cudf
    except ModuleNotFoundError as e:
        return False, e, None

def is_legacy_cuml():
    try:
        import cuml
        vs = cuml.__version__.split(".")
        if (vs[0] in ["0", "21"]) or (vs[0] == "22" and float(vs[1]) < 6):
            return True
        else:
            return False
    except ModuleNotFoundError:
        return False


### feature_utils lazy
def lazy_import_has_dependancy_text():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from sentence_transformers import SentenceTransformer
        return True, 'ok', SentenceTransformer
    except ModuleNotFoundError as e:
        return False, e, None

def lazy_import_has_min_dependancy():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import scipy.sparse  # noqa
        from scipy import __version__ as scipy_version
        from dirty_cat import __version__ as dirty_cat_version
        from sklearn import __version__ as sklearn_version
        logger.debug(f"SCIPY VERSION: {scipy_version}")
        logger.debug(f"Dirty CAT VERSION: {dirty_cat_version}")
        logger.debug(f"sklearn VERSION: {sklearn_version}")
        return True, 'ok'
    except ModuleNotFoundError as e:
        return False, e


### embed_utils lazy
def lazy_embed_import_dep():
    try:
        import torch
        import torch.nn as nn
        import dgl
        from dgl.dataloading import GraphDataLoader
        import torch.nn.functional as F
        from .networks import HeteroEmbed
        from tqdm import trange
        return True, torch, nn, dgl, GraphDataLoader, HeteroEmbed, F, trange
    except:
        return False, None, None, None, None, None, None, None

def check_cudf():
    try:
        import cudf
        return True, cudf
    except:
        return False, object


### cluster lazy
def lazy_dbscan_import_has_dependency():
    has_min_dependency = True
    DBSCAN = None
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        has_min_dependency = False
        logger.info("Please install sklearn for CPU DBSCAN")
    has_cuml_dependency = True
    cuDBSCAN = None
    try:
        from cuml import DBSCAN as cuDBSCAN
    except ImportError:
        has_cuml_dependency = False
        logger.info("Please install cuml for GPU DBSCAN")

    return has_min_dependency, DBSCAN, has_cuml_dependency, cuDBSCAN

def lazy_cudf_import_has_dependancy():
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import cudf  # type: ignore
        return True, "ok", cudf
    except ModuleNotFoundError as e:
        return False, e, None


### dgl_utils lazy
def lazy_dgl_import_has_dependency():
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import dgl  # noqa: F811
        return True, 'ok', dgl
    except ModuleNotFoundError as e:
        return False, e, None

def lazy_torch_import_has_dependency():
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import torch  # noqa: F811
        return True, 'ok', torch
    except ModuleNotFoundError as e:
        return False, e, None


### networks lazy
def lazy_dgl_import_has_dependency():
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import dgl  # noqa: F811
        return True, 'ok', dgl
    except ModuleNotFoundError as e:
        return False, e, None

def lazy_torch_import_has_dependency():
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import torch  # noqa: F811
        return True, 'ok', torch
    except ModuleNotFoundError as e:
        return False, e, None


