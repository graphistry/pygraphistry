import logging
import pandas as pd
from typing import Any, List, Union, TYPE_CHECKING
from typing_extensions import Literal
from collections import Counter

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.constants import CUML, UMAP_LEARN  # noqa type: ignore

logger = logging.getLogger("compute.cluster")

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


def lazy_dbscan_import_has_dependency():
    has_min_dependency = True
    DBSCAN = None
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        has_min_dependency = False
        logger.warning('Please install sklearn for CPU DBSCAN')

    has_cuml_dependency = True
    cuDBSCAN = None
    try:
        from cuml import DBSCAN as cuDBSCAN
    except ImportError:
        has_cuml_dependency = False
        logger.warning('Please install cuml for GPU DBSCAN')
    
    return has_min_dependency, DBSCAN, has_cuml_dependency, cuDBSCAN

    
    
def get_umap_embedding_df(g, kind='nodes'):
    """
        Returns a dataframe with the UMAP embeddings from the graphistry graph
    """
    if kind == 'nodes':
        df = g._node_embedding
    elif kind == 'edges':
        df = g._edge_embedding
    else:
        raise ValueError('kind must be one of nodes or edges')
        
    return df
    
def cluster(g, dbscan, kind='nodes', cols=None, umap=True):
    """
        Fits clustering on UMAP embeddings if umap is True, otherwise on the features dataframe
        
        args:
            g: graphistry graph
            kind: 'nodes' or 'edges'
            cols: list of columns to use for clustering given `g.featurize` has been run
            umap: whether to use UMAP embeddings or features dataframe
    """
    
    if cols is None:
        df = g._get_feature(kind)
    else: 
        df = g.get_features_by_cols(cols, kind)

    if umap and cols is None: 
        df = get_umap_embedding_df(g, kind)
    
    print(df.head())

    
    dbscan.fit(df)
    labels = dbscan.labels_
    
    if kind == 'nodes':    
        g._nodes['_cluster'] = labels
    elif kind == 'edges':
        g._edges['_cluster'] = labels
    else:
        raise ValueError('kind must be one of `nodes` or `edges`')
    
    kind = 'node' if kind == 'nodes' else 'edge'
    setattr(g, f'_{kind}_dbscan', dbscan)
    
    return g

class ClusterMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass
    
    def _cluster_dbscan(self, res, kind, cols, umap, eps, min_samples, **kwargs):
        """
            DBSCAN clustering on cpu or gpu infered by umap's .engine flag
        """
        _, DBSCAN, _, cuDBSCAN = lazy_dbscan_import_has_dependency()
        
        dbscan = cuDBSCAN(eps=eps, min_samples=min_samples, **kwargs) if self.engine == CUML else DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        res = cluster(res, dbscan, kind=kind, cols=cols, umap=umap)

        return res
    
    
    def dbscan(self, kind = 'nodes', cols = None, umap = True, eps: float = 1., min_samples: int = 1, **kwargs):
        """DBSCAN clustering on cpu or gpu infered by umap's .engine flag
        
        Args:
            kind: 'nodes' or 'edges'
            cols: list of columns to use for clustering given `g.featurize` has been run, nice way to slice features by fragments of interest, e.g. ['ip172', 'location', 'asn', 'warnings']
            umap: whether to use UMAP embeddings or features dataframe
        """
        res = self.bind()
        res = res._cluster_dbscan(res, kind=kind, cols=cols, umap=umap, eps=eps, min_samples=min_samples, **kwargs)
        
        return res

    def _is_cudf(self, df):
        return 'cudf' in str(type(df))
