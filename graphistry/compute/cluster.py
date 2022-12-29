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
    
    
def cluster(g, dbscan, kind='nodes'):
    """
        Fits clustering on UMAP embeddings
    """
    if kind=='nodes':
        df = g._node_embedding
    elif kind=='edges':
        df = g._edge_embedding
    else:
        raise ValueError('kind must be one of nodes or edges')
        
    dbscan.fit(df)
    labels = dbscan.labels_
    
    if kind=='nodes':    
        g._nodes['_cluster'] = labels
    elif kind=='edges':
        g._edges['_cluster'] = labels
    else:
        raise ValueError('kind must be one of nodes or edges')
    
    kind = 'node' if kind=='nodes' else 'edge'
    setattr(g, f'_{kind}_dbscan', dbscan)
    
    return g

class ClusterMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass
    
    def _cluster_dbscan(self, res, kind, eps, min_samples, **kwargs):
        """
            DBSCAN clustering on cpu or gpu infered by umap's .engine flag
        """
        if self.engine == UMAP_LEARN:
            try:
                from sklearn.cluster import DBSCAN
            except ImportError:
                raise ImportError('Please install sklearn')
            
        elif self.engine == CUML:
            try:
                from cuml import DBSCAN as cuDBSCAN
            except ImportError:
                raise ImportError('Please install cuml')
        else:
            raise ValueError(f'engine must be one of {UMAP_LEARN} or {CUML}')
        
        dbscan = cuDBSCAN(eps=eps, min_samples=min_samples, **kwargs) if self.engine == CUML else DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        res = cluster(res, dbscan, kind=kind)

        return res
    
    def dbscan(self, kind='nodes', eps: float = 1., min_samples: int = 1, **kwargs):
        """DBSCAN clustering
        """
        res = self.bind()
        res = self._cluster_dbscan(res, kind=kind, eps=eps, min_samples=min_samples, **kwargs)
        
        return res

    def _is_cudf(self, df):
        return 'cudf' in str(type(df))
