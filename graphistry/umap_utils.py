# a base class for UMAP to run on cpu or gpu
from time import time
from typing import Union, Any

import numpy as np
import pandas as pd

try:
    import umap, cuml
    has_dependancy = True
except:
    umap = Any
    has_dependancy = False
    
from . import constants as config
from .ai_utils import setup_logger

logger = setup_logger(__name__)

umap_kwargs_probs = {
    "n_components": 2,
    # "metric": "hellinger",  # info metric, can't use on textual encodings since they contain negative values...
    "n_neighbors": 15,
    "min_dist": 0.3,
    "verbose": True,
    "spread": 0.5,
    "local_connectivity": 1,
    "repulsion_strength": 1,
    "negative_sample_rate": 5,
    # "engine": 'cuml',
}

umap_kwargs_euclidean = {
    "n_components": 2,
    # "metric": "euclidean",
    "n_neighbors": 12,
    "min_dist": 0.1,
    "verbose": True,
    "spread": 0.5,
    "local_connectivity": 1,
    "repulsion_strength": 1,
    "negative_sample_rate": 5,
    # "engine": 'umap-learn',
}


class UMAPMixin(object):
    """
        UMAP Mixin for automagic UMAPing
        
    """
    def __init__(
        self,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread=0.5,
        local_connectivity=1,
        repulsion_strength=1,
        negative_sample_rate=5,
        n_components: int = 2,
        # metric: str = "euclidean",
        engine: str = "cuml",#"cuml"]
    ):
        if has_dependancy:
            umap_kwargs = dict(
                n_components=n_components,
                # metric=metric,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                negative_sample_rate=negative_sample_rate,
                # engine=engine,
            )
    
            self.n_components = n_components
            # self.metric = metric
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.spread = spread
            self.local_connectivity = local_connectivity
            self.repulsion_strength = repulsion_strength
            self.negative_sample_rate = negative_sample_rate
            self.engine = engine
            # if engine=='cuml':
            #     import cuml
            #     self._umap = cuml.UMAP(**umap_kwargs)
            # else:
            #     self._umap = umap.UMAP(**umap_kwargs)
            # self._umap = umap.UMAP(**umap_kwargs)
        

    def _set_new_kwargs(self, **kwargs):
        # self._umap = umap.UMAP(**kwargs)
        if self.engine=='cuml':
            import cuml
            self._umap=cuml.UMAP(**kwargs)
        else:
            self._umap=umap.UMAP(**kwargs)

    def _check_target_is_one_dimensional(self, y: Union[np.ndarray, None]):
        if y is None:
            return None
        if y.ndim == 1:
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return y
        else:
            logger.warning(
                f"* Ignoring target column of shape {y.shape} as it is not one dimensional"
            )
            return None

    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None):
        t = time()
        y = self._check_target_is_one_dimensional(y)
        logger.info(f"Starting UMAP-ing data of shape {X.shape}")
        self._umap.fit(X, y)
        self._edge_influence()
        mins = (time() - t) / 60
        logger.info(f"-UMAP-ing took {mins:.2f} minutes total")
        logger.info(f" - or {X.shape[0]/mins:.2f} rows per minute")
        return self

    def fit_transform(self, X: np.ndarray, y: Union[np.ndarray, None] = None):
        if self.engine=='cuml':
            # X.columns=['']
            self.fit(X.get(['source','destination']), y)
            logger.info(f"BAM! cuml umap")
        else:
            self.fit(X, y)
            logger.info(f"boo! umap-learn :(")
        return self._umap.transform(X)

    def _edge_influence(self):
        logger.debug("Calculating weighted adjacency (edge) DataFrame")
        coo = self._umap.graph_.tocoo()
        src, dst, weight_col = config.SRC, config.DST, config.WEIGHT

        self._weighted_edges_df = pd.DataFrame(
            {src: coo.row, dst: coo.col, weight_col: coo.data}
        )

        self._weighted_adjacency = self._umap.graph_
