# a base class for UMAP to run on cpu or gpu
import pandas as pd
import umap

import ml.constants as config
from ml.utils import setup_logger

logger = setup_logger(__name__)

umap_kwargs_probs = {
    "n_components": 2,
    "metric": "hellinger",  # info metric
    "n_neighbors": 15,
    "min_dist": 0.1,
}

umap_kwargs_euclidean = {
    "n_components": 2,
    "metric": "euclidean",
    "n_neighbors": 12,
    "min_dist": 0.1,
}


class BaseUMAPMixin(umap.UMAP):
    def __init__(self, **kwargs):
        self._is_fit = False
        super().__init__(**kwargs)

    def _set_new_kwargs(self, **kwargs):
        super().__init__(**kwargs)

    def _check_target_is_one_dimensional(self, y):
        if y is None:
            return None
        if y.ndim == 1:
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return y
        else:
            logger.warning(
                f"Ignoring target column of shape {y.shape} as it is not one dimensional"
            )
            return None

    def fit(self, X, y=None):
        y = self._check_target_is_one_dimensional(y)
        super().fit(X, y)
        self._is_fit = True
        self._edge_influence()
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _edge_influence(self):
        if self._is_fit:
            logger.debug("Calculating weighted adjacency (edge) DataFrame")
            coo = self.graph_.tocoo()
            src, dst, weight_col = config.SRC, config.DST, config.WEIGHT

            self._weighted_edges_df = pd.DataFrame(
                {src: coo.row, dst: coo.col, weight_col: coo.data}
            )

            self._weighted_adjacency = self.graph_
        else:
            logger.warning("Must call `fit(X, y)` first")
