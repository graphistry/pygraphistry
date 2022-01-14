# a base class for UMAP to run on cpu or gpu

import umap
import pandas as pd
from ml.utils import setup_logger, pandas_to_sparse_adjacency
import ml.constants as config

logger = setup_logger(__name__)

umap_kwargs_probs = {
    "n_components": 2,
    "metric": "hellinger",
    "n_neighbors": 15,
}

umap_kwargs_euclidean = {
    "n_components": 2,
    "metric": "euclidean",
    "n_neighbors": 7,
}


class baseUmap:
    def __init__(self, **kwargs):
        self._reducer = umap.UMAP(**kwargs)

    def _check_target_is_one_dimensional(self, y):
        if y is None:
            return None
        if y.shape[1] <= 1:
            return y
        else:
            logger.warning(
                f"Ignoring target column of shape {y.shape} as it is not one dimensional"
            )
            return None

    def fit(self, X, y=None):
        y = self._check_target_is_one_dimensional(y)
        self.reducer = self._reducer.fit(X, y)
        self._edge_influence()

    def transform(self, X):
        # After it is fit with y, transform doesn't use y
        return self.reducer.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _edge_influence(self):
        if hasattr(self, "reducer"):
            coo = self.reducer.graph_.tocoo()
            src, dst, weight_col = config.SRC, config.DST, config.WEIGHT

            self.weighted_edges_df = pd.DataFrame(
                {src: coo.row, dst: coo.col, weight_col: coo.data}
            )
            (
                self.weighted_adjacency,
                self.umap_entity_to_index,
            ) = pandas_to_sparse_adjacency(self.weighted_edges_df, src, dst, weight_col)
        else:
            logger.warning("Must call `fit(X, y)` first")
