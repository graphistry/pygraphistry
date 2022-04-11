from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np, pandas as pd
from time import time
from . import constants as config
from .ai_utils import setup_logger
from .feature_utils import (
    FeatureEngine,
    FeatureMixin,
    prune_weighted_edges_df_and_relabel_nodes,
    resolve_feature_engine,
    YSymbolic
)

logger = setup_logger(name=__name__, verbose=True)


if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object


###############################################################################


import_exn = None
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=ImportWarning)
        import umap
    has_dependancy = True
except ModuleNotFoundError as e:
    import_exn = e
    has_dependancy = False


def assert_imported():
    if not has_dependancy:
        logger.error("UMAP not found, trying running `pip install graphistry[ai]`")
        raise import_exn


###############################################################################


umap_kwargs_probs = {
    "n_components": 2,
    "metric": "hellinger",  # info metric, can't use on textual encodings since they contain negative values...
    "n_neighbors": 15,
    "min_dist": 0.3,
    "verbose": True,
    "spread": 0.5,
    "local_connectivity": 1,
    "repulsion_strength": 1,
    "negative_sample_rate": 5,
}

umap_kwargs_euclidean = {
    "n_components": 2,
    "metric": "euclidean",
    "n_neighbors": 12,
    "min_dist": 0.1,
    "verbose": True,
    "spread": 0.5,
    "local_connectivity": 1,
    "repulsion_strength": 1,
    "negative_sample_rate": 5,
}


def umap_graph_to_weighted_edges(umap_graph, cfg=config):
    logger.debug("Calculating weighted adjacency (edge) DataFrame")
    coo = umap_graph.tocoo()
    src, dst, weight_col = cfg.SRC, cfg.DST, cfg.WEIGHT

    _weighted_edges_df = pd.DataFrame(
        {src: coo.row, dst: coo.col, weight_col: coo.data}
    )

    return _weighted_edges_df


class UMAPMixin(MIXIN_BASE):
    """
    UMAP Mixin for automagic UMAPing

    """

    def __init__(self, *args, **kwargs):
        self.umap_initialized = False
        pass

    def umap_lazy_init(
        self,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread=0.5,
        local_connectivity=1,
        repulsion_strength=1,
        negative_sample_rate=5,
        n_components: int = 2,
        metric: str = "euclidean",
    ):

        # FIXME remove as set_new_kwargs will always replace?

        if has_dependancy and not self.umap_initialized:

            umap_kwargs = dict(
                n_components=n_components,
                metric=metric,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                negative_sample_rate=negative_sample_rate,
            )

            self.n_components = n_components
            self.metric = metric
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.spread = spread
            self.local_connectivity = local_connectivity
            self.repulsion_strength = repulsion_strength
            self.negative_sample_rate = negative_sample_rate
            self._umap = umap.UMAP(**umap_kwargs)

            self.umap_initialized = True


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

    # FIXME rename to umap_fit
    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")

        t = time()
        y = self._check_target_is_one_dimensional(y)
        logger.info(f"Starting UMAP-ing data of shape {X.shape}")
        self._umap.fit(X, y)
        self._weighted_edges_df = umap_graph_to_weighted_edges(self._umap.graph_)
        self._weighted_adjacency = self._umap.graph_
        mins = (time() - t) / 60
        logger.info(f"-UMAP-ing took {mins:.2f} minutes total")
        logger.info(f" - or {X.shape[0]/mins:.2f} rows per minute")
        return self

    # FIXME rename to umap_fit_transform
    def fit_transform(self, X: Any, y: Union[Any, None] = None):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")
        self.fit(X, y)
        return self._umap.transform(X)

    def umap(
        self,
        kind: str = "nodes",
        use_columns: Optional[List] = None,
        feature_engine: FeatureEngine = "auto",
        encode_position: bool = True,
        encode_weight: bool = True,
        inplace: bool = False,
        X: np.ndarray = None,
        y: YSymbolic = None,
        scale: float = 0.1,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        scale_xy: float = 10,
        suffix: str = "",
        play: Optional[int] = 0,
        engine: str = "umap_learn",
    ):
        """
            UMAP the featurized node or edges data, or pass in your own X, y (optional).

        :param kind: `nodes` or `edges` or None. If None, expects explicit X, y (optional) matrices, and will Not
                associate them to nodes or edges. If X, y (optional) is given, with kind = [nodes, edges],
                it will associate new matrices to nodes or edges attributes.
        :param use_columns: List of columns to use for featurization if featurization hasn't been applied.
        :param feature_engine: How to encode data ("none", "auto", "pandas", "dirty_cat", "torch")
        :param encode_weight: if True, will set new edges_df from implicit UMAP, default True.
        :param encode_position: whether to set default plotting bindings -- positions x,y from umap for .plot()
        :param X: ndarray of features
        :param y: ndarray of targets
        :param scale: multiplicative scale for pruning weighted edge DataFrame gotten from UMAP (mean + scale *std)
        :param n_neighbors: UMAP number of nearest neighbors to include for UMAP connectivity, lower makes more compact layouts. Minimum 2.
        :param min_dist: UMAP float between 0 and 1, lower makes more compact layouts.
        :param spread: UMAP spread of values for relaxation
        :param local_connectivity: UMAP connectivity parameter
        :param repulsion_strength: UMAP repulsion strength
        :param negative_sample_rate: UMAP negative sampling rate
        :param n_components: number of components in the UMAP projection, default 2
        :param metric: UMAP metric, default 'euclidean'. Other useful ones are 'hellinger', '..'
                see (UMAP-LEARN)[https://umap-learn.readthedocs.io/en/latest/parameters.html] documentation for more.
        :param suffix: optional suffix to add to x, y attributes of umap.
        :param play: Graphistry play parameter, default 0, how much to evolve the network during clustering
        :param engine: selects which engine to use to calculate UMAP: NotImplemented yet, default UMAP-LEARN
        :return: self, with attributes set with new data
        """
        assert_imported()
        self.umap_lazy_init()

        self.suffix = suffix
        xy = None
        umap_kwargs = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
        )

        if inplace:
            res : UMAPMixin = self
        else:
            res = self.bind()

        res._umap = umap.UMAP(**umap_kwargs)

        resolved_feature_engine = resolve_feature_engine(feature_engine)

        if kind == "nodes":

            # FIXME not sure if this is preserving the intent
            # ... when should/shouldn't we relabel?
            index_to_nodes_dict = None
            if res._node is None:
                res = res.nodes(  # type: ignore
                    res._nodes.reset_index(drop=True)
                    .reset_index()
                    .rename(columns={"index": config.IMPLICIT_NODE_ID}),
                    config.IMPLICIT_NODE_ID,
                )
                nodes = res._nodes[res._node].values
                index_to_nodes_dict = dict(zip(range(len(nodes)), nodes))

            X, y, res = res._featurize_or_get_nodes_dataframe_if_X_is_None(
                X, y, use_columns, feature_engine=resolved_feature_engine
            )
            xy = scale_xy * res.fit_transform(X, y)
            res._weighted_adjacency_nodes = res._weighted_adjacency
            res._node_embedding = xy
            # TODO add edge filter so graph doesn't have double edges
            # TODO user-guidable edge merge policies like upsert?
            res._weighted_edges_df_from_nodes = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,  # type: ignore
                    scale=scale,
                    index_to_nodes_dict=index_to_nodes_dict,
                )
            )
        elif kind == "edges":
            X, y, res = res._featurize_or_get_edges_dataframe_if_X_is_None(
                X, y, use_columns, feature_engine=resolved_feature_engine
            )
            xy = scale_xy * res.fit_transform(X, y)
            res._weighted_adjacency_edges = res._weighted_adjacency
            res._edge_embedding = xy
            res._weighted_edges_df_from_edges = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,  # type: ignore
                    scale=scale,
                    index_to_nodes_dict=None
                )
            )
        elif kind is None:
            logger.warning(
                "kind should be one of `nodes` or `edges` unless you are passing explicit matrices"
            )
            if X is not None:
                logger.info("New Matrix `X` passed in for UMAP-ing")
                xy = res.fit_transform(X, y)
                res._xy = xy
                res._weighted_edges_df = prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df, scale=scale
                )
                logger.info(
                    "Reduced Coordinates are stored in `._xy` attribute and "
                    "pruned weighted_edge_df in `._weighted_edges_df` attribute"
                )
            else:
                logger.error(
                    "If `kind` is `None`, `X` and optionally `y` must be given"
                )
        else:
            raise ValueError(
                f"`kind` needs to be one of `nodes`, `edges`, `None`, got {kind}"
            )
        res = self._bind_xy_from_umap(res, kind, encode_position, encode_weight, play)
        if not inplace:
            return res

    def _bind_xy_from_umap(
        self,
        res: Any,
        kind: str,
        encode_position: bool,
        encode_weight: bool,
        play: Optional[int],
    ):
        # todo make sure xy is two dim, might be 3 or more....
        df = res._nodes if kind == "nodes" else res._edges

        df = df.copy(deep=False)
        x_name = config.X + self.suffix
        y_name = config.Y + self.suffix
        if kind == "nodes":
            emb = res._node_embedding
        else:
            emb = res._edge_embedding
        df[x_name] = emb.T[0]
        df[y_name] = emb.T[1]

        res = res.nodes(df) if kind == "nodes" else res.edges(df)

        if encode_weight and kind == "nodes":
            w_name = config.WEIGHT + self.suffix
            umap_df = res._weighted_edges_df_from_nodes.copy(deep=False)
            umap_df = umap_df.rename({config.WEIGHT: w_name})
            res = res.edges(umap_df, config.SRC, config.DST)
            logger.info(
                f"Wrote new edges_dataframe from UMAP embedding of shape {res._edges.shape}"
            )
            res = res.bind(edge_weight=w_name)

        if encode_position and kind == "nodes":
            if play is not None:
                return res.bind(point_x=x_name, point_y=y_name).layout_settings(
                    play=play
                )
            else:
                return res.bind(point_x=x_name, point_y=y_name)

        return res


    def filter_weighted_edges(
        self,
        scale: float = 0.1,
        index_to_nodes_dict: Optional[Dict] = None,
        inplace: bool = False,
    ):
        if inplace:
            res = self
        else:
            res = self.bind()

        if res._weighted_edges_df is not None:
            res._weighted_edges_df_from_nodes = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,
                    scale=scale,
                    index_to_nodes_dict=index_to_nodes_dict,
                )
            )
        else:
            logger.error("UMAP has not been run, run g.featurize(...).umap(...) first")

        # write new res._edges df
        res = self._bind_xy_from_umap(
            res, "nodes", encode_position=True, encode_weight=True, play=0
        )

        if not inplace:
            return res
