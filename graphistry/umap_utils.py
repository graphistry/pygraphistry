import copy
from time import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd

from . import constants as config
from .feature_utils import (FeatureMixin, Literal, XSymbolic, YSymbolic,
                            prune_weighted_edges_df_and_relabel_nodes,
                            resolve_feature_engine)
from .PlotterBase import Plottable, WeakValueDictionary
from .util import check_set_memoize, setup_logger

logger = setup_logger(name=__name__, verbose=config.VERBOSE)

if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object


###############################################################################


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
        import cuml  # type: ignore

        return True, "ok", cuml
    except ModuleNotFoundError as e:
        return False, e, None


def assert_imported():
    has_dependancy_, import_exn, umap_learn = lazy_umap_import_has_dependancy()
    if not has_dependancy_:
        logger.error("UMAP not found, trying running " "`pip install graphistry[ai]`")
        raise import_exn


def assert_imported_cuml():
    has_cuml_dependancy_, import_cuml_exn, cuml = lazy_cuml_import_has_dependancy()
    if not has_cuml_dependancy_:
        logger.warning("cuML not found, trying running " "`pip install cuml`")
        raise import_cuml_exn


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


UMAPEngineConcrete = Literal["cuml", "umap_learn"]
UMAPEngine = Literal[UMAPEngineConcrete, "auto"]


def resolve_umap_engine(
    engine: UMAPEngine,
) -> UMAPEngineConcrete:  # noqa
    if engine in ["cuml", "umap_learn"]:
        return engine  # type: ignore
    if engine in ["auto"]:
        has_cuml_dependancy_, _, cuml = lazy_cuml_import_has_dependancy()
        if has_cuml_dependancy_:
            return "cuml"
        has_umap_dependancy_, _, _ = lazy_umap_import_has_dependancy()
        if has_umap_dependancy_:
            return "umap_learn"

    raise ValueError(  # noqa
        f'engine expected to be "auto", '
        '"umap_learn", or  "cuml" '
        f"but received: {engine} :: {type(engine)}"
    )


###############################################################################


umap_kwargs_probs = {
    "n_components": 2,
    "metric": "hellinger",  # info metric, can't use on
    # textual encodings since they contain negative values...
    # unless scaling min max etc
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

# #############################################################################
#
#      Fast Memoize
#
# #############################################################################


def reuse_umap(g: Plottable, memoize: bool, metadata: Any):  # noqa: C901
    return check_set_memoize(
        g, metadata, attribute="_umap_param_to_g", name="umap", memoize=memoize
    )


def umap_graph_to_weighted_edges(umap_graph, engine, is_legacy, cfg=config):
    logger.debug("Calculating weighted adjacency (edge) DataFrame")
    coo = umap_graph.tocoo()
    src, dst, weight_col = cfg.SRC, cfg.DST, cfg.WEIGHT
    if (engine == "umap_learn") or is_legacy:
        _weighted_edges_df = pd.DataFrame(
            {src: coo.row, dst: coo.col, weight_col: coo.data}
        )
    elif (engine == "cuml") and not is_legacy:
        _weighted_edges_df = pd.DataFrame(
            {src: coo.get().row, dst: coo.get().col, weight_col: coo.get().data}
        )
    return _weighted_edges_df


class UMAPMixin(MIXIN_BASE):
    """
    UMAP Mixin for automagic UMAPing

    """
    # FIXME where is this used? 
    _umap_memoize: WeakValueDictionary = WeakValueDictionary()

    def __init__(self, *args, **kwargs):
        self.umap_initialized = False

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
        engine: UMAPEngine = "auto",
        suffix: str = "",
    ):
        engine_resolved = resolve_umap_engine(engine)
        # FIXME remove as set_new_kwargs will always replace?
        if engine_resolved == "umap_learn":
            _, _, umap_engine = lazy_umap_import_has_dependancy()
        elif engine_resolved == "cuml":
            _, _, umap_engine = lazy_cuml_import_has_dependancy()
        else:
            raise ValueError(
                "No umap engine, ensure 'auto', 'umap_learn', or 'cuml', and the library is installed"
            )

        if not self.umap_initialized:
            umap_kwargs = dict(
                {
                    "n_components": n_components,
                    **({"metric": metric} if engine_resolved == "umap_learn" else {}),
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "spread": spread,
                    "local_connectivity": local_connectivity,
                    "repulsion_strength": repulsion_strength,
                    "negative_sample_rate": negative_sample_rate,
                }
            )

            self.n_components = n_components
            self.metric = metric
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.spread = spread
            self.local_connectivity = local_connectivity
            self.repulsion_strength = repulsion_strength
            self.negative_sample_rate = negative_sample_rate
            self._umap = umap_engine.UMAP(**umap_kwargs)
            self.umap_initialized = True
            self.engine = engine_resolved
            self.suffix = suffix

    def _check_target_is_one_dimensional(self, y: Union[pd.DataFrame, None]):
        if y is None:
            return None
        if y.ndim == 1:
            return y
        elif y.ndim == 2 and y.shape[1] == 1:
            return y
        else:
            logger.warning(
                f"* Ignoring target column of shape {y.shape} in UMAP fit, "
                "as it is not one dimensional"
            )
            return None

    def umap_fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")
        t = time()
        y = self._check_target_is_one_dimensional(y)
        logger.info("-" * 90)
        logger.info(f"Starting UMAP-ing data of shape {X.shape}")

        if self.engine == "cuml" and is_legacy_cuml():
            from cuml.neighbors import NearestNeighbors

            knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            cc = self._umap.fit(X, y, knn_graph=knn)
            knn.fit(cc.embedding_)
            self._umap.graph_ = knn.kneighbors_graph(cc.embedding_)
            self._weighted_adjacency = self._umap.graph_

        else:
            self._umap.fit(X, y)
            self._weighted_adjacency = self._umap.graph_
        # if changing, also update fresh_res
        self._weighted_edges_df = umap_graph_to_weighted_edges(
            self._umap.graph_, self.engine, is_legacy_cuml()
        )

        mins = (time() - t) / 60
        logger.info(f"-UMAP-ing took {mins:.2f} minutes total")
        logger.info(f" - or {X.shape[0]/mins:.2f} rows per minute")
        return self

    def umap_fit_transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")
        self.umap_fit(X, y)
        emb = self._umap.transform(X)
        emb = self._bundle_embedding(emb, index=X.index)
        return emb

    def transform_umap(  # noqa: E303
        self, df: pd.DataFrame, ydf: pd.DataFrame, kind: str = "nodes"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            logger.debug(f"Going into Transform umap {df.shape}, {ydf.shape}")
        except:
            pass
        x, y = self.transform(df, ydf, kind=kind)
        emb = self._umap.transform(x)  # type: ignore
        emb = self._bundle_embedding(emb, index=df.index)
        return emb, x, y

    def _bundle_embedding(self, emb, index):
        # Converts Embedding into dataframe and takes care if emb.dim > 2
        if emb.shape[1] == 2:
            emb = pd.DataFrame(emb, columns=[config.X, config.Y], index=index)
        else:
            columns = [config.X, config.Y] + [
                f"umap_{k}" for k in range(2, emb.shape[1] - 2)
            ]
            emb = pd.DataFrame(emb, columns=columns, index=index)
        return emb

    def _process_umap(
        self,
        res,
        X_: pd.DataFrame,
        y_: pd.DataFrame,
        kind,
        memoize: bool,
        featurize_kwargs,
        **umap_kwargs,
    ):
        """
        Returns res mutated with new _xy
        """
        res._umap = self._umap

        logger.debug("process_umap before kwargs: %s", umap_kwargs)
        umap_kwargs.update({"kind": kind, "X": X_, "y": y_})
        umap_kwargs = {**umap_kwargs, "featurize_kwargs": featurize_kwargs or {}}
        logger.debug("process_umap after kwargs: %s", umap_kwargs)

        old_res = reuse_umap(
            res, memoize, {**umap_kwargs, "featurize_kwargs": featurize_kwargs or {}}
        )
        if old_res:
            logger.info(" --- [[ RE-USING UMAP ]]")
            fresh_res = copy.copy(res)
            for attr in ["_xy", "_weighted_edges_df", "_weighted_adjacency"]:
                setattr(fresh_res, attr, getattr(old_res, attr))
            # have to set _raw_data attribute on umap?
            fresh_res._umap = old_res._umap  # this saves the day!
            return fresh_res

        emb = res.umap_fit_transform(X_, y_)
        res._xy = emb
        return res

    def _set_features(  # noqa: E303
        self, res, X, y, kind, feature_engine, featurize_kwargs
    ):
        """
        Helper for setting features for memoize
        """
        kv = {}

        if not hasattr(res, "_feature_params") or res._feature_params is None:
            res._feature_params = {"nodes": {}, "edges": {}}
        # if we have featurized previously
        if kind in res._feature_params:
            # add in all the stuff that got stored in res._feature_params
            kv.update(res._feature_params[kind])
            # kv.pop('kind') # pop off kind, as featurization
            # doesn't use it (we have one for nodes, and one for
            # edges explicitly)

        if len(featurize_kwargs):
            # overwrite with anything stated in featurize_kwargs
            kv.update(featurize_kwargs)

        kv.update({"feature_engine": resolve_feature_engine(feature_engine)})

        # potentially overwrite with explicit mention here
        if X is not None:
            kv.update({"X": X})

        if y is not None:
            kv.update({"y": y})

        # set the features fully, and if this in memoize,
        # it will skip and just returns previous .featurize/umap
        featurize_kwargs = kv

        return featurize_kwargs

    def umap(
        self,
        kind: str = "nodes",
        X: XSymbolic = None,
        y: YSymbolic = None,
        scale: float = 1.0,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        suffix: str = "",
        play: Optional[int] = 0,
        encode_position: bool = True,
        encode_weight: bool = True,
        engine: UMAPEngine = "auto",
        inplace: bool = False,
        feature_engine: str = "auto",
        memoize: bool = True,
        **featurize_kwargs,
    ):
        """
            UMAP the featurized node or edges data,
            or pass in your own X, y (optional).

        :param kind: `nodes` or `edges` or None.
                If None, expects explicit X, y (optional) matrices,
                and will Not associate them to nodes or edges.
                If X, y (optional) is given, with kind = [nodes, edges],
                it will associate new matrices to nodes or edges attributes.
        :param feature_engine: How to encode data
                ("none", "auto", "pandas", "dirty_cat", "torch")
        :param encode_weight: if True, will set new edges_df from
                implicit UMAP, default True.
        :param encode_position: whether to set default plotting bindings
                -- positions x,y from umap for .plot()
        :param X: either an ndarray of features, or column names to featurize
        :param y: either an ndarray of targets, or column names to featurize
                targets
        :param scale: multiplicative scale for pruning weighted edge DataFrame
                gotten from UMAP, between [0, ..) with high end meaning keep
                all edges
        :param n_neighbors: UMAP number of nearest neighbors to include for
                UMAP connectivity, lower makes more compact layouts. Minimum 2
        :param min_dist: UMAP float between 0 and 1, lower makes more compact
                layouts.
        :param spread: UMAP spread of values for relaxation
        :param local_connectivity: UMAP connectivity parameter
        :param repulsion_strength: UMAP repulsion strength
        :param negative_sample_rate: UMAP negative sampling rate
        :param n_components: number of components in the UMAP projection,
                default 2
        :param metric: UMAP metric, default 'euclidean'.
                see (UMAP-LEARN)[https://umap-learn.readthedocs.io/
                en/latest/parameters.html] documentation for more.
        :param suffix: optional suffix to add to x, y attributes of umap.
        :param play: Graphistry play parameter, default 0, how much to evolve
                the network during clustering
        :param engine: selects which engine to use to calculate UMAP:
                NotImplemented yet, default UMAP-LEARN
        :param memoize: whether to memoize the results of this method,
                default True.
        :return: self, with attributes set with new data
        """
        if engine == "umap_learn":
            assert_imported()
        elif engine == "cuml":
            assert_imported_cuml()

        umap_kwargs = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            engine=engine,
        )
        logger.debug("umap_kwargs: %s", umap_kwargs)

        if inplace:
            res = self
        else:
            res = self.bind()

        res.umap_lazy_init(engine=engine, suffix=suffix)
        # res.suffix = suffix

        logger.debug("umap input X :: %s", X)
        logger.debug("umap input y :: %s", y)

        featurize_kwargs = self._set_features(
            res, X, y, kind, feature_engine, {**featurize_kwargs, "memoize": memoize}
        )
        # umap_kwargs = {**umap_kwargs,
        # 'featurize_kwargs': featurize_kwargs or {}}

        if kind == "nodes":
            if res._node is None:

                logger.debug("-Writing new node name")
                res = res.nodes(  # type: ignore
                    res._nodes.reset_index(drop=True)
                    .reset_index()
                    .rename(columns={"index": config.IMPLICIT_NODE_ID}),
                    config.IMPLICIT_NODE_ID,
                )

            nodes = res._nodes[res._node].values
            index_to_nodes_dict = dict(zip(range(len(nodes)), nodes))

            logger.debug("propagating with featurize_kwargs: %s", featurize_kwargs)
            (
                X_,
                y_,
                res,
            ) = res._featurize_or_get_nodes_dataframe_if_X_is_None(  # type: ignore
                **featurize_kwargs
            )

            logger.debug("umap X_: %s", X_)
            logger.debug("umap y_: %s", y_)

            res = res._process_umap(
                res, X_, y_, kind, memoize, featurize_kwargs, **umap_kwargs
            )

            res._weighted_adjacency_nodes = res._weighted_adjacency
            if res._xy is None:
                raise RuntimeError("This should not happen")
            res._node_embedding = res._xy
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

            logger.debug("propagating with featurize_kwargs: %s", featurize_kwargs)
            (
                X_,
                y_,
                res,
            ) = res._featurize_or_get_edges_dataframe_if_X_is_None(  # type: ignore
                **featurize_kwargs
            )

            res = res._process_umap(
                res, X_, y_, kind, memoize, featurize_kwargs, **umap_kwargs
            )
            res._weighted_adjacency_edges = res._weighted_adjacency
            if res._xy is None:
                raise RuntimeError("This should not happen")
            res._edge_embedding = res._xy
            res._weighted_edges_df_from_edges = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,  # type: ignore
                    scale=scale,
                    index_to_nodes_dict=None,
                )
            )
        elif kind is None:
            logger.warning(
                "kind should be one of `nodes` or `edges` unless"
                "you are passing explicit matrices"
            )
            if X is not None and isinstance(X, pd.DataFrame):
                logger.info("New Matrix `X` passed in for UMAP-ing")
                xy = res.umap_fit_transform(X, y)
                res._xy = xy
                res._weighted_edges_df = prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df, scale=scale
                )
                logger.info(  # noqa: E501
                    "Reduced Coordinates are stored in `._xy` attribute and "  # noqa: E501
                    "pruned weighted_edge_df in `._weighted_edges_df` attribute"  # noqa: E501
                )
            else:
                logger.error(
                    "If `kind` is `None`, `X` and optionally `y`"
                    "must be given and be of type pd.DataFrame"
                )
        else:
            raise ValueError(
                f"`kind` needs to be one of `nodes`, `edges`, `None`, got {kind}"  # noqa: E501
            )
        res = res._bind_xy_from_umap(
            res, kind, encode_position, encode_weight, play
        )  # noqa: E501

        if res.engine == "cuml" and is_legacy_cuml():
            res = res.prune_self_edges()

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
        df = res._nodes if kind == "nodes" else res._edges

        df = df.copy(deep=False)
        x_name = config.X + res.suffix
        y_name = config.Y + res.suffix
        if kind == "nodes":
            emb = res._node_embedding
        else:
            emb = res._edge_embedding

        df[x_name] = emb.values.T[0]  # if embedding is greater
        # than two dimensions will only take first two coordinates
        df[y_name] = emb.values.T[1]
        #
        res = res.nodes(df) if kind == "nodes" else res.edges(df)

        if encode_weight and kind == "nodes":
            # adds the implicit edge dataframe and binds it to
            # graphistry instance
            w_name = config.WEIGHT + res.suffix
            umap_edges_df = res._weighted_edges_df_from_nodes.copy(deep=False)
            umap_edges_df = umap_edges_df.rename(columns={config.WEIGHT: w_name})
            res = res.edges(umap_edges_df, config.SRC, config.DST)
            logger.info(
                " - Wrote new edges_dataframe from UMAP "
                f"embedding of shape {res._edges.shape}"
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
        scale: float = 1.0,
        index_to_nodes_dict: Optional[Dict] = None,
        inplace: bool = False,
        kind: str = "nodes",
    ):
        """
        Filter edges based on _weighted_edges_df (ex: from .umap())
        """
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
            raise RuntimeError("UMAP has not been run, run g.umap(...) first")

        # write new res._edges df
        res = self._bind_xy_from_umap(
            res, kind, encode_position=True, encode_weight=True, play=0
        )

        if not inplace:
            return res
