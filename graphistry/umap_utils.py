import copy
from time import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from inspect import getmodule

import numpy as np
import pandas as pd

from graphistry.utils.lazy_import import (
    lazy_cudf_import,
    lazy_umap_import,
    lazy_cuml_import,
)
from . import constants as config
from .constants import CUML, UMAP_LEARN
from .feature_utils import (FeatureMixin, Literal, XSymbolic, YSymbolic,
                            resolve_feature_engine)
from .PlotterBase import Plottable, WeakValueDictionary
from .util import check_set_memoize, setup_logger


logger = setup_logger(__name__)

if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object


DataFrameLike = Union[pd.DataFrame, Any]

###############################################################################


def assert_imported():
    has_dependancy_, import_exn, _ = lazy_umap_import()
    if not has_dependancy_:
        logger.error("UMAP not found, trying running " "`pip install graphistry[ai]`")
        raise import_exn


def assert_imported_cuml():
    has_cuml_dependancy_, import_cuml_exn, _ = lazy_cuml_import()
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


UMAPEngineConcrete = Literal['cuml', 'umap_learn']
UMAPEngine = Literal[UMAPEngineConcrete, "auto"]


def resolve_umap_engine(
    engine: UMAPEngine,
) -> UMAPEngineConcrete:  # noqa
    if engine in [CUML, UMAP_LEARN]:
        return engine  # type: ignore
    if engine in ["auto"]:
        has_cuml_dependancy_, _, _ = lazy_cuml_import()
        if has_cuml_dependancy_:
            return 'cuml'
        has_umap_dependancy_, _, _ = lazy_umap_import()
        if has_umap_dependancy_:
            return 'umap_learn'

    raise ValueError(  # noqa
        f'engine expected to be "auto", '
        '"umap_learn", or  "cuml" '
        f"but received: {engine} :: {type(engine)}"
    )


def make_safe_gpu_dataframes(X, y, engine):

    def safe_cudf(X, y):
        # remove duplicate columns
        if len(X.columns) != len(set(X.columns)):
            X = X.loc[:, ~X.columns.duplicated()]
        try:
            y = y.loc[:, ~y.columns.duplicated()]
        except:
            pass
        new_kwargs = {}
        kwargs = {'X': X, 'y': y}
        for key, value in kwargs.items():
            if isinstance(value, cudf.DataFrame) and engine in ["pandas", "umap_learn", "dirty_cat"]:
                new_kwargs[key] = value.to_pandas()
            elif isinstance(value, pd.DataFrame) and engine in ["cuml", "cu_cat"]:
                new_kwargs[key] = cudf.from_pandas(value)
            else:
                new_kwargs[key] = value
        return new_kwargs['X'], new_kwargs['y']

    has_cudf_dependancy_, _, cudf = lazy_cudf_import()
    if has_cudf_dependancy_:
        return safe_cudf(X, y)
    else:
        return X, y

###############################################################################

# #############################################################################
#
#      Fast Memoize
#
# #############################################################################


def reuse_umap(g: Plottable, memoize: bool, metadata: Any):  # noqa: C901
    return check_set_memoize(
        g, metadata, attribute="_umap_param_to_g", name="umap", memoize=memoize
    )


def umap_graph_to_weighted_edges(umap_graph, engine: UMAPEngineConcrete, is_legacy, cfg=config):
    logger.debug("Calculating weighted adjacency (edge) DataFrame")
    coo = umap_graph.tocoo()
    src, dst, weight_col = cfg.SRC, cfg.DST, cfg.WEIGHT
    if (engine == "umap_learn") or is_legacy:
        return pd.DataFrame(
            {src: coo.row, dst: coo.col, weight_col: coo.data}
        )
    assert engine == "cuml"
    import cudf
    return cudf.DataFrame(
        {src: coo.get().row, dst: coo.get().col, weight_col: coo.get().data}
    )


##############################################################################


def prune_weighted_edges_df_and_relabel_nodes(
    wdf: DataFrameLike, scale: float = 0.1, index_to_nodes_dict: Optional[Dict] = None
) -> pd.DataFrame:
    """Prune the weighted edge DataFrame so to return high fidelity similarity scores.

    :param wdf: weighted edge DataFrame gotten via UMAP
    :param scale: lower values means less edges > (max - scale * std)
    :param index_to_nodes_dict: dict of index to node name;
            remap src/dst values if provided
    :return: pd.DataFrame
    """
    # we want to prune edges, so we calculate some statistics
    desc = wdf.describe()
    eps = 1e-3

    mean = desc[config.WEIGHT]["mean"]
    std = desc[config.WEIGHT]["std"]
    max_val = desc[config.WEIGHT]["max"] + eps
    min_val = desc[config.WEIGHT]["min"] - eps
    thresh = np.max(
        [max_val - scale, min_val]
    )  # if std =0 we add eps so we still have scale in the equation

    logger.info(
        f" -- edge weights: mean({mean:.2f}), "
        f"std({std:.2f}), max({max_val}), "
        f"min({min_val:.2f}), thresh({thresh:.2f})"
    )
    wdf2 = wdf[
        wdf[config.WEIGHT] >= thresh
    ]  # adds eps so if scale = 0, we have small window/wiggle room
    logger.info(
        " -- Pruning weighted edge DataFrame "
        f"from {len(wdf):,} to {len(wdf2):,} edges."
    )
    if index_to_nodes_dict is not None:
        wdf2[config.SRC] = wdf2[config.SRC].map(index_to_nodes_dict)
        wdf2[config.DST] = wdf2[config.DST].map(index_to_nodes_dict)
    return wdf2



class UMAPMixin(MIXIN_BASE):
    """
    UMAP Mixin for automagic UMAPing
    """
    # FIXME where is this used? 
    _umap_memoize: WeakValueDictionary = WeakValueDictionary()

    def __init__(self, *args, **kwargs):
        #self._umap_initialized = False
        #self.engine = self.engine if hasattr(self, "engine") else None
        pass


    def umap_lazy_init(
        self,
        res,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        engine: UMAPEngine = "auto",
        suffix: str = "",
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
    ):
        from graphistry.features import ModelDict

        engine_resolved = resolve_umap_engine(engine)
        # FIXME remove as set_new_kwargs will always replace?
        if engine_resolved == UMAP_LEARN:
            _, _, umap_engine = lazy_umap_import()
        elif engine_resolved == CUML:
            _, _, umap_engine = lazy_cuml_import()
        else:
            raise ValueError(
                "No umap engine, ensure 'auto', 'umap_learn', or 'cuml', and the library is installed"
            )
        umap_params = ModelDict("UMAP Parameters",
                **{
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "spread": spread,
                    "local_connectivity": local_connectivity,
                    "repulsion_strength": repulsion_strength,
                    "negative_sample_rate": negative_sample_rate,
                    "n_components": n_components,
                    **({"metric": metric} if engine_resolved == UMAP_LEARN else {}),  # type: ignore
                    **umap_kwargs
                }
            )
        
        if (
            getattr(res, '_umap_params', None) == umap_params
            and getattr(res, '_umap_fit_kwargs', None) == umap_fit_kwargs
            and getattr(res, '_umap_transform_kwargs', None) == umap_transform_kwargs
        ):
            logger.debug('Same umap params as last time, skipping new init')
            return res
        
        logger.debug('lazy init')
        # set new umap kwargs
        res._umap_params = umap_params
        res._umap_fit_kwargs = umap_fit_kwargs
        res._umap_transform_kwargs = umap_transform_kwargs

        res._n_components = n_components
        res._metric = metric
        res._n_neighbors = n_neighbors
        res._min_dist = min_dist
        res._spread = spread
        res._local_connectivity = local_connectivity
        res._repulsion_strength = repulsion_strength
        res._negative_sample_rate = negative_sample_rate
        res._umap = umap_engine.UMAP(**umap_params)
        logger.debug('Initialized UMAP with params: %s', umap_params)
        res.engine = engine_resolved
        res._suffix = suffix
                                                            
        return res

    #@safe_gpu_dataframes
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
    
    def _get_embedding(self, kind='nodes'):
        if kind == 'nodes':
            return self._node_embedding
        elif kind == 'edges':
            return self._edge_embedding
        else:
            raise ValueError('kind must be one of `nodes` or `edges`')

    def umap_fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
        umap_fit_kwargs: Dict[str, Any] = {}
    ):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")
        t = time()
        y = self._check_target_is_one_dimensional(y)
        logger.info("-" * 90)
        logger.info(f"Starting UMAP-ing data of shape {X.shape}")

        if self.engine == CUML and is_legacy_cuml():  # type: ignore
            from cuml.neighbors import NearestNeighbors

            knn = NearestNeighbors(n_neighbors=self._n_neighbors)  # type: ignore
            cc = self._umap.fit(X, y, knn_graph=knn, **umap_fit_kwargs)
            knn.fit(cc.embedding_)
            self._umap.graph_ = knn.kneighbors_graph(cc.embedding_)
        else:
            self._umap.fit(X, y, **umap_fit_kwargs)
            
        self._weighted_adjacency = self._umap.graph_
        # if changing, also update fresh_res
        self._weighted_edges_df = umap_graph_to_weighted_edges(
            self._umap.graph_, self.engine, is_legacy_cuml()  # type: ignore
        )

        mins = (time() - t) / 60
        logger.info(f"-UMAP-ing took {mins:.2f} minutes total")
        logger.info(f" - or {X.shape[0]/mins:.2f} rows per minute")
        return self


    def _umap_fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {}
    ):
        if self._umap is None:
            raise ValueError("UMAP is not initialized")
        self.umap_fit(X, y, umap_fit_kwargs)
        logger.debug('_umap_fit_transform:\nX::%s\n%s\n%s\nkwargs:\n%s\ny:\n%s', type(X), X.dtypes, X, umap_transform_kwargs, y)
        #logger.debug('per col types: %s', {k: (type(X[k]), X[k].dtype) for k in X.columns})
        try:
            logger.debug('X as pandas', X.to_pandas())  # type: ignore
        except:
            pass
        emb = self._umap.transform(X, **umap_transform_kwargs)
        emb = self._bundle_embedding(emb, index=X.index)
        return emb


    def transform_umap(self, df: pd.DataFrame, 
                    y: Optional[pd.DataFrame] = None, 
                    kind: str = 'nodes', 
                    min_dist: Union[str, float, int] = 'auto', 
                    n_neighbors: int = 7,
                    merge_policy: bool = False,
                    sample: Optional[int] = None, 
                    return_graph: bool = True,
                    fit_umap_embedding: bool = True,
                    umap_transform_kwargs: Dict[str, Any] = {}
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Plottable]:
        """Transforms data into UMAP embedding
        
        Args:
            :df: Dataframe to transform
            :y: Target column
            :kind: One of `nodes` or `edges`
            :min_dist: Epsilon for including neighbors in infer_graph
            :n_neighbors: Number of neighbors to use for contextualization
            :merge_policy: if True, use previous graph, adding new batch to existing graph's neighbors
                useful to contextualize new data against existing graph. If False, `sample` is irrelevant.

            sample: Sample number of existing graph's neighbors to use for contextualization -- helps make denser graphs
            return_graph: Whether to return a graph or just the embeddings
            fit_umap_embedding: Whether to infer graph from the UMAP embedding on the new data, default True
        """
        df, y = make_safe_gpu_dataframes(df, y, 'pandas')
        X, y_ = self.transform(df, y, kind=kind, return_graph=False)
        X, y_ = make_safe_gpu_dataframes(X, y_, self.engine)  # type: ignore
        emb = self._umap.transform(X, **umap_transform_kwargs)  # type: ignore
        emb = self._bundle_embedding(emb, index=df.index)
        if return_graph and kind not in ["edges"]:
            emb, _ = make_safe_gpu_dataframes(emb, None, 'pandas')  # for now so we don't have to touch infer_edges, force to pandas
            X, y_ = make_safe_gpu_dataframes(X, y_, 'pandas')
            g = self._infer_edges(emb, X, y_, df, 
                                  infer_on_umap_embedding=fit_umap_embedding, merge_policy=merge_policy,
                                  eps=min_dist, sample=sample, n_neighbors=n_neighbors) 
            return g
        return emb, X, y_

    def _bundle_embedding(self, emb, index):
        # Converts Embedding into dataframe and takes care if emb.dim > 2
        if emb.shape[1] == 2 and 'cudf.core.dataframe' not in str(getmodule(emb)) and not hasattr(emb, 'device'):
            emb = pd.DataFrame(emb, columns=[config.X, config.Y], index=index)
        elif emb.shape[1] == 2 and 'cudf.core.dataframe' in str(getmodule(emb)):
            emb.rename(columns={0: config.X, 1: config.Y}, inplace=True)
        elif emb.shape[1] == 2 and hasattr(emb, 'device'):
            import cudf
            emb = cudf.DataFrame(emb, columns=[config.X, config.Y], index=index)
        else:
            columns = [config.X, config.Y] + [
                f"umap_{k}" for k in range(2, emb.shape[1])
            ]
            if 'cudf.core.dataframe' not in str(getmodule(emb)):
                emb = pd.DataFrame(emb, columns=columns, index=index)
            elif 'cudf.core.dataframe' in str(getmodule(emb)):
                emb.columns = columns
        return emb

    def _process_umap(
        self,
        res,
        X_: pd.DataFrame,
        y_: pd.DataFrame,
        kind,
        memoize: bool,
        featurize_kwargs,
        **rest: Any
    ):
        """
        Returns res mutated with new _xy
        """

        umap_kwargs = rest.pop('umap_kwargs', {})
        umap_fit_kwargs = rest.pop('umap_fit_kwargs', {})
        umap_transform_kwargs = rest.pop('umap_transform_kwargs', {})

        umap_params = {**rest, **umap_kwargs}

        umap_kwargs_reuse = {
            "kind": kind,
            "X": X_,
            "y": y_,
            "featurize_kwargs": featurize_kwargs or {},
            "umap_params": umap_params,
            "umap_fit_kwargs": umap_fit_kwargs,
            "umap_transform_kwargs": umap_transform_kwargs,
        }

        old_res = reuse_umap(res, memoize, umap_kwargs_reuse)
        if old_res:
            logger.debug(" --- [[ RE-USING UMAP ]], umap previous n_components: %s", umap_params['n_components'])
            fresh_res = copy.copy(res)
            for attr in ["_xy", "_weighted_edges_df", "_weighted_adjacency"]:
                if hasattr(old_res, attr):
                    setattr(fresh_res, attr, getattr(old_res, attr))
            # have to set _raw_data attribute on umap?
            fresh_res._umap = old_res._umap  # this saves the day!
            fresh_res._umap_params = umap_params
            fresh_res._umap_fit_kwargs = umap_fit_kwargs
            fresh_res._umap_transform_kwargs = umap_transform_kwargs
            return fresh_res

        logger.debug('** Fitting UMAP')
        res = res.umap_lazy_init(
            res,
            **rest,
            umap_kwargs=umap_kwargs,
            umap_fit_kwargs=umap_fit_kwargs,
            umap_transform_kwargs=umap_transform_kwargs)
        
        emb = res._umap_fit_transform(X_, y_, umap_fit_kwargs, umap_transform_kwargs)
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
        X: XSymbolic = None,
        y: YSymbolic = None,
        kind: str = "nodes",
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
        dbscan: bool = False,
        engine: UMAPEngine = "auto",
        feature_engine: str = "auto",
        inplace: bool = False,
        memoize: bool = True,
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
        **featurize_kwargs,
    ):
        """UMAP the featurized nodes or edges data, or pass in your own X, y (optional) dataframes of values
        
        Example

        >>> import graphistry   
        >>> g = graphistry.nodes(pd.DataFrame({'node': [0,1,2], 'data': [1,2,3], 'meta': ['a', 'b', 'c']}))
        >>> g2 = g.umap(n_components=3, spread=1.0, min_dist=0.1, n_neighbors=12, negative_sample_rate=5, local_connectivity=1, repulsion_strength=1.0, metric='euclidean', suffix='', play=0, encode_position=True, encode_weight=True, dbscan=False, engine='auto', feature_engine='auto', inplace=False, memoize=True)
        >>> g2.plot()
        
        Parameters
 
            :X: either a dataframe ndarray of features, or column names to featurize
            :y: either an dataframe ndarray of targets, or column names to featurize
                    targets
            :kind: `nodes` or `edges` or None.
                    If None, expects explicit X, y (optional) matrices,
                    and will Not associate them to nodes or edges.
                    If X, y (optional) is given, with kind = [nodes, edges],
                    it will associate new matrices to nodes or edges attributes.
            :scale: multiplicative scale for pruning weighted edge DataFrame
                    gotten from UMAP, between [0, ..) with high end meaning keep
                    all edges
            :n_neighbors: UMAP number of nearest neighbors to include for
                    UMAP connectivity, lower makes more compact layouts. Minimum 2
            :min_dist: UMAP float between 0 and 1, lower makes more compact
                    layouts.
            :spread: UMAP spread of values for relaxation
            :local_connectivity: UMAP connectivity parameter
            :repulsion_strength: UMAP repulsion strength
            :negative_sample_rate: UMAP negative sampling rate
            :n_components: number of components in the UMAP projection,
                    default 2
            :metric: UMAP metric, default 'euclidean'.
                    see (UMAP-LEARN)[https://umap-learn.readthedocs.io/
                    en/latest/parameters.html] documentation for more.
            :suffix: optional suffix to add to x, y attributes of umap.
            :play: Graphistry play parameter, default 0, how much to evolve
                    the network during clustering. 0 preserves the original UMAP layout.
            :encode_weight: if True, will set new edges_df from
                    implicit UMAP, default True.
            :encode_position: whether to set default plotting bindings
                    -- positions x,y from umap for .plot(), default True
            :dbscan: whether to run DBSCAN on the UMAP embedding, default False.
            :engine: selects which engine to use to calculate UMAP:
                    default "auto" will use cuML if available, otherwise UMAP-LEARN.
            :feature_engine: How to encode data
                    ("none", "auto", "pandas", "dirty_cat", "torch")
            :inplace: bool = False, whether to modify the current object, default False.
                    when False, returns a new object, useful for chaining in a functional paradigm.
            :memoize: whether to memoize the results of this method,
                    default True.
            :umap_kwargs: Optional kwargs to pass to underlying UMAP library constructor
            :umap_fit_kwargs: Optional kwargs to pass to underlying UMAP fit method, including fit part of fit_transform
            :umap_transform_kwargs: Optional kwargs to pass to underlying UMAP transform method, including transform part of fit_transform
            :featurize_kwargs: Optional kwargs to pass to .featurize()

        :return: self, with attributes set with new data
        """
        if engine == UMAP_LEARN:
            assert_imported()
        elif engine == CUML:
            assert_imported_cuml()

        umap_kwargs_combined = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            engine=engine,
            suffix=suffix,
            umap_kwargs=umap_kwargs,
            umap_fit_kwargs=umap_fit_kwargs,
            umap_transform_kwargs=umap_transform_kwargs,
        )
        logger.debug("umap_kwargs: %s", umap_kwargs_combined)

        # temporary until we have full cudf support in feature_utils.py
        has_cudf, _, cudf = lazy_cudf_import()

        if inplace:
            res = self
        else:
            res = self.bind()

        if has_cudf:
            flag_nodes_cudf = isinstance(self._nodes, cudf.DataFrame)
            flag_edges_cudf = isinstance(self._edges, cudf.DataFrame)

            #if flag_nodes_cudf or flag_edges_cudf:
            if False:
                if flag_nodes_cudf:
                    res._nodes = res._nodes.to_pandas()
                if flag_edges_cudf:
                    res._edges = res._edges.to_pandas()
                if (X is not None) or (y is not None):
                    res = res.umap(X=X, y=y, kind=kind, feature_engine=feature_engine, **umap_kwargs_combined, **featurize_kwargs)  # type: ignore
                else:
                    res = res.umap(X=self._nodes, y=self._edges, kind=kind, feature_engine=feature_engine, **umap_kwargs_combined, **featurize_kwargs)  # type: ignore
                return res

        res = res.umap_lazy_init(
            res,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            n_components=n_components,
            metric=metric,
            engine=engine,
            suffix=suffix,
            umap_kwargs=umap_kwargs,
            umap_fit_kwargs=umap_fit_kwargs,
            umap_transform_kwargs=umap_transform_kwargs
        )

        logger.debug("umap input X :: %s", X)
        logger.debug("umap input y :: %s", y)

        featurize_kwargs = self._set_features(
            res, X, y, kind, feature_engine, {**featurize_kwargs, "memoize": memoize}
        )

        if kind == "nodes":
            index = res._nodes.index
            if res._node is None:
                logger.debug("-Writing new node name")
                res = res.nodes(  # type: ignore
                    res._nodes.reset_index(drop=True)
                    .reset_index()
                    .rename(columns={"index": config.IMPLICIT_NODE_ID}),
                    config.IMPLICIT_NODE_ID,
                )
                res._nodes.index = index

            nodes = res._nodes[res._node].values

            logger.debug("propagating with featurize_kwargs: %s", featurize_kwargs)
            (
                X_,
                y_,
                res,
            ) = res.featurize_or_get_nodes_dataframe_if_X_is_None(  # type: ignore
                **featurize_kwargs
            )

            logger.debug("umap X_ (%s): %s", type(X_), X_)
            logger.debug("umap y_ (%s): %s", type(y_), y_)
            logger.debug("data is type :: %s", (type(X_)))
            if isinstance(X_, pd.DataFrame):
                index_to_nodes_dict = dict(zip(range(len(nodes)), nodes))
            elif 'cudf.core.dataframe' in str(getmodule(X_)):
                assert isinstance(X_, cudf.DataFrame)
                logger.debug('nodes type: %s', type(nodes))
                import cupy as cp
                index_to_nodes_dict = dict(zip(range(len(nodes)), cp.asnumpy(nodes)))

            # add the safe coercion here 
            X_, y_ = make_safe_gpu_dataframes(X_, y_, res.engine)  # type: ignore

            res = res._process_umap(
                res, X_, y_, kind, memoize, featurize_kwargs, **umap_kwargs_combined
            )

            res._weighted_adjacency_nodes = res._weighted_adjacency
            if res._xy is None:
                raise RuntimeError("This should not happen")
            res._node_embedding = res._xy
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
            ) = res.featurize_or_get_edges_dataframe_if_X_is_None(  # type: ignore
                **featurize_kwargs
            )

            # add the safe coercion here 
            X_, y_ = make_safe_gpu_dataframes(X_, y_, res.engine)  # type: ignore

            res = res._process_umap(
                res, X_, y_, kind, memoize, featurize_kwargs, **umap_kwargs_combined
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
            if X is not None and isinstance(X, pd.DataFrame) or '':
                logger.info("New Matrix `X` passed in for UMAP-ing")
                xy = res._umap_fit_transform(X, y, umap_fit_kwargs, umap_transform_kwargs)
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
                    "must be given."
                )
        else:
            raise ValueError(
                f"`kind` needs to be one of `nodes`, `edges`, `None`, got {kind}"  # noqa: E501
            )
        res = res._bind_xy_from_umap(
            res, kind, encode_position, encode_weight, play
        )  # noqa: E501

        if res.engine == CUML and is_legacy_cuml():  # type: ignore
            res = res.prune_self_edges()

        if dbscan:
            res = res.dbscan(min_dist=min_dist, kind=kind, fit_umap_embedding=True)  # type: ignore

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
        x_name = config.X + res._suffix
        y_name = config.Y + res._suffix
        if kind == "nodes":
            emb = res._node_embedding
        else:
            emb = res._edge_embedding
            
        if isinstance(df, type(emb)):
            df[x_name] = emb.values.T[0]
            df[y_name] = emb.values.T[1]
        elif isinstance(df, pd.DataFrame) and 'cudf.core.dataframe' in str(getmodule(emb)):
            df[x_name] = emb.to_numpy().T[0]
            df[y_name] = emb.to_numpy().T[1]

        res = res.nodes(df) if kind == "nodes" else res.edges(df)

        if encode_weight and kind == "nodes":
            # adds the implicit edge dataframe and binds it to
            # graphistry instance
            w_name = config.WEIGHT + res._suffix
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
