from typing import Any, List, Union, TYPE_CHECKING, Tuple, Optional, cast
from typing_extensions import Literal
from collections import Counter
from inspect import getmodule
import numpy as np
import pandas as pd
import logging
import warnings

from graphistry.Engine import Engine, resolve_engine
from graphistry.models.compute.dbscan import (
    DBSCANEngine, DBSCANEngineAbstract,
    dbscan_engine_values
)
from graphistry.models.compute.features import GraphEntityKind, graph_entity_kind_values
from graphistry.Plottable import Plottable
from graphistry.constants import CUML, DBSCAN
from graphistry.models.ModelDict import ModelDict
from graphistry.feature_utils import get_matrix_by_column_parts
from graphistry.utils.lazy_import import lazy_dbscan_import
from graphistry.util import setup_logger

logger = setup_logger("compute.cluster")

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


def resolve_dbscan_engine(
    engine: DBSCANEngineAbstract,
    g_or_df: Optional[Any] = None
) -> DBSCANEngine:
    """
    Resolves the engine to use for DBSCAN clustering

    If 'auto', decide by checking if cuml or sklearn is installed, and if provided, natural type of the dataset. GPU is used if both a GPU dataset and GPU library is installed. Otherwise, CPU library.
    """
    if engine in dbscan_engine_values:
        return engine  # type: ignore
    if engine == "umap_learn":
        warnings.warn("engine value 'umap_learn' is deprecated, use engine='cuml' or 'sklearn' instead; defaulting to sklearn")
        return "sklearn"
    if engine == "auto":

        preferred_engine = None if g_or_df is None else resolve_engine('auto', g_or_df)
        if preferred_engine in [Engine.DASK, Engine.DASK_CUDF]:
            raise ValueError('dask not supported for DBSCAN clustering, .compute() values first')
        assert preferred_engine in [None, Engine.PANDAS, Engine.CUDF]

        (
            has_min_dependency,
            _,
            has_cuml_dependency,
            _,
        ) = lazy_dbscan_import()
        if has_cuml_dependency and preferred_engine in [None, 'cudf']:
            return "cuml"
        if has_min_dependency:
            return "sklearn"

    raise ValueError(f'Engine expected to be "auto" with cuml/sklearn installed, "sklearn", or "cuml", but received: {engine} :: {type(engine)}')

def make_safe_gpu_dataframes(
    X: Optional[Any], y: Optional[Any], engine: Engine
) -> Tuple[Optional[Any], Optional[Any]]:
    """Coerce a dataframe to pd vs cudf based on engine"""

    assert engine in [Engine.PANDAS, Engine.CUDF], f"Expected engine to be 'pandas' or 'cudf', got {engine}"

    def df_as_dbscan_engine(df: Optional[Any], engine: Engine) -> Optional[Any]:
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and engine == Engine.CUDF:
            import cudf
            return cudf.from_pandas(df)
        elif 'cudf' in str(getmodule(df)) and engine == Engine.PANDAS:
            return df.to_pandas()
        return df
    
    return df_as_dbscan_engine(X, engine), df_as_dbscan_engine(y, engine)


def get_model_matrix(
    g: Plottable, kind: GraphEntityKind, cols: Optional[Union[List, str]], umap, target
) -> Any:
    """
        Allows for a single function to get the model matrix for both nodes and edges as well as targets, embeddings, and features

    Args:
            :g: graphistry graph
            :kind: 'nodes' or 'edges'
            :cols: list of columns to use for clustering given `g.featurize` has been run
            :umap: whether to use UMAP embeddings or features dataframe
            :target: whether to use the target dataframe or features dataframe

    Returns:
        pd.DataFrame: dataframe of model matrix given the inputs
    """
    assert kind in graph_entity_kind_values, f'Expected kind of {graph_entity_kind_values}, got: {kind}'
    assert (
        hasattr(g, "_node_encoder") if kind == "nodes" else hasattr(g, "_edge_encoder")
    )

    engine = g._dbscan_engine
    assert engine is not None, 'DBSCAN engine not set'

    df_engine: Engine = Engine.CUDF if engine == 'cuml' else Engine.PANDAS

    ###

    from graphistry.feature_utils import FeatureMixin
    assert isinstance(g, FeatureMixin)

    # TODO does get_matrix do cudf?
    df = g.get_matrix(cols, kind=kind, target=target)

    # TODO does _get_embedding do cudf?
    if umap and cols is None and g._umap is not None:
        from graphistry.umap_utils import UMAPMixin
        assert isinstance(g, UMAPMixin)
        df = g._get_embedding(kind)
    
    df2, _ = make_safe_gpu_dataframes(df, None, df_engine)

    return df2


def dbscan_fit_inplace(
    res: Plottable, dbscan: Any, kind: GraphEntityKind = "nodes",
    cols: Optional[Union[List, str]] = None, use_umap_embedding: bool = True,
    target: bool = False, verbose: bool = False
) -> None:
    """
    Fits clustering on UMAP embeddings if umap is True, otherwise on the features dataframe
        or target dataframe if target is True.

    Sets:
        - `res._dbscan_edges` or `res._dbscan_nodes` to the DBSCAN model
        - `res._edges` or `res._nodes`  gains column `_dbscan`

    Args:
        :res: graphistry graph
        :kind: 'nodes' or 'edges'
        :cols: list of columns to use for clustering given `g.featurize` has been run
        :use_umap_embedding: whether to use UMAP embeddings or features dataframe for clustering (default: True)
        :target: whether to use the target dataframe or features dataframe (typically False, for features)
    """
    X = get_model_matrix(res, kind, cols, use_umap_embedding, target)
    
    if X.empty:
        raise ValueError("No features found for clustering")

    logger.debug('dbscan_fit dbscan: %s', str(getmodule(dbscan)))

    labels: np.ndarray
    if res._dbscan_engine == 'cuml':
        import cupy as cp
        from cuml import DBSCAN
        assert isinstance(dbscan, DBSCAN), f'Expected cuml.DBSCAN, got: {type(dbscan)}'
        dbscan.fit(X, calc_core_sample_indices=True)
        labels = dbscan.labels_
        core_sample_indices = dbscan.core_sample_indices_

        # Convert core_sample_indices_ to cupy if it's not already
        # (Sometimes it's already cupy; if it's a CumlArray, we can cast or just index directly)
        core_sample_indices_cupy = core_sample_indices.astype(cp.int32)

        # The actual core-sample points (a.k.a. "components_" in sklearn terms)
        components = X[core_sample_indices_cupy]
        dbscan.components_ = components

        # dbscan.components_ = X[dbscan.core_sample_indices_.to_pandas()]  # can't believe len(samples) != unique(labels) ... #cumlfail
    else:
        from sklearn.cluster import DBSCAN
        assert isinstance(dbscan, DBSCAN), f'Expected sklearn.DBSCAN, got: {type(dbscan)}'
        dbscan.fit(X)
        labels = dbscan.labels_

    if kind == "nodes":
        res._nodes = res._nodes.assign(_dbscan=labels)
        res._dbscan_nodes = dbscan
    elif kind == "edges":
        res._edges = res._edges.assign(_dbscan=labels)
        res._dbscan_edges = dbscan
    else:
        raise ValueError(f"kind must be one of `nodes` or `edges`, got {kind}")

    setattr(res, f"_{kind}_dbscan", dbscan)
    
    if cols is not None:  # set False since we used the features for verbose
        use_umap_embedding = False

    if verbose:
        cnt = Counter(labels)
        message = f"DBSCAN found {len(cnt)} clusters with {cnt[-1]} outliers"
        logger.debug(message)
        logger.debug(f"--fit on {'umap embeddings' if use_umap_embedding else 'feature embeddings'} of size {X.shape} :: {X.dtypes}")


# TODO what happens in gpu mode?
def dbscan_predict_sklearn(X: pd.DataFrame, model: Any) -> np.ndarray:
    """
    DBSCAN has no predict per se, so we reverse engineer one here
    from https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    
    """
    n_samples = X.shape[0]

    y_new = np.ones(shape=n_samples, dtype=int) * -1

    for i in range(n_samples):
        diff = model.components_ - X.iloc[i, :].values  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new

def dbscan_predict_cuml(X: Any, model: Any) -> Any:

    import cudf
    import cupy as cp
    from sklearn.cluster import DBSCAN as skDBSCAN
    from cuml import DBSCAN
    #assert isinstance(X, cudf.DataFrame), f'Expected cudf.DataFrame, got: {type(X)}'
    if isinstance(X, cudf.DataFrame):
        X = X.to_pandas()
    
    if isinstance(X, pd.DataFrame) and isinstance(model, skDBSCAN):
        return dbscan_predict_sklearn(X, model)

    assert isinstance(model, DBSCAN), f'Expected cuml.DBSCAN, got: {type(model)}'

    #raise NotImplementedError('cuml lacks predict, and for cpu fallback, components_')
    warnings.warn('cuml lacks predict, cpu fallback, components_')

    n_samples = X.shape[0]

    y_new = np.ones(shape=n_samples, dtype=int) * -1

    components = model.components_.to_pandas() if isinstance(model.components_, cudf.DataFrame) else model.components_

    for i in range(n_samples):
        diff = components - X.iloc[i, :].values  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new




class ClusterMixin(MIXIN_BASE):
    
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def _cluster_dbscan(
        self, kind: GraphEntityKind, cols, fit_umap_embedding, target, min_dist, min_samples, engine_dbscan: DBSCANEngineAbstract, verbose, *args, **kwargs
    ):
        """DBSCAN clustering on cpu or gpu infered by .engine flag
        """
        _, DBSCAN, _, cuDBSCAN = lazy_dbscan_import()

        res = self.bind()

        engine_dbscan = resolve_dbscan_engine(engine_dbscan, res)

        if engine_dbscan in [CUML]:
            warnings.warn('`_cluster_dbscan(..)` experimental')
            #engine_dbscan = 'sklearn'

        dbscan_engine = cuDBSCAN if engine_dbscan == CUML else DBSCAN

        res._dbscan_engine = engine_dbscan
        res._dbscan_params = ModelDict(
            "latest DBSCAN params",
            kind=kind,
            cols=cols,
            target=target,
            fit_umap_embedding=fit_umap_embedding,
            min_dist=min_dist,
            min_samples=min_samples,
            engine_dbscan=engine_dbscan,
            verbose=verbose,
        )

        dbscan = dbscan_engine(eps=min_dist, min_samples=min_samples, *args, **kwargs)
        dbscan_fit_inplace(res, dbscan, kind=kind, cols=cols, use_umap_embedding=fit_umap_embedding, verbose=verbose)

        return res

    def dbscan(
        self,
        min_dist: float = 0.2,
        min_samples: int = 1,
        cols: Optional[Union[List, str]] = None,
        kind: GraphEntityKind = "nodes",
        fit_umap_embedding: bool = True,
        target: bool = False,
        verbose: bool = False,
        engine_dbscan: DBSCANEngineAbstract = 'auto',
        *args,
        **kwargs,
    ):
        """DBSCAN clustering on cpu or gpu infered automatically. Adds a `_dbscan` column to nodes or edges.
           NOTE: g.transform_dbscan(..) currently unsupported on GPU.

           Saves model as g._dbscan_nodes or g._dbscan_edges

        Examples:
        ::

            g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')

            # cluster by UMAP embeddings
            kind = 'nodes' | 'edges'
            g2 = g.umap(kind=kind).dbscan(kind=kind)
            print(g2._nodes['_dbscan']) | print(g2._edges['_dbscan'])

            # dbscan in umap or featurize API
            g2 = g.umap(dbscan=True, min_dist=1.2, min_samples=2, **kwargs)
            # or, here dbscan is infered from features, not umap embeddings
            g2 = g.featurize(dbscan=True, min_dist=1.2, min_samples=2, **kwargs)

            # and via chaining,
            g2 = g.umap().dbscan(min_dist=1.2, min_samples=2, **kwargs)

            # cluster by feature embeddings
            g2 = g.featurize().dbscan(**kwargs)

            # cluster by a given set of feature column attributes, or with target=True
            g2 = g.featurize().dbscan(cols=['ip_172', 'location', 'alert'], target=False, **kwargs)

            # equivalent to above (ie, cols != None and umap=True will still use features dataframe, rather than UMAP embeddings)
            g2 = g.umap().dbscan(cols=['ip_172', 'location', 'alert'], umap=True | False, **kwargs)

            g2.plot() # color by `_dbscan` column

        Useful:
            Enriching the graph with cluster labels from UMAP is useful for visualizing clusters in the graph by color, size, etc, as well as assessing metrics per cluster, e.g. https://github.com/graphistry/pygraphistry/blob/master/demos/ai/cyber/cyber-redteam-umap-demo.ipynb

        Args:
            :min_dist float: The maximum distance between two samples for them to be considered as in the same neighborhood.
            :kind str: 'nodes' or 'edges'
            :cols: list of columns to use for clustering given `g.featurize` has been run, nice way to slice features or targets by fragments of interest, e.g. ['ip_172', 'location', 'ssh', 'warnings']
            :fit_umap_embedding bool: whether to use UMAP embeddings or features dataframe to cluster DBSCAN
            :min_samples: The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself.
            :target: whether to use the target column as the clustering feature

        """

        res = self._cluster_dbscan(
            kind=kind,
            cols=cols,
            fit_umap_embedding=fit_umap_embedding,
            target=target,
            min_dist=min_dist,
            min_samples=min_samples,
            engine_dbscan=engine_dbscan,
            verbose=verbose,
            *args,
            **kwargs,
        )   # type: ignore

        return res

    def _transform_dbscan(
        self, df: pd.DataFrame, ydf, kind, verbose
    ) -> Tuple[Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        res = self.bind()
        if hasattr(res, "_dbscan_params") and res._dbscan_params is not None:
            # Assume that we are transforming to last fit of dbscan
            cols = res._dbscan_params["cols"]
            umap = res._dbscan_params["fit_umap_embedding"]
            target = res._dbscan_params["target"]

            dbscan = res._dbscan_nodes if kind == "nodes" else res._dbscan_edges
            # print('DBSCAN TYPE IN TRANSFORM', type(dbscan))

            emb = None
            if umap and cols is None:
                emb, X, y = res.transform_umap(df, ydf, kind=kind, return_graph=False)
            else:
                X, y = res.transform(df, ydf, kind=kind, return_graph=False)
            XX = X
            if target:
                XX = y
            if cols is not None:
                XX = get_matrix_by_column_parts(XX, cols)

            if umap:
                X_ = emb
            else:
                X_ = XX
            
            if res._dbscan_engine == 'cuml':
                print('Transform DBSCAN not yet supported for engine_dbscan=`cuml`, use engine=`umap_learn`, `pandas` or `sklearn` instead')
                return emb, X, y, df
            
            X_, emb = make_safe_gpu_dataframes(X_, emb, Engine.PANDAS)  

            labels = dbscan_predict_cuml(X_, dbscan)  # type: ignore
            #print('after dbscan predict', type(labels))
            if umap and cols is None:
                df = df.assign(_dbscan=labels, x=emb.x, y=emb.y)  # type: ignore
            else:
                df = df.assign(_dbscan=labels)
            
            if verbose:
                print(f"Transformed DBSCAN: {len(df[DBSCAN].unique())} clusters")

            return emb, X, y, df  # type: ignore
        else:
            raise Exception("No dbscan model found. Please run `g.dbscan()` first")

    def transform_dbscan(
        self,
        df: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        min_dist: Union[float, str] = "auto",
        infer_umap_embedding: bool = False,
        sample: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        kind: str = "nodes",
        return_graph: bool = True,
        verbose: bool = False,
        ):  # type: ignore
        """Transforms a minibatch dataframe to one with a new column '_dbscan' containing the DBSCAN cluster labels on the minibatch and generates a graph with the minibatch and the original graph, with edges between the minibatch and the original graph inferred from the umap embedding or features dataframe. Graph nodes | edges will be colored by '_dbscan' column.
            
            Examples:
            ::

                fit:
                    g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')
                    g2 = g.featurize().dbscan()

                predict:
                ::

                    emb, X, _, ndf = g2.transform_dbscan(ndf, return_graph=False)
                    # or
                    g3 = g2.transform_dbscan(ndf, return_graph=True)
                    g3.plot()

            likewise for umap:
            ::

                fit:
                    g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')
                    g2 = g.umap(X=.., y=..).dbscan()

                predict:
                ::

                    emb, X, y, ndf = g2.transform_dbscan(ndf, ndf, return_graph=False)
                    # or
                    g3 = g2.transform_dbscan(ndf, ndf, return_graph=True)
                    g3.plot()


        Args:
            :df: dataframe to transform
            :y: optional labels dataframe
            :min_dist: The maximum distance between two samples for them to be considered as in the same neighborhood.
                smaller values will result in less edges between the minibatch and the original graph.
                Default 'auto', infers min_dist from the mean distance and std of new points to the original graph
            :fit_umap_embedding: whether to use UMAP embeddings or features dataframe when inferring edges between
                the minibatch and the original graph. Default False, uses the features dataframe
            :sample: number of samples to use when inferring edges between the minibatch and the original graph,
                if None, will only use closest point to the minibatch. If greater than 0, will sample the closest `sample` points
                in existing graph to pull in more edges. Default None
            :kind: 'nodes' or 'edges'
            :return_graph: whether to return a graph or the (emb, X, y, minibatch df enriched with DBSCAN labels), default True
                infered graph supports kind='nodes' only. 
            :verbose: whether to print out progress, default False

        """
        emb, X, y, df = self._transform_dbscan(df, y, kind=kind, verbose=verbose)
        if return_graph and kind not in ["edges"]:
            #raise NotImplementedError("Engine specificity")
            #if 'cudf' in str(getmodule(df)) or 'cudf' in str(getmodule(y)):
            #    warnings.warn("transform_dbscan using cpu fallback")
            #df, y = make_safe_gpu_dataframes(df, y, Engine.PANDAS)
            #X, emb = make_safe_gpu_dataframes(X, emb, Engine.PANDAS)
            engine = self._dbscan_engine
            engine_df = Engine.CUDF if engine == 'cuml' else Engine.PANDAS
            df2, y2 = make_safe_gpu_dataframes(df, y, engine_df)
            X2, emb2 = make_safe_gpu_dataframes(X, emb, engine_df)
            g = self._infer_edges(emb2, X2, y2, df2, eps=min_dist, sample=sample, n_neighbors=n_neighbors,  # type: ignore
                infer_on_umap_embedding=infer_umap_embedding
            )
            return g
        return emb, X, y, df
