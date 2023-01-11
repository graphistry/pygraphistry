import logging
import pandas as pd
import numpy as np

from typing import Any, List, Union, TYPE_CHECKING, Tuple, Optional, Dict, Callable
from typing_extensions import Literal
from collections import Counter

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.constants import CUML, UMAP_LEARN  # noqa type: ignore
from graphistry.features import ModelDict
from graphistry.feature_utils import get_matrix_by_column_parts

logger = logging.getLogger("compute.cluster")

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object

DBSCANEngineConcrete = Literal["cuml", "umap_learn"]
DBSCANEngine = Literal[DBSCANEngineConcrete, "auto"]


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


def resolve_cpu_gpu_engine(
    engine: DBSCANEngine,
) -> DBSCANEngineConcrete:  # noqa
    if engine in [CUML, UMAP_LEARN]:
        return engine  # type: ignore
    if engine in ["auto"]:
        (
            has_min_dependency,
            _,
            has_cuml_dependency,
            _,
        ) = lazy_dbscan_import_has_dependency()
        if has_cuml_dependency:
            return "cuml"
        if has_min_dependency:
            return "umap_learn"

    raise ValueError(  # noqa
        f'engine expected to be "auto", '
        '"umap_learn", or  "cuml" '
        f"but received: {engine} :: {type(engine)}"
    )


def get_model_matrix(g, kind, cols, umap, target):
    """
        Allows for a single function to get the model matrix for both nodes and edges as well as targets, embeddings, and features

    Args:
        g (_type_): _description_
        kind (_type_): _description_
        cols (_type_): _description_
        umap (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: dataframe of model matrix given the inputs
    """
    assert kind in ["nodes", "edges"]
    assert (
        hasattr(g, "_node_encoder") if kind == "nodes" else hasattr(g, "_edge_encoder")
    )


    df = g.get_features_by_cols(cols, kind=kind, target=target)

    if umap and cols is None and g._umap is not None:
        df = g._get_embedding(kind)

    return df


def dbscan_fit(g, dbscan, kind="nodes", cols=None, use_umap_embedding=True, target=False):
    """
    Fits clustering on UMAP embeddings if umap is True, otherwise on the features dataframe
        or target dataframe if target is True.

    args:
        g: graphistry graph
        kind: 'nodes' or 'edges'
        cols: list of columns to use for clustering given `g.featurize` has been run
        umap: whether to use UMAP embeddings or features dataframe
    """
    X = get_model_matrix(g, kind, cols, use_umap_embedding, target)
    
    if X.empty:
        raise ValueError("No features found for clustering")

    dbscan.fit(X)
    labels = dbscan.labels_

    if kind == "nodes":
        g._nodes = g._nodes.assign(_cluster=labels)
    elif kind == "edges":
        g._edges = g._edges.assign(_cluster=labels)
    else:
        raise ValueError("kind must be one of `nodes` or `edges`")

    kind = "node" if kind == "nodes" else "edge"
    setattr(g, f"_{kind}_dbscan", dbscan)

    return g


def dbscan_predict(X: pd.DataFrame, model):
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


# def dbscan_predict2(g, kind="nodes", cols=None, umap=True):
#     X = g._get_feature(kind)
#     dbscan = g._node_dbscan if kind == "nodes" else g._edge_dbscan

#     preds = dbscan_predict(X, dbscan)
#     return X, preds


class ClusterMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass

    def _cluster_dbscan(
        self, res, kind, cols, fit_umap_embedding, target, eps, min_samples, *args, **kwargs
    ):
        """
        DBSCAN clustering on cpu or gpu infered by .engine flag
        """
        _, DBSCAN, _, cuDBSCAN = lazy_dbscan_import_has_dependency()

        res.engine = resolve_cpu_gpu_engine("auto")
        res._kwargs_dbscan = ModelDict(
            "latest dbscan kwargs",
            kind=kind,
            cols=cols,
            target=target,
            umap=fit_umap_embedding,
            eps=eps,
            min_samples=min_samples,
            *args,
            **kwargs,
        )

        dbscan = (
            cuDBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            if res.engine == CUML
            else DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        )

        res = dbscan_fit(
            res, dbscan, kind=kind, cols=cols, use_umap_embedding=fit_umap_embedding
        )

        return res

    def dbscan(
        self,
        eps: float = 0.2,
        min_samples: int = 1,
        cols=None,
        kind="nodes",
        fit_umap_embedding=True,
        target=False,
        **kwargs,
    ):
        """DBSCAN clustering on cpu or gpu infered automatically

        Examples:
            g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')

            # cluster by UMAP embeddings
            kind = 'nodes' | 'edges'
            g2 = g.umap(kind=kind).dbscan(kind=kind)
            print(g2._nodes['_dbscan']) | print(g2._edges['_dbscan'])

            # dbscan with fixed parameters is default in umap
            g2 = g.umap(dbscan=True)

            # and with greater control over parameters via chaining,
            g2 = g.umap().dbscan(eps=1.2, min_samples=2, **kwargs)

            # cluster by feature embeddings
            g2 = g.featurize().dbscan(**kwargs)

            # cluster by a given set of feature column attributes
            g2 = g.featurize().dbscan(cols=['ip_172', 'location', 'alert'], **kwargs)

            # equivalent to above (ie, cols != None and umap=True will still use features dataframe, rather than UMAP embeddings)
            g2 = g.umap().dbscan(cols=['ip_172', 'location', 'alert'], umap=True | False, **kwargs)

            g2.plot() # color by `_dbscan`

        Useful:
            Enriching the graph with cluster labels from UMAP is useful for visualizing clusters in the graph by color, size, etc,
            as well as assessing metrics per cluster, e.g.
             https://github.com/graphistry/pygraphistry/blob/master/demos/ai/cyber/cyber-redteam-umap-demo.ipynb

        Args:
            eps float: The maximum distance between two samples for them to be considered as in the same neighborhood.
            kind str: 'nodes' or 'edges'
            cols: list of columns to use for clustering given `g.featurize` has been run, nice way to slice features by
                fragments of interest, e.g. ['ip_172', 'location', 'ssh', 'warnings']
            fit_umap_embedding bool: whether to use UMAP embeddings or features dataframe to cluster DBSCAN
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
                This includes the point itself.

        """

        res = self.bind()
        res = res._cluster_dbscan(
            res,
            kind=kind,
            cols=cols,
            fit_umap_embedding=fit_umap_embedding,
            target=target,
            eps=eps,
            min_samples=min_samples,
            **kwargs,
        )

        return res

    def _transform_dbscan(
        self, df: pd.DataFrame, ydf=None, kind: str = "nodes"
    ) -> Tuple[Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        """
        Transforms a dataframe to one with a new column '_dbscan' containing the DBSCAN cluster labels
            and returns feature[cols] or UMAP embedding
        Examples:
            fit:
                g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')
                g2 = g.featurize().dbscan()

            predict:
                emb, X, y, ndf = g2.transform_dbscan(ndf, return_graph=False)
                # or
                g3 = g2.transform_dbscan(ndf, return_graph=True)
                g3.plot()

        likewise for umap:
            fit:
                g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node')
                g2 = g.umap().dbscan()

            predict:
                emb, X, y, ndf = g2.transform_dbscan(ndf, return_graph=False)
                # or
                g3 = g2.transform_dbscan(ndf, return_graph=True)
                g3.plot()

        args:
            df: dataframe to transform
            ydf: optional labels dataframe
            kind: 'nodes' or 'edges'

        """

        res = self.bind()
        if hasattr(res, "_kwargs_dbscan"):
            # Assume that we are transforming to last fit of dbscan
            cols = res._kwargs_dbscan["cols"]
            umap = res._kwargs_dbscan["umap"]

            dbscan = res._node_dbscan if kind == "nodes" else res._edge_dbscan

            emb = None
            if umap and cols is None:
                emb, X, y = res.transform_umap(df, ydf, kind=kind, return_graph=False)
            else:
                X, y = res.transform(df, ydf, kind=kind, return_graph=False)
                if cols is not None:
                    X = get_matrix_by_column_parts(X, cols)

            if umap:
                X_ = emb
            else:
                X_ = X

            labels = dbscan_predict(X_, dbscan)  # type: ignore
            if umap and cols is None:
                df = df.assign(_dbscan=labels, x=emb.x, y=emb.y)  # type: ignore
            else:
                df = df.assign(_dbscan=labels)

            return emb, X, y, df  # type: ignore
        else:
            raise Exception("No dbscan model found. Please run `g.dbscan()` first")

    def transform_dbscan(
        self,
        df: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        eps: Union[float, str] = "auto",
        fit_umap_embedding: bool = False,
        sample: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        kind: str = "nodes",
        return_graph=True,
        verbose=False,
        ):  # type: ignore
        """
        Transforms a minibatch dataframe to one with a new column '_dbscan' containing the DBSCAN cluster labels on the minibatch
            and generates a graph with the minibatch and the original graph, with edges between the minibatch and the original graph inferred
            from the umap embedding or features dataframe.

        args:
            df: dataframe to transform
            y: optional labels dataframe
            eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
                smaller values will result in less edges between the minibatch and the original graph.
                Default 'auto', infers eps from the mean distance and std of new points to the original graph
            fit_umap_embedding: whether to use UMAP embeddings or features dataframe when inferring edges between
                the minibatch and the original graph. Default False, uses the features dataframe
            sample: number of samples to use when inferring edges between the minibatch and the original graph,
                if None, will only use closest point to the minibatch. If greater than 0, will sample the closest `sample` points
                in existing graph to pull in more edges. Default None
            kind: 'nodes' or 'edges'
            return_graph: whether to return a graph or the (emb, X, y, minibatch df enriched with DBSCAN labels), default True

        """
        emb, X, y, df = self._transform_dbscan(df, y, kind=kind)
        if return_graph and kind not in ["edges"]:
            g = self._infer_edges(emb, X, y, df, eps=eps, sample=sample, n_neighbors=n_neighbors,  # type: ignore
                infer_on_umap_embedding=fit_umap_embedding, 
                verbose=verbose
            )
            return g
        return emb, X, y, df
