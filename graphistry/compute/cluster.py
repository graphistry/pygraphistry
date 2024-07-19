import logging
import pandas as pd
import numpy as np

from typing import Any, List, Union, TYPE_CHECKING, Tuple, Optional
from typing_extensions import Literal
from collections import Counter

from graphistry.Plottable import Plottable
from graphistry.constants import CUML, UMAP_LEARN, DBSCAN  # noqa type: ignore
from graphistry.features import ModelDict
from graphistry.feature_utils import get_matrix_by_column_parts
from graphistry.utils.lazy_import import lazy_cudf_import, lazy_dbscan_import

logger = logging.getLogger("compute.cluster")

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object

DBSCANEngineConcrete = Literal["cuml", "umap_learn"]
DBSCANEngine = Literal[DBSCANEngineConcrete, "auto"]


def resolve_cpu_gpu_engine(
    engine: DBSCANEngine,
) -> DBSCANEngineConcrete:  # noqa
    if engine in [CUML, UMAP_LEARN, 'sklearn']:
        return engine  # type: ignore
    if engine in ["auto"]:
        (
            has_min_dependency,
            _,
            has_cuml_dependency,
            _,
        ) = lazy_dbscan_import()
        if has_cuml_dependency:
            return "cuml"
        if has_min_dependency:
            return "umap_learn"

    raise ValueError(  # noqa
        f'engine expected to be "auto", '
        '"umap_learn", "pandas", "sklearn", or  "cuml" '
        f"but received: {engine} :: {type(engine)}"
    )

def make_safe_gpu_dataframes(X, y, engine):
    """helper method to coerce a dataframe to the correct type (pd vs cudf)"""
    def safe_cudf(X, y):
        new_kwargs = {}
        kwargs = {'X': X, 'y': y}
        for key, value in kwargs.items():
            if isinstance(value, cudf.DataFrame) and engine in ["pandas", 'sklearn', 'umap_learn']:
                new_kwargs[key] = value.to_pandas()
            elif isinstance(value, pd.DataFrame) and engine == "cuml":
                new_kwargs[key] = cudf.from_pandas(value)
            else:
                new_kwargs[key] = value
        return new_kwargs['X'], new_kwargs['y']

    has_cudf_dependancy_, _, cudf = lazy_cudf_import()
    if has_cudf_dependancy_:
        # print('DBSCAN CUML Matrices')
        return safe_cudf(X, y)
    else:
        return X, y


def get_model_matrix(g, kind: str, cols: Optional[Union[List, str]], umap, target):
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
    assert kind in ["nodes", "edges"]
    assert (
        hasattr(g, "_node_encoder") if kind == "nodes" else hasattr(g, "_edge_encoder")
    )

    df = g.get_matrix(cols, kind=kind, target=target)

    if umap and cols is None and g._umap is not None:
        df = g._get_embedding(kind)            
    
    #if g.engine_dbscan in [CUML]:
    df, _ = make_safe_gpu_dataframes(df, None, g.engine_dbscan)
    #print('\n df:', df.shape, df.columns)
    return df


def dbscan_fit(g: Any, dbscan: Any, kind: str = "nodes", cols: Optional[Union[List, str]] = None, use_umap_embedding: bool = True, target: bool = False, verbose: bool = False):
    """
    Fits clustering on UMAP embeddings if umap is True, otherwise on the features dataframe
        or target dataframe if target is True.

    Args:
        :g: graphistry graph
        :kind: 'nodes' or 'edges'
        :cols: list of columns to use for clustering given `g.featurize` has been run
        :use_umap_embedding: whether to use UMAP embeddings or features dataframe for clustering (default: True)
    """
    X = get_model_matrix(g, kind, cols, use_umap_embedding, target)
    
    if X.empty:
        raise ValueError("No features found for clustering")

    dbscan.fit(X)
    # this is a future feature one cuml supports it
    if g.engine_dbscan == 'cuml':
        labels = dbscan.labels_.to_numpy()
        # dbscan.components_ = X[dbscan.core_sample_indices_.to_pandas()]  # can't believe len(samples) != unique(labels) ... #cumlfail
    else:
        labels = dbscan.labels_

    if kind == "nodes":
        g._nodes = g._nodes.assign(_dbscan=labels)
    elif kind == "edges":
        g._edges = g._edges.assign(_dbscan=labels)
    else:
        raise ValueError("kind must be one of `nodes` or `edges`")

    kind = "node" if kind == "nodes" else "edge"
    setattr(g, f"_{kind}_dbscan", dbscan)
    
    if cols is not None:  # set False since we used the features for verbose
        use_umap_embedding = False

    if verbose:
        cnt = Counter(labels)
        message = f"DBSCAN found {len(cnt)} clusters with {cnt[-1]} outliers"
        print()
        print('-' * len(message))
        print(message)
        print(f"--fit on {'umap embeddings' if use_umap_embedding else 'feature embeddings'} of size {X.shape}")
        print('-' * len(message))

    return g


def dbscan_predict(X: pd.DataFrame, model: Any):
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


class ClusterMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass

    def _cluster_dbscan(
        self, res, kind, cols, fit_umap_embedding, target, min_dist, min_samples, engine_dbscan, verbose, *args, **kwargs
    ):
        """DBSCAN clustering on cpu or gpu infered by .engine flag
        """
        _, DBSCAN, _, cuDBSCAN = lazy_dbscan_import()

        if engine_dbscan in [CUML]:
            print('`g.transform_dbscan(..)` not supported for engine=cuml, will return `g.transform_umap(..)` instead')

        res.engine_dbscan = engine_dbscan  # resolve_cpu_gpu_engine(engine_dbscan)  # resolve_cpu_gpu_engine("auto")
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

        dbscan = (
            cuDBSCAN(eps=min_dist, min_samples=min_samples, *args, **kwargs)
            if res.engine_dbscan == CUML
            else DBSCAN(eps=min_dist, min_samples=min_samples, *args, **kwargs)
        )
        # print('dbscan:', dbscan)

        res = dbscan_fit(
            res, dbscan, kind=kind, cols=cols, use_umap_embedding=fit_umap_embedding, verbose=verbose
            )

        return res

    def dbscan(
        self,
        min_dist: float = 0.2,
        min_samples: int = 1,
        cols: Optional[Union[List, str]] = None,
        kind: str = "nodes",
        fit_umap_embedding: bool = True,
        target: bool = False,
        verbose: bool = False,
        engine_dbscan: str = 'sklearn',
        *args,
        **kwargs,
    ):
        """DBSCAN clustering on cpu or gpu infered automatically. Adds a `_dbscan` column to nodes or edges.
           NOTE: g.transform_dbscan(..) currently unsupported on GPU.

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

        res = self.bind()
        res = res._cluster_dbscan(
            res,
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
        )

        return res

    def _transform_dbscan(
        self, df: pd.DataFrame, ydf, kind, verbose
    ) -> Tuple[Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        res = self.bind()
        if hasattr(res, "_dbscan_params"):
            # Assume that we are transforming to last fit of dbscan
            cols = res._dbscan_params["cols"]
            umap = res._dbscan_params["fit_umap_embedding"]
            target = res._dbscan_params["target"]

            dbscan = res._node_dbscan if kind == "nodes" else res._edge_dbscan
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
            
            if res.engine_dbscan == 'cuml':
                print('Transform DBSCAN not yet supported for engine_dbscan=`cuml`, use engine=`umap_learn`, `pandas` or `sklearn` instead')
                return emb, X, y, df
            
            X_, emb = make_safe_gpu_dataframes(X_, emb, 'pandas')  

            labels = dbscan_predict(X_, dbscan)  # type: ignore
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
            df, y = make_safe_gpu_dataframes(df, y, 'pandas')
            X, emb = make_safe_gpu_dataframes(X, emb, 'pandas')
            g = self._infer_edges(emb, X, y, df, eps=min_dist, sample=sample, n_neighbors=n_neighbors,  # type: ignore
                infer_on_umap_embedding=infer_umap_embedding
                )
            return g
        return emb, X, y, df
