import os
from time import time
import numpy as np
import pandas as pd

from .feature_utils import FeatureMixin
from .ai_utils import search_to_df, setup_logger
from .constants import WEIGHT, N_TREES, DISTANCE, VERBOSE, TRACE

from typing import (
    Hashable,
    List,
    Union,
    Dict,
    Any,
    Optional,
    Tuple,
    TYPE_CHECKING, 
    Type
)  # noqa


logger = setup_logger(__name__, verbose=VERBOSE, fullpath=TRACE)

if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object

class SearchToGraphMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def assert_fitted(self):
        # assert self._umap is not None, 'Umap needs to be fit first, run g.umap(..) to fit a model'
        assert (
            self._get_feature('nodes') is not None
        ), "Graphistry Instance is not fit, run g.featurize(kind='nodes', ..) to fit a model' \
        'if you have nodes & edges dataframe or g.umap(kind='nodes', ..) if you only have nodes dataframe"

    def assert_features_line_up_with_nodes(self):
        ndf = self._nodes
        X = self._get_feature('nodes')
        a, b = ndf.shape[0], X.shape[0]
        assert a == b, 'Nodes dataframe and feature vectors are not same size, '\
        f'found nodes: {a}, feats: {b}. Did you mutate nodes between fit?'

    def build_index(self, angular=False, n_trees=None):
        from annoy import AnnoyIndex  # type: ignore
        # builds local index
        self.assert_fitted()
        self.assert_features_line_up_with_nodes()
        
        X = self._get_feature('nodes')

        logger.info(f"Building Index of size {X.shape}")

        if angular:
            logger.info('-using angular metric')
            metric = 'angular'
        else:
            logger.info('-using euclidean metric')
            metric = 'euclidean'
            
        search_index = AnnoyIndex(X.shape[1], metric)
        # Add all the feature vectors to the search index
        for i in range(len(X)):
            search_index.add_item(i, X.values[i])
        if n_trees is None:
            n_trees = N_TREES

        logger.info(f'-building index with {n_trees} trees')
        search_index.build(n_trees)

        self.search_index = search_index

    def _query_from_dataframe(self, qdf: pd.DataFrame, top_n: int, thresh: float):
        # Use the loaded featurizers to transform the dataframe
        vect, _ = self.transform(qdf, None, kind="nodes")
        
        indices, distances = self.search_index.get_nns_by_vector(
            vect.values[0], top_n, include_distances=True
        )
        
        results = self._nodes.iloc[indices]
        results[DISTANCE] = distances
        results = results.query(f"{DISTANCE} < {thresh}")

        results = results.sort_values(by=[DISTANCE])

        return results, vect
    
    def _query(self, query: str, top_n: int, thresh: float):
        # build the query dataframe
        if not hasattr(self, 'search_index'):
            self.build_index()

        qdf = pd.DataFrame([])
                
        cols_text = self._node_encoder.text_cols  # type: ignore

        if len(cols_text) == 0:
            logger.warn('** Querying is only possible using Transformer/Ngrams embeddings')    
            return pd.DataFrame([]), None
            
        qdf[cols_text[0]] = [query]
        if len(cols_text) > 1:
            for col in cols_text[1:]:
                qdf[col] = ['']   

        # this is hookey and needs to be fixed on dirty_cat side (with errors='ignore')
        # if however min_words = 0, all columns will be textual, 
        # and no other data_encoder will be generated
        if hasattr(self._node_encoder.data_encoder, 'columns_'):  # type: ignore

            other_cols = self._node_encoder.data_encoder.columns_  # type: ignore

            if other_cols is not None and len(other_cols):
                logger.warn('** There is no easy way to encode categorical or other features at query time. '
                            f'Set `thresh` to a large value if no results show up.\ncolumns: {other_cols}')
                df = self._nodes
                dt = df[other_cols].dtypes
                for col, v in zip(other_cols, dt.values):
                    if str(v) in ["string", "object", "category"]:
                        qdf[col] = df.sample(1)[col].values  # so hookey 
                    elif str(v) in [
                    "int",
                    "float", 
                    "float64",
                    "float32",
                    "float16",
                    "int64",
                    "int32",
                    "int16",
                    "uint64",
                    "uint32",
                    "uint16",
                ]:
                        qdf[col] = df[col].mean()

        return self._query_from_dataframe(qdf, thresh=thresh, top_n=top_n)

    def search(
        self, query: str, cols = None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10
    ):  
        """Natural language query over nodes that returns a dataframe of results sorted by relevance column "distance".

            If node data is not yet feature-encoded (and explicit edges are given), 
            run automatic feature engineering: 
            ```
                g2 = g.featurize(kind='nodes', X=['text_col_1', ..], 
                min_words=0 # forces all named columns are textually encoded
                )  
            ```    
            
            If edges do not yet exist, generate them via 
            ```
                g2 = g.umap(kind='nodes', X=['text_col_1', ..], 
                min_words=0 # forces all named columns are textually encoded
                )  
            ``` 
            If an index is not yet built, it is generated `g2.build_index()` on the fly at search time.
            Otherwise, can set `g2.build_index()` and then subsequent `g2.search(...)` 
            calls will be not rebuilt index.

        Args:
            query (str): natural language query.
            cols (list or str, optional): if fuzzy=False, select which column to query. 
                                            Defaults to None since fuzzy=True by defaul.
            thresh (float, optional): distance threshold from query vector to returned results.
                                        Defaults to 5000, set large just in case, 
                                        but could be as low as 10.
            fuzzy (bool, optional): if True, uses embedding + annoy index for recall, 
                                        otherwise does string matching over given `cols` 
                                        Defaults to True.
            top_n (int, optional): how many results to return. Defaults to 100.

        Returns:
            pd.DataFrame, vector_encoding_of_query: 
                * rank ordered dataframe of results matching query
                * vector encoding of query via given transformer/ngrams model if fuzzy=True
                    else None
        """
        if not fuzzy:
            if cols is None:
                logger.error(f'Columns to search for `{query}` \
                             need to be given when fuzzy=False, found {cols}')
                
            logger.info(f"-- Word Match: [[ {query} ]]")
            return (
                pd.concat([search_to_df(query, col, self._nodes, as_string=True) for col in cols]),
                None
            )
        else:
            logger.info(f"-- Search: [[ {query} ]]")
            return self._query(query, thresh=thresh, top_n=top_n)

    def search_graph(
        self,
        query: str,
        scale: float = 0.5,
        top_n: int = 100,
        thresh: float = 5000,
        broader: bool = False,
        inplace: bool = False,
    ):
        """Input a natural language query and return a graph of results. 
            See help(g.search) for more information

        Args:
            query (str): query input eg "coding best practices"
            scale (float, optional): edge weigh threshold,  Defaults to 0.5.
            top_n (int, optional): how many results to return. Defaults to 100.
            thresh (float, optional): distance threshold from query vector to returned results.
                                        Defaults to 5000, set large just in case, 
                                        but could be as low as 10.
            broader (bool, optional): if True, will retrieve entities connected via an edge
                that were not necessarily bubbled up in the results_dataframe. Defaults to False.
            inplace (bool, optional): whether to return new instance (default) or mutate self.
                                        Defaults to False.

        Returns:
            graphistry Instance: g
        """
        if inplace:
            res = self
        else:
            res = self.bind()
                    
        edf = edges = res._edges
        rdf = df = res._nodes
        node = res._node
        indices = rdf[node]
        src = res._source
        dst = res._destination
        if query != "":
            # run a real query, else return entire graph
            rdf, _ = res.search(query, thresh=thresh, fuzzy=True, top_n=top_n)
            if not rdf.empty:
                indices = rdf[node]
                # now get edges from indices
                if broader:  # this will make a broader graph, finding NN in src OR dst
                    edges = edf[
                        (edf[src].isin(indices)) | (edf[dst].isin(indices))
                    ]
                else:  # finds only edges between results from query, if they exist, 
                    # default smaller graph
                    edges = edf[
                        (edf[src].isin(indices)) & (edf[dst].isin(indices))
                    ]
            else:
                logger.warn('**No results found due to empty DataFrame, returning original graph')
                return res
            
        try:  # for umap'd edges
            edges = edges.query(f"{WEIGHT} > {scale}")
        except:  # for explicit edges
            pass
        
        found_indices = pd.concat([edges[src], edges[dst], indices], axis=0).unique()
        try:
            tdf = rdf.iloc[found_indices]
        except:  # for explicit relabeled nodes
            tdf = rdf[df[node].isin(found_indices)]
        logger.info(f" - Returning edge dataframe of size {edges.shape[0]}")
        # get all the unique nodes
        logger.info(f" - Returning {tdf.shape[0]} unique nodes given scale {scale} and thresh {thresh}")
        
        g = res.edges(edges, src, dst).nodes(tdf, node)
        
        if g._name is not None:
            name = f'{g._name}-query:{query}'
        else:
            name = f'query:{query}'
        g = g.name(name)  # type: ignore
        return g

    def save_search_instance(self, savepath):
        from joblib import dump  # type: ignore   # need to make this onnx or similar
        self.build_index()
        search = self.search_index
        del self.search_index  # can't pickle Annoy
        dump(self, savepath)
        self.search_index = search  # add it back
        logger.info(f"Saved: {savepath}")

    @classmethod
    def load_search_instance(self, savepath):
        from joblib import load  # type: ignore   # need to make this onnx or similar
        cls = load(savepath)
        cls.build_index()
        return cls
