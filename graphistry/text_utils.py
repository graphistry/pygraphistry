import os
from time import time
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from joblib import load, dump # need to make this onnx or similar

from .feature_utils import make_array
from .ai_utils import search_to_df, setup_logger

from .constants import SRC, DST, WEIGHT, N_TREES, DISTANCE, VERBOSE, TRACE

logger = setup_logger(__name__, verbose=VERBOSE, fullpath=TRACE)


class SearchToGraphMixin:
    def __init__(self, metric="euclidean", n_trees = N_TREES) -> None:
        super().__init__()
        self._model_kwargs = {} if getattr(self, "_model_kwargs", None) is None else self._model_kwargs
        self._model_kwargs['text_search'] = {'metric': metric,
                                             'n_trees': n_trees}
        self._n_trees = n_trees
        #self.build_index()

    def assert_fitted(self):
        assert self._umap is not None, 'Umap needs to be fit first, run g.umap(..) to fit a model'
        x = self._node_features
        assert (
            x is not None
        ), "Graphistry Instance is not fit, run g.umap(..) to fit a model"


    def build_index(self):
        # builds local index
        self.assert_fitted()

        X = self._node_features
        print(f"Building Index of size {X.shape}")

        search_index = AnnoyIndex(X.shape[1], self.metric)
        # Add all the feature vectors to the search index
        for i in range(len(X)):
            search_index.add_item(i, X.values[i])

        search_index.build(self._n_trees) 
        self.search_index = search_index

    def _query_from_dataframe(self, qdf: pd.DataFrame, top_k: int, thresh: float):
        self.assert_fitted()
        if not hasattr(self, 'search_index'):
            self.build_index()
        # Use the loaded featurizers to transform the dataframe
        vect, _ = self.transform(qdf, None, kind="nodes")
        
        indices, distances = self.search_index.get_nns_by_vector(
            vect.values[0], top_k, include_distances=True
        )
        
        results = self._nodes.iloc[indices]
        results[DISTANCE] = distances
        results = results.query(f"{DISTANCE} < {thresh}")

        results = results.sort_values(by=[DISTANCE])

        return results, vect
    
    def _query(self, query: str, top_k: int, thresh: float):
        # build the query dataframe
        self.assert_fitted()
        if not hasattr(self, 'search_index'):
            self.build_index()

        qdf = pd.DataFrame([])
                
        cols_text = self._node_encoder.text_cols
        if len(cols_text) == 0:
            print(f'**Querying is only possible using Transformer embeddings')    
            return pd.DataFrame([]), None
            
        qdf[cols_text[0]] = [query]
        if len(cols_text) > 1:
            for col in cols_text[1:]:
                qdf[col] = ['']   

        if hasattr(self._node_encoder.data_encoder, 'columns_'):
            other_cols = self._node_encoder.data_encoder.columns_
            if other_cols is not None:
                logger.warn(f'There is no easy way to encode categorical or other features at query time.\
                            Set `thresh` to a large value if no results show up.')
                df = self._nodes
                dt = df[other_cols].dtypes
                for col, v in zip(other_cols, dt.values):
                    if str(v) in ["string", "object", "category"]:
                        qdf[col] = df.sample(1)[col].values # so hookey 
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
                        qdf[col] = 0 #df[col].mean()
        
        return self._query_from_dataframe(qdf, thresh=thresh, top_k=top_k)

    def query(
        self, query: str, cols=None, thresh: float=50, fuzzy: bool = True, top_k: int = 10
    ):  
        if not fuzzy:
            if cols is None:
                logger.error(f'Columns to search for `{query}` \
                             need to be given when fuzzy=False, found {cols}')
                
            print(f"-- Word Match: [[ {query} ]]")
            return (
                pd.concat([search_to_df(query, col, self._nodes) for col in cols]),
                None
            )
        else:
            print(f"-- Search: [[ {query} ]]")
            return self._query(query, thresh=thresh, top_k=top_k)

    def query_to_graph(
        self,
        query: str,
        scale: float = 0.5,
        top_k: int = 100,
        thresh: float = 500,
        broader: bool = False,
        inplace: bool = False,
    ):

        if inplace:
            res = self
        else:
            res = self.bind()
                    
        edf = edges = res._edges
        df = res._nodes
        node = res._node
        src = res._source
        dst = res._destination
        if query != "":
            # run a real query, else return entire graph
            rdf, _ = res.query(query, thresh=thresh, fuzzy=True, top_k=top_k)
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
                print('**No results found due to empty DataFrame, returning original graph')
                return res

        edges = edges.query(f"{WEIGHT} > {scale}")
        found_indices = pd.concat([edges[src], edges[dst]], axis=0).unique()
        tdf = df.iloc[found_indices]
        print(f"  - Returning edge dataframe of size {edges.shape[0]}")
        # get all the unique nodes
        print(f"  - Returning {tdf.shape[0]} unique nodes given scale {scale}")
        g = res.edges(edges, src, dst).nodes(tdf, node)
        return g

    def save(self, savepath):
        search = self.search_index
        del self.search_index  # can't pickle Annoy
        dump(self, savepath)
        self.search_index = search  # add it back
        print(f"Saved: {savepath}")

    @classmethod
    def load(self, savepath):
        cls = load(savepath)
        cls.build_index()
        return cls
    