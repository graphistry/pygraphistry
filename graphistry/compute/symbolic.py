#import sympy
from typing import TYPE_CHECKING
from collections import Counter
import numpy as np
#from graphistry.ai_utils import build_annoy_index, query_by_vector, DISTANCE
from graphistry.text_utils import SearchToGraphMixin
import pandas as pd

try:
    import symai as ai
    from symai import *
except ImportError:
    ai = None
    
__doc__  =  """graphistry.compute symbolic
                providing symbolic graph operations
                (e.g., computing the characteristic polynomial of a graph's adjacency matrix)
                or symbolic graph operations using the graphistry.ai chatbot"""



if TYPE_CHECKING:
    MIXIN_BASE = SearchToGraphMixin
else:
    MIXIN_BASE = object

def compute_characteristic_polynomial(g, as_expression=True, bound=42):
    """Compute the characteristic polynomial of a graph's adjacency matrix"""
    import sympy
    if hasattr(g, '_adjacency'):
        A = g._adjacency
    elif hasattr(g, '_weighted_adjacency'):
        A = g._weighted_adjacency
        
    if A.shape[0] > bound:
        Warning(f"Computing the characteristic polynomial of a graph with {A.shape[0]} nodes may take a long time")
        
    M = sympy.SparseMatrix(A.todense())
    x = sympy.Symbol("x")
    if as_expression:
        return M.charpoly(x).as_expr()
    return x, M.charpoly(x)


def process_df_to_syms(ndf, as_records):
    if as_records:
        syms = [ai.Symbol(row) for row in ndf.to_dict('records')]
    else:
        syms = [ai.Symbol(str(row)) for row in ndf.values]
    return syms

def process_df_to_sym(df, as_records):
    if as_records:
        syms = ai.Symbol(df.to_dict('records'))
    else:
        syms = ai.Symbol(list(df.values))
    return syms

class Lambda(ai.Expression):
    def __init__(self, callable: ai.Callable):
        super().__init__()
        def _callable(*args, **kwargs):
            kw = {
                'args': args,
                'kwargs': kwargs,
            }
            return callable(kw)
        self.callable: ai.Callable = _callable
    
    def forward(self, *args, **kwargs) -> Symbol:
        return self.callable(*args, **kwargs)

CONTEXTS = [None, 'summary', 'search', 'entities', 'relationships', 'money', 'time', 'threats']

class SymbolicMixin(MIXIN_BASE):
    
    def __init__(self, *args, **kwargs):
        pass
    
    def _encode_df_as_sym(self, context_df, as_records):
        sym = process_df_to_sym(context_df, as_records)
        stream = ai.Stream(Lambda(lambda x: x['args'][0]))
        rr = ai.Symbol(list(stream(sym)))
        return rr
    
    def _add_context_and_query(self, sym, query, context='summary'):
        if context == 'summary':
            res = sym.query(f'Summarize: {query}')
        elif context == 'entities':
            res = sym.query(f'extract relevant entities with a short discription of each in bullet format')
        elif context == 'relationships':
            res = sym.query(f'express relationships found between entities extracted in this document in bullet form')
        elif context == 'money':
            res = sym.query(f'extract money related entities and relationships')    
        elif context == 'time':
            res = sym.query(f'extract time related entities and relationships')
        elif context == 'threats':
            res = sym.query(f'extract threats and alerts related entities and relationships')
        elif context == 'search':
            res = ai.Expression(sym).query(query)
        elif context is None:
            res = sym.query(query)
        else:
            res = sym.query(context)
        return res
        
    def forward(self, query, context='summary', cols=None, top_n = 4, as_records=True, fuzzy=True):
        context_df = self.search(query, top_n=top_n, fuzzy=fuzzy, cols=cols)[0]
        if cols is not None:
            # encode only relevant columns
            context_df = context_df[cols]
        sym = self._encode_df_as_sym(context_df, as_records)
        return self._add_context_and_query(sym, query, context)

    def _analyze(self, g, query=None, context='summary', cols=None, cluster_col='_dbscan', sample=4, max_clusters=10, as_records=True, verbose=False):
        reports = []
        ndf = g._nodes
        label_cnts = Counter(ndf[cluster_col]).most_common()
        labels = [label for label, cnt in label_cnts]
        
        if max_clusters is None:
            max_clusters = len(label_cnts)
        
        if verbose:
            print(f'Found {len(labels)} clusters from `{cluster_col}`')
            print(f' will analyze the top {max_clusters} clusters')
            
        for label in labels[:max_clusters]:
            context_df = ndf[ndf[cluster_col] == label]
            if context_df.empty:
                continue
            if sample is not None and len(context_df) >= sample:
                context_df = context_df.sample(sample)
            if cols is not None:
                context_df = context_df[cols]
            sym = self._encode_df_as_sym(context_df, as_records)
            report = self._add_context_and_query(sym, query, context)
            reports.append(report)
            if verbose:
                print('-'*80)
                print(f'cluster {label+1}:')
                print(report)
                print()
        return reports
    
    def query_graph(self, query, context='summary', cols=None, cluster_col='_dbscan', top_n=4, sample=4, as_records=True, verbose=True):
        gg = self.search_graph(query, top_n=top_n).dbscan()
        # these are the reports for each cluster
        reports = self._analyze(gg, query, context, cols, cluster_cols=cluster_col, as_records=as_records, verbose=verbose, sample=sample)

        ndf = gg._nodes
        edf = gg._edges
        src = gg._source
        dst = gg._destination
        node = gg._node
        print(node, src, dst)
        df = [] 
        df2 = [] 
        labels = np.unique(ndf[cluster_col])

        for i, label in enumerate(labels[:len(reports)]):
            df.append([label, context, query, reports[i].value])
            # find edges that connect to this cluster
            for j, label2 in enumerate(labels[:len(reports)]):
                if label2 == label:
                    continue
                srcdst = edf[src].isin(ndf[ndf._dbscan == label][node]) & edf[dst].isin(ndf[ndf._dbscan == label2][node])
                dstsrc = edf[src].isin(ndf[ndf._dbscan == label2][node]) & edf[dst].isin(ndf[ndf._dbscan == label][node])
                edges = edf[srcdst | dstsrc]
                if not edges.empty:
                    print([label, label2, len(edges)])
                    df2.append([label, label2, len(edges)])

        df = pd.DataFrame(df, columns = [cluster_col, 'context', 'query', 'report'])
        df2 = pd.DataFrame(df2, columns = ['src', 'dst', 'weight'])
        g_cluster = gg.nodes(df, cluster_col).edges(df2, 'src', 'dst').bind(edge_weight='weight')
        # should aggregate vectors etc ... 
        return g_cluster
    
    def on_select(self, query, context, nodeIDs):
        context_df = self._nodes.iloc[nodeIDs]
        sym = self._encode_df_as_sym(context_df, as_records=True)
        return self._add_context_and_query(sym, query, context)
        
    # def on_select_by_name(self, query, context, names):
    #     # total hack
    #     #cdf = self._nodes[self._nodes.full_name.isin(names)] #['full_name', 'title']
    #     #cdf = pd.concat([cdf, context_df], axis=0)
    #     context_df = pd.read_csv('~/dev/pygraphistry/data/context_df.csv', index_col=0)
    #     cdf = context_df.sample(20)
    #     cdf = pd.concat([cdf, pd.DataFrame({'full_name': names})], axis=0)

    #     sym = self._encode_df_as_sym(cdf, as_records=True)
    #     return self._add_context_and_query(sym, query, context)

