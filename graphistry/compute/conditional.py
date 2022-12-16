import logging
import pandas as pd
from typing import Any, List, Union, TYPE_CHECKING
from typing_extensions import Literal

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable

logger = logging.getLogger("compute.conditional")

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


# ############################################################################
#
#      Conditional Probability
#
# ############################################################################

def conditional_probability(x, given, df: pd.DataFrame):
    """conditional probability function over categorical variables
       p(x | given) = p(x, given)/p(given)
        
    Args:
        x: the column variable of interest given the column 'given'
        given: the variabe to fix constant
        df: dataframe with columns [given, x]

    Returns:
        pd.DataFrame: the conditional probability of x given the column 'given'
    """
    
    return df.groupby([ given ])[ x ].apply(lambda g : g.value_counts()/len(g))  # noqa type: ignore


def probs(x, given, df: pd.DataFrame, how='index'): 
    """Produces a Dense Matrix of the conditional probability of x given `y=given`

    Args:
        x: the column variable of interest given the column 'y'
        given : the variabe to fix constant
        df pd.DataFrame: dataframe
        how (str, optional): One of 'column' or 'index'. Defaults to 'index'.

    Returns:
        pd.DataFrame: the conditional probability of x given the column 'y' 
        as dense array like dataframe
    """
    assert how in ['index', 'columns'], "how must be one of 'index' or 'columns'"
    res = pd.crosstab(df[x], df[given], margins=True, normalize=how)
    if how == 'index':  # normalize over columns so .sum(0) = 1 irrespective of `how`
        return res.T
    return res

class ConditionalMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass

    def conditional_graph(self, x, given, kind='nodes', *args, **kwargs):
        """
        conditional_graph -- p(x|given) = p(x, given) / p(given)
        
        Useful for finding the conditional probability of a node or edge attribute
        
        returned dataframe sums to 1 on each column
        -----------------------------------------------------------
        :param x: target column
        :param given: the dependent column
        :param kind: 'nodes' or 'edges'
        :param args/kwargs: additional arguments for g.bind(...)
        :return: a graphistry instance with the conditional graph
                edges weighted by the conditional probability.
                edges are between `x` and `given`, keep in mind that 
                g._edges.columns = [given, x, _probs]
                
        """

        res = self.bind()
        
        if kind == 'nodes':
            df = res._nodes
        else:
            df = res._edges
        
        condprobs = conditional_probability(x, given, df)
        
        cprob = pd.DataFrame(list(condprobs.index), columns=[given, x])
        cprob['_probs'] = condprobs.values
    
        res = res.edges(cprob, x, given).bind(edge_weight='_probs', *args, **kwargs)
        
        return res
    
    def conditional_probs(self, x, given, kind = 'nodes', how = 'index'):
        """Produces a Dense Matrix of the conditional probability of x given y

        Args:
            x: the column variable of interest given the column y=given
            given : the variabe to fix constant
            df pd.DataFrame: dataframe
            how (str, optional): One of 'column' or 'index'. Defaults to 'index'.
            kind (str, optional): 'nodes' or 'edges'. Defaults to 'nodes'.
        Returns:
            pd.DataFrame: the conditional probability of x given the column y
            as dense array like dataframe
        """
        res = self.bind()
        
        if kind == 'nodes':
            df = res._nodes    
        else:
            df = res._edges
            
        condprobs = probs(x, given, df, how=how) 
        return condprobs
