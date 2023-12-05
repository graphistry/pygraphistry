from typing import Dict, Optional
import pandas as pd

from graphistry.Plottable import Plottable
from .predicates.ASTPredicate import ASTPredicate


def filter_by_dict(df, filter_dict: Optional[dict] = None) -> pd.DataFrame:
    """
    return df where rows match all values in filter_dict
    """

    if filter_dict is None or filter_dict == {}:
        return df

    predicates: Dict[str, ASTPredicate] = {}
    for col, val in filter_dict.items():
        if col not in df.columns:
            raise ValueError(f'Key "{col}" not in columns of df, available columns are: {df.columns}')
        if isinstance(val, ASTPredicate):
            predicates[col] = val
    filter_dict_concrete = filter_dict if not predicates else {
        k: v
        for k, v in filter_dict.items()
        if not isinstance(v, ASTPredicate)
    }

    if filter_dict_concrete:
        hits = (df[list(filter_dict_concrete)] == pd.Series(filter_dict_concrete)).all(axis=1)
    else:
        hits = df[[]].assign(x=True).x
    if predicates:
        for col, op in predicates.items():
            hits = hits & op(df[col])
    return df[hits]


def filter_nodes_by_dict(self: Plottable, filter_dict: dict) -> Plottable:
    """
    filter nodes to those that match all values in filter_dict
    """
    nodes2 = filter_by_dict(self._nodes, filter_dict)
    return self.nodes(nodes2)


def filter_edges_by_dict(self: Plottable, filter_dict: dict) -> Plottable:
    """
    filter edges to those that match all values in filter_dict
    """
    edges2 = filter_by_dict(self._edges, filter_dict)
    return self.edges(edges2)
