from typing import Optional, TYPE_CHECKING
import pandas as pd

from graphistry.Plottable import Plottable


def filter_by_dict(df, filter_dict: Optional[dict] = None) -> pd.DataFrame:
    """
    return df where rows match all values in filter_dict
    """

    if filter_dict is None or filter_dict == {}:
        return df

    for col in filter_dict.keys():
        if col not in df.columns:
            raise ValueError(f'Key "{col}" not in columns of df, available columns are: {df.columns}')

    hits = (df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)
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
