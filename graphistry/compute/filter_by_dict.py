from typing import Any, Dict, Optional, TYPE_CHECKING, Union
import pandas as pd
from graphistry.Engine import EngineAbstract, df_to_engine, resolve_engine, s_cons
from graphistry.util import setup_logger

from graphistry.Plottable import Plottable
from .predicates.ASTPredicate import ASTPredicate
from .typing import DataFrameT


logger = setup_logger(__name__)


def filter_by_dict(df: DataFrameT, filter_dict: Optional[dict] = None, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> DataFrameT:
    """
    return df where rows match all values in filter_dict
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    if filter_dict is None or filter_dict == {}:
        return df
    
    engine_concrete = resolve_engine(engine, df)
    df = df_to_engine(df, engine_concrete)
    logger.debug('filter_by_dict engine: %s => %s', engine, engine_concrete)

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
        S = s_cons(engine_concrete)
        hits = (df[list(filter_dict_concrete)] == S(filter_dict_concrete)).all(axis=1)
    else:
        hits = df[[]].assign(x=True).x
    if predicates:
        for col, op in predicates.items():
            hits = hits & op(df[col])
    return df[hits]


def filter_nodes_by_dict(self: Plottable, filter_dict: dict, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
    """
    filter nodes to those that match all values in filter_dict
    """
    nodes2 = filter_by_dict(self._nodes, filter_dict, engine)
    return self.nodes(nodes2)


def filter_edges_by_dict(self: Plottable, filter_dict: dict, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
    """
    filter edges to those that match all values in filter_dict
    """
    edges2 = filter_by_dict(self._edges, filter_dict, engine)
    return self.edges(edges2)
