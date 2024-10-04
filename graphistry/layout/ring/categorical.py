from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from graphistry.Engine import Engine, EngineAbstract, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.layout.ring.util import polar_to_xy


MIN_R_DEFAULT = 100
MAX_R_DEFAULT = 1000



def gen_axis(
    order: List[str],  # if combine_unhandled, missing those, else, with
    val_to_r: Dict[Any, float],
    unhandled: Set[Any],
    combine_unhandled: bool,
    append_unhandled: bool,
    axis: Optional[Dict[Any, str]],
    label: Optional[Callable[[Any, int, float], str]],
    reverse: bool
) -> List[Dict]:
    
    # also includes items from unhandled when not combine_unhandled
    axis_out = [
        {
            "label":
                axis[val]
                if axis is not None else (
                    str(val)
                    if label is None else label(
                        val,
                        i,
                        val_to_r[val]
                    )
                ),
            "r": val_to_r[val],
            "internal": True
        }
        for i, val in enumerate(order)
    ]

    if combine_unhandled and len(unhandled) > 0:
        # add other ring, position based on reverse
        axis_out += [{
            "label":
                "Other"
                if axis is not None or label is None else
                label(
                    unhandled,
                    len(order) if (append_unhandled and not reverse) or (not append_unhandled and reverse) else 0,
                    val_to_r[next(iter(unhandled))]
                ),
            "r": val_to_r[next(iter(unhandled))],
            "internal": True
        }]
    
    return axis_out

def find_first_numeric_column(df: Any) -> str:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            return col
    raise ValueError('No numeric columns found')

def ring_categorical(
    g: Plottable,
    ring_col: str,
    order: Optional[List[Any]] = None,
    drop_empty: bool = True,
    combine_unhandled: bool = False,
    append_unhandled: bool = True,
    min_r: float = MIN_R_DEFAULT,
    max_r: float = MAX_R_DEFAULT,
    axis: Optional[Dict[Any,str]] = None,
    format_axis: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    format_labels: Optional[Callable[[Any, int, float], str]] = None,
    reverse: bool = False,
    play_ms: int = 0,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:

    """Radial graph layout where nodes are positioned based on a categorical column ring_col

    Uses GPU when cudf nodes are used, otherwise pandas

    min_r, max_r are the first/last axis positions

    :g: Plottable
    :ring_col: Optional[str] Column name of nodes numerica-typed column; defaults to first numeric node column
    :order: Optional[List[Any]] Order of axis specified in category values
    :drop_empty: bool (default True) Whether to drop axis when no values populating them
    :combine_unhandled: bool (default False) Whether to collapse all unexpected values into one ring or one-per-unique-value
    :append_unhandled: bool (default True) Whether to append or prepend the unexpected items axis 
    :min_r: float Minimum radius, default 100
    :max_r: float Maximum radius, default 1000
    :ring_step: Optional[float] Distance between rings in terms of pixels
    :axis: Optional[Dict[Any, str]], Set to provide labels for each ring by mapping from the categorical input domain values. Requires all values to be mapped.
    :format_axis: Optional[Callable[[List[Dict]], List[Dict]]] Optional transform function to format axis
    :format_label: Optional[Callable[[Any, int, float], str]] Optional transform function to format axis label text based on axis value, ring number, and ring position
    :reverse: bool Reverse the direction of the rings
    :play_ms: int initial layout time in milliseconds, default 2000
    :engine: Union[EngineAbstract, str], default EngineAbstract.AUTO, pick CPU vs GPU engine via 'auto', 'pandas', 'cudf' 

    :returns: Plotter
    :rtype: Plotter


    **Example: Minimal categorical ring layout**

    ::

            assert 'a_cat_node_column' in g._nodes
            g.ring_categorical_layout('a_cat_node_column').plot()

    **Example: Categorical ring layout with a few rings, and rest as Other**

    ::

            g2 = g.ring_categorical_layout('a_cat_node_column', order=['a', 'b', 'c'], combine_unhandled=True)
            g2.plot()
    
    **Example: Categorical ring layout with relabeled axis rings**

    ::

            g2 = g.ring_categorical_layout(
                'a_cat_node_column',
                axis={
                    'a': 'ring a',
                    'b': 'ring b',
                    'c': 'ring c'
                }
            )
            g2.plot()

    **Example: Categorical ring layout without labels**

    ::

            EMPTY_AXIS_LIST = []
            g2 = g.ring_categorical_layout('a_cat_node_column', format_labels=lambda axis: EMPTY_AXIS_LIST)

    **Example: Categorical ring layout with specific first and last ring positions**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_categorical_layout(
                'a_cat_node_column',
                min_r=400,
                max_r=1000,
            )
            g2.plot()

    **Example: Categorical ring layout in reverse order**

    ::

            g2 = g.ring_categorical_layout('a_cat_node_column', order=['a', 'b', 'c'], reverse=True)
            g2.plot()

    """

    if g._nodes is None:
        raise ValueError('Missing nodes')

    g = g.nodes(g._nodes.reset_index(drop=True))

    engine_concrete = resolve_engine(engine, g._nodes)

    if ring_col is None or not isinstance(ring_col, str):
        raise ValueError('Must name a column for ring_col (string)')

    if ring_col not in g._nodes.columns:
        raise ValueError('Missing ring column')

    @lru_cache()
    def unique() -> List[Any]:
        s = g._nodes[ring_col].unique()
        if engine_concrete == Engine.PANDAS:
            return list(s)
        elif engine_concrete == Engine.CUDF:
            return list(s.to_pandas())
        else:
            raise ValueError(f'Unexpected engine, expected "all", "pandas", or "cudf", received: {engine} => {engine_concrete}')

    if order is None:
        order = unique()
    else:
        assert len(order) == len(set(order)), "order must not contain duplicate values"

    if drop_empty:
        order_hits = set(order) & set(unique())
        order = [x for x in order if x in order_hits]
    
    unhandled = set(unique()) - set(order)
    if len(unhandled) > 0:
        if not combine_unhandled:
            if append_unhandled:
                order += list(unhandled)
            else:
                order = list(unhandled) + order

    val_to_ring: Dict[Any, int] = {
        v: i if not reverse else len(order) - i - 1
        for i, v in enumerate(order)
    }

    if len(unhandled) > 0 and combine_unhandled:
        if (append_unhandled and not reverse) or (not append_unhandled and reverse):
            unhandled_ring = len(order)
        else:
            unhandled_ring = 0
            val_to_ring = {
                k: ring + 1
                for k, ring in val_to_ring.items()
            }
        for v in unhandled:
            val_to_ring[v] = unhandled_ring
    
    num_rings = len(set(val_to_ring.values())) - 1
    scalar = (max_r - min_r) / num_rings
    val_to_r = {
        val: ring * scalar + min_r
        for val, ring in val_to_ring.items()
    }

    r = g._nodes[ring_col].map(val_to_r)

    angle = r.reset_index(drop=True).index.to_series()
    x, y = polar_to_xy(g, r, angle, engine_concrete)

    axis_out = gen_axis(
        order,
        val_to_r,
        unhandled,
        combine_unhandled,
        append_unhandled,
        axis,
        format_labels,
        reverse
    )

    if format_axis is not None:
        axis_out = format_axis(axis_out)

    #print('axis', axis)

    g2 = (
        g
          .nodes(lambda g: g._nodes.assign(x=x, y=y, r=r))
          .encode_axis(axis_out)
          .bind(point_x='x', point_y='y')
          .settings(url_params={
              'play': play_ms,
              'lockedR': True,
              'bg': '%23E2E2E2'  # Light grey due to labels being fixed to dark
          })
    )

    return g2
