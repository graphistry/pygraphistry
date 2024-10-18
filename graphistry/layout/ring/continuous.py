from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from graphistry.Engine import EngineAbstract, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.layout.ring.util import polar_to_xy


MIN_R_DEFAULT = 100
MAX_R_DEFAULT = 1000


def gen_axis(
    axis_input: Optional[Union[Dict[float,str],List[str]]],
    num_rings: int,
    v_start: float,
    v_step: float,
    r_start: float,
    step_r: float,
    label: Optional[Callable[[float, int, float], str]] = None,
    reverse: bool = False
) -> List[Dict]:

    if axis_input is not None:
        assert len(axis_input) == num_rings + 1

    if axis_input is None or isinstance(axis_input, list):
        axis = [
            {
                "label":
                    axis_input[step]
                    if isinstance(axis_input, list) else (
                        str(
                            v_start + v_step * step
                        ) if label is None else label(
                            v_start + v_step * step,
                            step,
                            v_step
                        )
                    ),
                "r": r_start + ((num_rings - step) if reverse else step) * step_r,
                "internal": True
            }
            for step in range(0, num_rings + 1)
        ]
        return axis
    elif isinstance(axis_input, dict):
        axis = [
            {
                "label": v if label is None else label(k, 0, 0.0),
                "r": k,
                "internal": True
            }
            for k, v in axis_input.items()
        ]
        return axis

def find_first_numeric_column(df: Any) -> str:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            return col
    raise ValueError('No numeric columns found')

def ring_continuous(
    g: Plottable,
    ring_col: Optional[str] = None,
    v_start: Optional[float] = None,
    v_end: Optional[float] = None,
    v_step: Optional[float] = None,
    min_r: Optional[float] = MIN_R_DEFAULT,
    max_r: Optional[float] = MAX_R_DEFAULT,
    normalize_ring_col: bool = True,
    num_rings: Optional[int] = None,
    ring_step: Optional[float] = None,
    axis: Optional[Union[Dict[float,str],List[str]]] = None,
    format_axis: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    format_labels: Optional[Callable[[float, int, float], str]] = None,
    reverse: bool = False,
    play_ms: int = 0,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:

    """Radial graph layout where nodes are positioned based on a numeric-typed column ring_col

    Uses GPU when cudf nodes are used, otherwise pandas

    min_r, max_r are the first/last axis positions

    optional v_start, v_end are used to line up the input value domain to the axis:
      - v_start: corresponds to the first axis at min_r, defaulting to g._nodes[ring_col].min() 
      - v_end: corresponds to the last axis at max_r, defaulting to g._nodes[ring_col].max() 

    :g: Plottable
    :ring_col: Optional[str] Column name of nodes numerica-typed column; defaults to first numeric node column
    :v_start: Optional[float] Value at innermost axis (at min_r), defaults to g._nodes[ring_col].min()
    :v_end: Optional[float] Value at outermost axis (at max_r), defaults to g._nodes[ring_col].max()
    :v_step: Optional[float] Distance between rings in terms of ring column value domain
    :min_r: float Minimum radius, default 100
    :max_r: float Maximum radius, default 1000
    :normalize_ring_col: bool, default True, Whether to recale to min/max r, or pass through existing values
    :num_rings: Optional[int] Number of rings
    :ring_step: Optional[float] Distance between rings in terms of pixels
    :axis: Optional[Union[Dict[float,str],List[str]]], Set to provide labels for each ring, and in dict mode, also specify radius for each
    :format_axis: Optional[Callable[[List[Dict]], List[Dict]]] Optional transform function to format axis
    :format_label: Optional[Callable[[float, int, float], str]] Optional transform function to format axis label text based on axis value, ring number, and ring width
    :reverse: bool Reverse the direction of the rings
    :play_ms: int initial layout time in milliseconds, default 2000
    :engine: Union[EngineAbstract, str], default EngineAbstract.AUTO, pick CPU vs GPU engine via 'auto', 'pandas', 'cudf' 

    :returns: Plotter
    :rtype: Plotter


    **Example: Minimal continuous ring layout**

    ::

            g.ring_continuous_layout().plot()

    **Example: Continuous ring layout**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_continuous_layout('my_numeric_col')
            g2.plot()        

    **Example: Continuous ring layout with 7 rings**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_continuous_layout('my_numeric_col', num_rings=7)
            g2.plot()
    
    **Example: Continuous ring layout using small steps**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_continuous_layout('my_numeric_col', ring_step=20.0)
            g2.plot()

    **Example: Continuous ring layout without labels**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            EMPTY_AXIS_LIST = []
            g2 = g.ring_continuous_layout('my_numeric_col', format_labels=lambda axis: EMPTY_AXIS_LIST)

    **Example: Continuous ring layout with specific first and last ring positions**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_continuous_layout(
                'my_numeric_col',
                min_r=200,
                max_r=2000,
                v_start=32,  # corresponding column value at first axis radius at pixel radius 200
                v_end=83,  # corresponding column value at last axius radius at pixel radius 2000
            )
            g2.plot()

    **Example: Continuous ring layout in reverse order**

    ::

            assert 'float' in g._nodes.my_numeric_col.dtype.name
            g2 = g.ring_continuous_layout('my_numeric_col', reverse=True)
            g2.plot()

    """

    if num_rings is None and axis is not None:
        num_rings = len(axis) - 1

    if g._nodes is None:
        raise ValueError('Missing nodes')

    g = g.nodes(g._nodes.reset_index(drop=True))

    engine_concrete = resolve_engine(engine, g._nodes)

    if ring_col is None:
        ring_col = find_first_numeric_column(g._nodes)

    if ring_col not in g._nodes.columns:
        raise ValueError('Missing ring column')
    
    if v_start is None:
        v_start = g._nodes[ring_col].min()
    if v_end is None:
        v_end = g._nodes[ring_col].max()

    if normalize_ring_col:
        assert min_r is not None, 'Cannot set min_r to None when normalizing ring_col'
        assert max_r is not None, 'Cannot set max_r to None when normalizing ring_col'
        scalar = (max_r - min_r) / (v_end - v_start)
        r = (g._nodes[ring_col] - v_start) * scalar + min_r
    else:
        r = g._nodes[ring_col]
        if min_r is None:
            min_r = r.min()
        if max_r is None:
            max_r = r.max()

    if reverse:
        r = -r + (max_r + min_r)

    #num_rings, v_step, ring_step
    if num_rings is not None:
        if ring_step is None:
            ring_step = (max_r - min_r) / num_rings
        if v_step is None:
            v_step = (v_end - v_start) / num_rings
    else:
        if ring_step is None and v_step is None:
            num_rings = 5
            ring_step = (max_r - min_r) / num_rings
            v_step = (v_end - v_start) / num_rings
        elif v_step is not None:
            num_rings = int((v_end - v_start) / v_step)
            ring_step = (max_r - min_r) / num_rings        
        else:
            assert ring_step is not None
            num_rings = int((max_r - min_r) / ring_step)
            v_step = (v_end - v_start) / num_rings

    angle = r.reset_index(drop=True).index.to_series()
    x, y = polar_to_xy(g, r, angle, engine_concrete)

    axis_out = gen_axis(
        axis,
        num_rings,
        v_start,
        v_step,
        min_r,
        ring_step,
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
