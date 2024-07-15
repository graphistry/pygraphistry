from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from graphistry.Plottable import Plottable


MIN_R_DEFAULT = 100
MAX_R_DEFAULT = 1000


def gen_axis(
    num_rings: int,
    domain_start: float,
    domain_step: float,
    r_start: float,
    step_r: float,
    label: Optional[Callable[[float, int, float], str]] = None,
    reverse: bool = False
) -> List[Dict]:
    axis = [
      {
          "label": str(
              domain_start + domain_step * step
          ) if label is None else label(
              domain_start + domain_step * step,
              step,
              domain_step
          ),
          "r": r_start + ((num_rings - step) if reverse else step) * step_r,
          "internal": True
      }
      for step in range(0, num_rings + 1)
    ]
    return axis

def ring_continuous(
    g: Plottable,
    ring_col: str,
    min_r: Optional[float] = MIN_R_DEFAULT,
    max_r: Optional[float] = MAX_R_DEFAULT,
    normalize_ring_col: bool = True,
    num_rings: Optional[int] = None,
    ring_step: Optional[float] = None,
    format_axis: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    format_labels: Optional[Callable[[float, int, float], str]] = None,
    reverse: bool = False,
    play_ms: int = 2000,
) -> Plottable:

    if g._nodes is None:
        raise ValueError('Missing nodes')
    if ring_col not in g._nodes.columns:
        raise ValueError('Missing ring column')

    if normalize_ring_col:
        assert min_r is not None, 'Cannot set min_r to None when normalizing ring_col'
        assert max_r is not None, 'Cannot set max_r to None when normalizing ring_col'
        r = g._nodes[ring_col] - g._nodes[ring_col].min()
        r = r / (r.max() - r.min())
        r = r * (max_r - min_r) + min_r
    else:
        r = g._nodes[ring_col]
        if min_r is None:
            min_r = r.min()
        if max_r is None:
            max_r = r.max()

    if reverse:
        r = -r + (max_r + min_r)

    if num_rings is not None:
        if ring_step is None:
            ring_step = (max_r - min_r) / num_rings
    else:
        if ring_step is None:
            num_rings = 5
            ring_step = (max_r - min_r) / num_rings
        else:
            num_rings = int((max_r - min_r) / ring_step)

    idx = r.reset_index(drop=True).index.to_series()
    if 'cudf' in str(g._nodes.__class__):
        import cudf
        x = r * cudf.Series(np.cos(idx))
        y = r * cudf.Series(np.sin(idx))
    else:
        x = r * pd.Series(np.cos(idx))
        y = r * pd.Series(np.sin(idx))

    domain_min = g._nodes[ring_col].min()
    domain_max = g._nodes[ring_col].max()
    domain_range = domain_max - domain_min
    domain_step = domain_range / num_rings

    axis = gen_axis(
        num_rings,
        domain_min,
        domain_step,
        min_r,
        (max_r - min_r) / num_rings,
        format_labels,
        reverse
    )

    if format_axis is not None:
        axis = format_axis(axis)

    #print('axis', axis)

    g2 = (
        g
          .nodes(lambda g: g._nodes.assign(x=x, y=y, r=r))
          .encode_axis(axis)
          .settings(url_params={
              'play': play_ms,
              'lockedR': True,
              'bg': '%23E2E2E2'  # Light grey due to labels being fixed to dark
          })
    )

    return g2
