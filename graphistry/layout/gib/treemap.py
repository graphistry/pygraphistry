import math, squarify
from typing import Dict, List, Optional

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


def treemap(
    self: Plottable,
    x=0,
    y=0,
    w: Optional[float] = None,
    h: Optional[float] = None,
    partition_key='partition',
    engine: Engine = Engine.PANDAS
) -> Dict[str, Dict[int, float]]:
    """
    Group nodes by partition key and compute treemap cell positions
    Output dictionary format is prop_name -> partition id -> prop_value
    """
    from timeit import default_timer as timer
    start = timer()
    
    w = w or h
    h = h or w
    if w is None or h is None:
        w = math.sqrt(30 * len(self._nodes))
        h = w

    partitions_sorted_df = (
        self._nodes
            .groupby(partition_key).agg({self._node: 'count'})
            .sort_values(by=self._node, ascending=False)
    ).reset_index()

    sorted_box_sizes: List[int] = partitions_sorted_df[self._node].to_numpy().tolist()
    normalized: List[float] = squarify.normalize_sizes(sorted_box_sizes, w, h)
    # [ {'x', 'y', 'dx', 'dy'} ]
    rects: List[dict] = squarify.squarify(normalized, x, y, w, h)

    props = ['x', 'y', 'dx', 'dy']
    propname_to_partition_to_prop = {
        prop: {
            p: rects[i][prop]
            for i, p in enumerate(
                partitions_sorted_df.reset_index()[partition_key].to_numpy()
            )
        } for prop in props
    }
    # propname_to_partition_to_prop.keys(), 'x ->', propname_to_partition_to_prop['x']

    end = timer()
    logger.debug('treemap: %s s', end - start)
    return propname_to_partition_to_prop
