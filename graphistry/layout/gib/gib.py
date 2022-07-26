from typing import List, Optional, Union
from typing_extensions import Literal
import pandas as pd

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .partition import partition
from .partitioned_layout import partitioned_layout
from .style import style_layout
from .treemap import treemap
logger = setup_logger(__name__)


def resolve_partition_key(g, partition_key=None):
    if partition_key is not None:
        return partition_key
    elif g._nodes is not None and 'partition' in g._nodes:
        return 'partition'
    elif g._nodes is not None and 'community' in g._nodes:
        return 'community'
    elif g._nodes is not None and 'cluster' in g._nodes:
        return 'cluster'
    else:
        return 'partition'


def group_in_a_box_layout(
    self,
    partition_alg=None,
    partition_params=None,
    layout_alg=None,
    layout_params=None,
    x=0,
    y=0,
    w=None,
    h=None,
    encode_colors=True,
    colors: Optional[List[str]] = None,
    partition_key: Optional[str] = None,
    engine: Union[Engine, Literal["auto"]] = "auto"
) -> 'Plottable':
    """
    Group nodes in a box layout, including node colors, and both CPU and GPU modes

    """
    from timeit import default_timer as timer
    start = timer()

    resolved_partition_key = resolve_partition_key(self, partition_key)
    #print('resolved_partition_key', resolved_partition_key)
    #print('engine', engine)

    if engine == "auto":
        if isinstance(self._edges, pd.DataFrame):
            engine = Engine.PANDAS
        else:
            try:
                import cudf
                if isinstance(self._edges, cudf.DataFrame):
                    engine = Engine.CUDF
                else:
                    raise ValueError('Could not infer engine, please specify')
            except:
                raise ValueError('Could not infer engine, please specify')

    g_partitioned = partition(
        self,
        partition_alg=partition_alg,
        partition_params=partition_params,
        partition_key=resolved_partition_key,
        engine=engine
    )
    print('type g_partitioned', type(g_partitioned))
    print('_nodes ::', type(g_partitioned._nodes))
    print('_edges ::', type(g_partitioned._edges))
    partition_offsets = treemap(
        g_partitioned, x=x, y=y, w=w, h=h,
        partition_key=resolved_partition_key,
        engine=engine
    )
    print('type partition_offsets', type(partition_offsets))
    g_positioned = partitioned_layout(
        g_partitioned,
        partition_offsets=partition_offsets,
        layout_alg=layout_alg,
        layout_params=layout_params,
        partition_key=resolved_partition_key,
        engine=engine
    )
    print('type g_positioned', type(g_positioned))
    print('_nodes ::', type(g_positioned._nodes))
    print('_edges ::', type(g_positioned._edges))
    out = style_layout(
        g_positioned,
        encode_color=encode_colors,
        colors=colors,
        partition_key=resolved_partition_key,
        engine=engine
    )
    print('type out', type(out))
    print('_nodes ::', type(out._nodes))
    print('_edges ::', type(out._edges))

    end = timer()
    logger.debug('GROUP IN THE BOX: %s s', end - start)
    return out
