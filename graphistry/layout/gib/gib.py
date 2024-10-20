from typing import Any, Callable, Dict, List, Optional, Union
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


def resolve_partition_key(g, partition_key=None, partition_alg: Optional[str] = None):
    if partition_key is not None:
        return partition_key
    elif partition_alg is not None:
        return partition_alg
    elif g._nodes is not None and 'partition' in g._nodes:
        return 'partition'
    elif g._nodes is not None and 'community' in g._nodes:
        return 'community'
    elif g._nodes is not None and 'cluster' in g._nodes:
        return 'cluster'
    else:
        return 'partition'


def group_in_a_box_layout(
    self: Plottable,
    partition_alg: Optional[str] = None,
    partition_params: Optional[Dict[str, Any]] = None,
    layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]] = None,
    layout_params: Optional[Dict[str, Any]] = None,
    x: float = 0,
    y: float = 0,
    w: Optional[float] = None,
    h: Optional[float] = None,
    encode_colors: bool = True,
    colors: Optional[List[str]] = None,
    partition_key: Optional[str] = None,
    engine: Union[Engine, Literal["auto"]] = "auto"
) -> 'Plottable':
    """
    Perform a group-in-a-box layout on a graph, supporting both CPU and GPU execution modes.

    This layout algorithm organizes nodes into rectangular bounding boxes based on a partitioning algorithm.
    It supports various layout algorithms within each partition and optional color encoding based on the partition.

    Supports passing in a custom per-partition layout algorithm handler.

    :param partition_alg: (optional) The algorithm to use for partitioning the graph nodes. Examples include 'community' or 'louvain'.
    :type partition_alg: Optional[str]
    :param partition_params: (optional) Parameters for the partition algorithm, passed as a dictionary.
    :type partition_params: Optional[Dict[str, Any]]
    :param layout_alg: (optional) The layout algorithm to arrange nodes within each partition.

        - In GPU mode, defaults to :meth:`graphistry.layout.fa2.fa2_layout` for individual partitions.

        - CPU mode defaults to :meth:`graphistry.plugins.igraph.layout_igraph` with layout `"fr"`.

        - Can be a string referring to an igraph algorithm (CPU), cugraph algorithm (GPU), or a callable function.

    :type layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]]
    :param layout_params: (optional) Parameters for the layout algorithm.
    :type layout_params: Optional[Dict[str, Any]]
    :param x: (optional) The x-coordinate for the top-left corner of the layout. Default is 0.
    :type x: float
    :param y: (optional) The y-coordinate for the top-left corner of the layout. Default is 0.
    :type y: float
    :param w: (optional) The width of the layout. If None, it will be automatically determined based on the number of partitions.
    :type w: Optional[float]
    :param h: (optional) The height of the layout. If None, it will be automatically determined based on the number of partitions.
    :type h: Optional[float]
    :param encode_colors: (optional) Whether to apply color encoding to nodes based on partitions. Default is True.
    :type encode_colors: bool
    :param colors: (optional) List of colors to use for the partitions. If None, default colors will be applied.
    :type colors: Optional[List[str]]
    :param partition_key: (optional) The key for partitioning nodes. If not provided, defaults to a relevant partitioning key for the algorithm.
    :type partition_key: Optional[str]
    :param engine: (optional) The execution engine for the layout, either "auto" (default), "cpu", or "gpu".
    :type engine: Union[graphistry.Engine.EngineAbstract, Literal["auto"]]

    :returns: A graph object with nodes arranged in a group-in-a-box layout.
    :rtype: graphistry.Plottable.Plottable

    **Example 1: Basic GPU Group-in-a-Box Layout Using ECG Community Detection**
        ::
        
            g_final = g.group_in_a_box_layout(partition_alg='ecg')

    **Example 2: Group-in-a-Box on a precomputed partition key**
        ::

            g_partitioned = g.compute_cugraph('ecg')
            g_final = g_partitioned.group_in_a_box_layout(partition_key='ecg')

    **Example 3: Custom Group-in-a-Box Layout with FA2 for Layout and Color Encoding**
        ::
        
            g_final = g.group_in_a_box_layout(
                partition_alg='louvain',
                partition_params={'resolution': 1.0},
                layout_alg=lambda g: fa2_with_circle_singletons(g),
                encode_colors=True,
                colors=['#ff0000', '#00ff00', '#0000ff']
            )

    **Example 4: Advanced Usage with Custom Bounding Box and GPU Execution**
        ::
        
            g_final = g.group_in_a_box_layout(
                partition_alg='louvain',
                layout_alg='force_atlas2',
                x=100, y=100, w=500, h=500,  # Custom bounding box
                engine='gpu'  # Use GPU for faster layout
            )
    """
    from timeit import default_timer as timer
    start = timer()

    resolved_partition_key = resolve_partition_key(self, partition_key, partition_alg)
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
            except Exception:
                raise ValueError('Could not infer engine, please specify')

    g_partitioned = partition(
        self,
        partition_alg=partition_alg,
        partition_params=partition_params,
        partition_key=resolved_partition_key,
        engine=engine
    )
    partition_offsets = treemap(
        g_partitioned, x=x, y=y, w=w, h=h,
        partition_key=resolved_partition_key,
        engine=engine
    )
    g_positioned = partitioned_layout(
        g_partitioned,
        partition_offsets=partition_offsets,
        layout_alg=layout_alg,
        layout_params=layout_params,
        partition_key=resolved_partition_key,
        engine=engine
    )
    out = style_layout(
        g_positioned,
        encode_color=encode_colors,
        colors=colors,
        partition_key=resolved_partition_key,
        engine=engine
    )

    # Dict[str, Dict[int, float]]
    out._partition_offsets = partition_offsets

    end = timer()
    logger.debug('GROUP IN THE BOX: %s s', end - start)
    return out
