from typing import Any, Callable, Dict, Optional, Union
import pandas as pd

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


def layout_bulk_mode(
    self: Plottable,
    nodes: pd.DataFrame,  # or cudf.DataFrame
    partition_key: str,
    layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]],
    layout_params: Optional[Dict[str, Any]],
    engine: Engine
) -> pd.DataFrame:  # or cudf.DataFrame
    """
    Handles layout for bulk mode. Applies layout to the entire graph
     
    Assumes cross-partition edges already removed

    :param nodes: The nodes of the graph.
    :type nodes: DataFrame
    :param partition_key: The partition key.
    :type partition_key: str
    :param layout_alg: Layout algorithm to be applied.
    :type layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]]
    :param layout_params: Parameters for the layout algorithm.
    :type layout_params: Optional[Dict[str, Any]]
    :param engine: The engine being used (Pandas or CUDF).
    :type engine: Engine
    :return: The resulting DataFrame of positioned nodes.
    """

    if callable(layout_alg):
        positioned_graph = layout_alg(self)
        layout_name = 'custom'

    elif engine == Engine.PANDAS:

        if layout_alg is None:
            layout_name = 'force_atlas2'
            positioned_graph = self.fa2_layout(
                fa2_params=layout_params,
                circle_layout_params={'partition_by': partition_key},
                partition_key=partition_key
            )
        else:
            layout_name = layout_alg or 'fr'
            positioned_graph = self.layout_igraph(
                layout=layout_name,
                params=layout_params if layout_params is not None else {}
            )
    elif engine == Engine.CUDF:

        if layout_alg is None:
            layout_name = 'force_atlas2'
            positioned_graph = self.fa2_layout(
                fa2_params=layout_params,
                circle_layout_params={'partition_by': partition_key},
                partition_key=partition_key
            )
        else:
            layout_name = layout_alg
            positioned_graph = self.layout_cugraph(
                layout=layout_name,
                params=layout_params if layout_params is not None else {}
            )
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    positioned_graph = positioned_graph.nodes(
        positioned_graph._nodes.assign(type=layout_name)
    )

    return positioned_graph._nodes
