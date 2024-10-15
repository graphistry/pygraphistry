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
    layout_params: Dict[str, Any],
    engine: Engine
) -> pd.DataFrame:  # or cudf.DataFrame
    """
    Handles layout for bulk mode. Applies layout to the entire graph
     
    Assumes cross-partition edges already removed

    :param nodes: The nodes of the graph.
    :param partition_key: The partition key.
    :param layout_alg: Layout algorithm to be applied.
    :param layout_params: Parameters for the layout algorithm.
    :param engine: The engine being used (Pandas or CUDF).
    :return: The resulting DataFrame of positioned nodes.
    """

    if callable(layout_alg):
        positioned_graph = layout_alg(self)
        layout_name = 'custom'

    elif engine == Engine.PANDAS:

        if layout_alg is None:
            layout_name = 'force_atlas2'
            positioned_graph = self.fa2_layout(
                fa2_params={**(
                    {'max_iter': min(len(nodes), 300)}
                    if layout_name == 'force_atlas2' else {}
                ), **layout_params},
                circle_layout_params={'partition_by': partition_key},
                partition_key=partition_key
            )
        else:
            layout_name = layout_alg or 'fr'
            positioned_graph = self.layout_igraph(
                layout=layout_name,
                params={**({'niter': min(len(nodes), 300)} if layout_name == 'fr' else {}), **layout_params}
            )
    elif engine == Engine.CUDF:

        if layout_alg is None:
            layout_name = 'force_atlas2'
            positioned_graph = self.fa2_layout(
                fa2_params={**(
                    {'max_iter': min(len(nodes), 300)}
                    if layout_name == 'force_atlas2' else {}
                ), **layout_params},
                circle_layout_params={'partition_by': partition_key},
                partition_key=partition_key
            )
        else:
            layout_name = layout_alg
            positioned_graph = self.layout_cugraph(
                layout=layout_name,
                params={**({'max_iter': min(len(nodes), 300)} if layout_name == 'force_atlas2' else {}), **layout_params}
            )
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    positioned_graph = positioned_graph.nodes(
        positioned_graph._nodes.assign(type=layout_name)
    )

    print('BULK generated node columns', positioned_graph._nodes.columns)

    return positioned_graph._nodes
