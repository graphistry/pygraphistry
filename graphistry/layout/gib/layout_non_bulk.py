from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


def layout_non_bulk_mode(
    self: Plottable,
    node_partitions: List[pd.DataFrame],  # or cudf.DataFrame
    remaining: pd.DataFrame,  # or cudf.DataFrame
    partition_key: str,
    layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]],
    layout_params: Optional[Dict[str, Any]],
    engine: Engine,
    self_selfless: Plottable
) -> Tuple[List[pd.DataFrame], float, float, Dict[int, Tuple[int, float]]]:
    """
    Handles the layout in non-bulk mode by applying the layout separately for each partition.

    :param node_partitions: List of DataFrames for node partitions.
    :type node_partitions: List[pd.DataFrame]
    :param remaining: DataFrame of remaining nodes after filtering.
    :type remaining: DataFrame
    :param partition_key: The partition key.
    :type partition_key: str
    :param layout_alg: Layout algorithm to be applied.
    :type layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]]
    :param layout_alg: Layout algorithm to be applied.
    :type layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]]
    :param layout_params: Parameters for the layout algorithm.
    :type layout_params: Optional[Dict[str, Any]]
    :param engine: The engine being used (Pandas or CUDF).
    :type engine: Engine
    :param self_selfless: Graph excluding self-edges.
    :type self_selfless: Plottable
    :return: Tuple containing node partitions, layout time, keep time, and layout by size.
    """

    s_keep = 0.0
    s_layout = 0.0
    s_layout_by_size = {}

    for partition in remaining[partition_key].to_numpy():
        start_i = timer()

        node_ids = self._nodes[self._nodes[partition_key] == partition][self._node]
        subgraph_g = self_selfless.nodes(self._nodes).keep_nodes({self._node: node_ids})
        start_i_mid = timer()
        s_keep += start_i_mid - start_i

        niter = min(len(subgraph_g._nodes), 300)
        if callable(layout_alg):
            positioned_subgraph_g = layout_alg(subgraph_g)
            layout_name = 'custom'
        elif engine == Engine.PANDAS:
            layout_name = layout_alg or 'fr'
            try:
                positioned_subgraph_g = subgraph_g.layout_igraph(
                    layout=layout_name,
                    params={**({'niter': niter} if layout_name == 'fr' else {}), **(layout_params or {})}
                )
            except Exception:
                logger.warning(
                    "igraph layout failed for partition %s; using fallback", partition, exc_info=True
                )
                node_count = len(subgraph_g._nodes)
                if node_count == 0:
                    positioned_subgraph_g = subgraph_g.nodes(
                        subgraph_g._nodes.assign(x=0.0, y=0.0)
                    )
                else:
                    angles = pd.Series(np.linspace(0, 2 * np.pi, node_count, endpoint=False), dtype='float32')
                    coords = pd.DataFrame({
                        'x': angles.map(np.cos).astype('float32'),
                        'y': angles.map(np.sin).astype('float32')
                    })
                    positioned_subgraph_g = subgraph_g.nodes(
                        subgraph_g._nodes.assign(x=coords['x'].values, y=coords['y'].values)
                    )
        elif engine == Engine.CUDF:
            layout_name = layout_alg or 'force_atlas2'
            positioned_subgraph_g = subgraph_g.layout_cugraph(
                layout=layout_name,
                params={**({'max_iter': niter} if layout_name == 'force_atlas2' else {}), **(layout_params or {})}
            )
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        positioned_subgraph_g = positioned_subgraph_g.nodes(
            positioned_subgraph_g._nodes.assign(type=layout_name)
        )
        node_partitions.append(positioned_subgraph_g._nodes)
        end_i = timer()
        s_layout += end_i - start_i_mid

        if len(positioned_subgraph_g._nodes) not in s_layout_by_size:
            s_layout_by_size[len(positioned_subgraph_g._nodes)] = (0, 0.0)
        n, t = s_layout_by_size[len(positioned_subgraph_g._nodes)]
        s_layout_by_size[len(positioned_subgraph_g._nodes)] = (n + 1, t + (end_i - start_i_mid))

    return node_partitions, s_layout, s_keep, s_layout_by_size
