from typing import Any, List, Optional, Union
from typing_extensions import Literal
from graphistry.hyper_dask import hypergraph as hypergraph_new
from graphistry.plugins_types.hypergraph import HypergraphResult
from .util import setup_logger
logger = setup_logger(__name__)


class Hypergraph(object):

    @staticmethod
    def hypergraph(
        g, raw_events=None,
        *,
        entity_types: Optional[List[str]] = None, opts: dict = {},
        drop_na: bool = True, drop_edge_attrs: bool = False, verbose: bool = True, direct: bool = False,
        engine: str = 'pandas', npartitions: Optional[int] = None, chunksize: Optional[int] = None,
        from_edges: bool = False,
        return_as: Literal['graph', 'all', 'entities', 'events', 'edges', 'nodes'] = 'graph'
    ) -> Union[HypergraphResult, Any]:
        """
            raw_events can be pd.DataFrame or cudf.DataFrame
        """

        out = hypergraph_new(
            g, raw_events,
            entity_types=entity_types, opts=opts,
            drop_na=drop_na, drop_edge_attrs=drop_edge_attrs, verbose=verbose, direct=direct,
            engine=engine, npartitions=npartitions, chunksize=chunksize,
            from_edges=from_edges, return_as=return_as)

        # Route return based on return_as parameter
        result_dict: HypergraphResult = {
            'entities': out.entities,
            'events': out.events,
            'edges': out.edges,
            'nodes': out.nodes,
            'graph': out.graph
        }

        if return_as == 'graph':
            # Return just the Plottable graph (for chaining)
            return result_dict['graph']
        elif return_as == 'all':
            # Return full dict with all components (backward compatible)
            return result_dict
        elif return_as == 'entities':
            return out.entities
        elif return_as == 'events':
            return out.events
        elif return_as == 'edges':
            return out.edges
        elif return_as == 'nodes':
            return out.nodes
        else:
            # Should never reach here due to Literal typing
            return result_dict
