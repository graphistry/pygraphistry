from typing import Any, Optional
import pandas as pd

from graphistry.Plottable import Plottable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


##############################################################################


class ASTObject(object):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self, name: Optional[str] = None):
        self._name = name
        pass

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame]) -> Plottable:
        raise RuntimeError('__call__ not implemented')
        
    def reverse(self) -> 'ASTObject':
        raise RuntimeError('reverse not implemented')


##############################################################################


class ASTNode(ASTObject):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self, filter_dict: Optional[dict] = None, name: Optional[str] = None):

        super().__init__(name)

        if filter_dict == {}:
            filter_dict = None
        self._filter_dict = filter_dict

    def __repr__(self) -> str:
        return f'ASTNode(filter_dict={self._filter_dict}, name={self._name})'

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame]) -> Plottable:
        out_g = (g
            .nodes(prev_node_wavefront if prev_node_wavefront is not None else g._nodes)
            .filter_nodes_by_dict(self._filter_dict)
            .edges(g._edges[:0])
        )
        if self._name is not None:
            out_g = out_g.nodes(out_g._nodes.assign(**{self._name: True}))

        logger.debug(f'CALL NODE {self} ===>')
        logger.debug(out_g._nodes)
        logger.debug(out_g._edges)
        logger.debug('----------------------------------------')

        return out_g

    def reverse(self) -> 'ASTNode':
        return self

n = ASTNode  # noqa: E305


###############################################################################


DEFAULT_HOPS = 1
DEFAULT_FIXED_POINT = False
DEFAULT_DIRECTION = 'forward'
DEFAULT_FILTER_DICT = None

class ASTEdge(ASTObject):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(
        self,
        direction: Optional[str] = DEFAULT_DIRECTION,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        name: Optional[str] = None
    ):

        super().__init__(name)

        if direction not in ['forward', 'reverse', 'undirected']:
            raise ValueError('direction must be one of "forward", "reverse", or "undirected"')
        if source_node_match == {}:
            source_node_match = None
        if edge_match == {}:
            edge_match = None
        if destination_node_match == {}:
            destination_node_match = None

        self._hops = hops
        self._to_fixed_point = to_fixed_point
        self._direction = direction
        self._source_node_match = source_node_match
        self._edge_match = edge_match
        self._destination_node_match = destination_node_match

    def __repr__(self) -> str:
        return f'ASTEdge(direction={self._direction}, edge_match={self._edge_match}, hops={self._hops}, to_fixed_point={self._to_fixed_point}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, name={self._name})'

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame]) -> Plottable:

        out_g = g.hop(
            nodes=prev_node_wavefront,
            hops=self._hops,
            to_fixed_point=self._to_fixed_point,
            direction=self._direction,
            source_node_match=self._source_node_match,
            edge_match=self._edge_match,
            destination_node_match=self._destination_node_match,
            return_as_wave_front=True
        )

        if self._name is not None:
            out_g = out_g.edges(out_g._edges.assign(**{self._name: True}))

        logger.debug(f'CALL EDGE {self} ===>')
        logger.debug(out_g._nodes)
        logger.debug(out_g._edges)
        logger.debug('----------------------------------------')

        return out_g

    def reverse(self) -> 'ASTEdge':
        # updates both edges and nodes
        return ASTEdge(
            direction=(
                'forward' if self._direction == 'reverse' else 'reverse'
            ) if self._direction != 'undirected' else 'undirected',
            edge_match=self._edge_match,
            hops=self._hops,
            to_fixed_point=self._to_fixed_point,
            source_node_match=self._destination_node_match,
            destination_node_match=self._source_node_match
        )
e = ASTEdge  # noqa: E305

class ASTEdgeForward(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self, 
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None
    ):
        super().__init__(
            direction='forward',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name
        )

    def __repr__(self) -> str:
        return f'ASTEdgeForward(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name})'

e_forward = ASTEdgeForward  # noqa: E305

class ASTEdgeReverse(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None
    ):
        super().__init__(
            direction='reverse',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name
        )
    
    def __repr__(self) -> str:
        return f'ASTEdgeReverse(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name})'

e_reverse = ASTEdgeReverse  # noqa: E305

class ASTEdgeUndirected(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None
    ):
        super().__init__(
            direction='undirected',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name
        )

    def __repr__(self) -> str:
        return f'ASTEdgeUndirected(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name})'

e_undirected = ASTEdgeUndirected  # noqa: E305
