import logging
from typing import Optional, cast
import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .predicates.ASTPredicate import ASTPredicate
from .predicates.is_in import (
    is_in, IsIn
)
from .predicates.categorical import (
    duplicated, Duplicated,
)
from .predicates.temporal import (
    is_month_start, IsMonthStart,
    is_month_end, IsMonthEnd,
    is_quarter_start, IsQuarterStart,
    is_quarter_end, IsQuarterEnd,
    is_year_start, IsYearStart,
    is_leap_year, IsLeapYear
)
from .predicates.numeric import (
    gt, GT,
    lt, LT,
    ge, GE,
    le, LE,
    eq, EQ,
    ne, NE,
    between, Between,
    isna, IsNA,
    notna, NotNA
)
from .predicates.str import (
    contains, Contains,
    startswith, Startswith,
    endswith, Endswith,
    match, Match,
    isnumeric, IsNumeric,
    isalpha, IsAlpha,
    isdigit, IsDigit,
    islower, IsLower,
    isupper, IsUpper,
    isspace, IsSpace,
    isalnum, IsAlnum,
    isdecimal, IsDecimal,
    istitle, IsTitle,
    isnull, IsNull,
    notnull, NotNull
)
from .filter_by_dict import filter_by_dict


logger = setup_logger(__name__)


##############################################################################


class ASTObject(object):
    """
    Internal, not intended for use outside of this module.
    These are operator-level expressions used as g.chain(List<ASTObject>)
    """
    def __init__(self, name: Optional[str] = None):
        self._name = name
        pass

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame], target_wave_front: Optional[pd.DataFrame]) -> Plottable:
        raise RuntimeError('__call__ not implemented')
        
    def reverse(self) -> 'ASTObject':
        raise RuntimeError('reverse not implemented')


##############################################################################


class ASTNode(ASTObject):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self, filter_dict: Optional[dict] = None, name: Optional[str] = None, query: Optional[str] = None):

        super().__init__(name)

        if filter_dict == {}:
            filter_dict = None
        self._filter_dict = filter_dict
        self._query = query

    def __repr__(self) -> str:
        return f'ASTNode(filter_dict={self._filter_dict}, name={self._name})'

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame], target_wave_front: Optional[pd.DataFrame]) -> Plottable:
        out_g = (g
            .nodes(prev_node_wavefront if prev_node_wavefront is not None else g._nodes)
            .filter_nodes_by_dict(self._filter_dict)
            .nodes(lambda g_dynamic: g_dynamic._nodes.query(self._query) if self._query is not None else g_dynamic._nodes)
            .edges(g._edges[:0])
        )
        if target_wave_front is not None:
            assert g._node is not None
            reduced_nodes = cast(pd.DataFrame, out_g._nodes).merge(target_wave_front[[g._node]], on=g._node, how='inner')
            out_g = out_g.nodes(reduced_nodes)

        if self._name is not None:
            out_g = out_g.nodes(out_g._nodes.assign(**{self._name: True}))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('CALL NODE %s ====>\nnodes:\n%s\nedges:\n%s\n', self, out_g._nodes, out_g._edges)
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
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
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
        self._source_node_query = source_node_query
        self._destination_node_query = destination_node_query
        self._edge_query = edge_query

    def __repr__(self) -> str:
        return f'ASTEdge(direction={self._direction}, edge_match={self._edge_match}, hops={self._hops}, to_fixed_point={self._to_fixed_point}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, name={self._name}, source_node_query={self._source_node_query}, destination_node_query={self._destination_node_query}, edge_query={self._edge_query})'

    def __call__(self, g: Plottable, prev_node_wavefront: Optional[pd.DataFrame], target_wave_front: Optional[pd.DataFrame]) -> Plottable:

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('----------------------------------------')
            logger.debug('@CALL EDGE START {%s} ===>\n', self)
            logger.debug('prev_node_wavefront:\n%s\n', prev_node_wavefront)
            logger.debug('target_wave_front:\n%s\n', target_wave_front)
            logger.debug('g._nodes:\n%s\n', g._nodes)
            logger.debug('g._edges:\n%s\n', g._edges)
            logger.debug('----------------------------------------')

        out_g = g.hop(
            nodes=prev_node_wavefront,
            hops=self._hops,
            to_fixed_point=self._to_fixed_point,
            direction=self._direction,
            source_node_match=self._source_node_match,
            edge_match=self._edge_match,
            destination_node_match=self._destination_node_match,
            return_as_wave_front=True,
            target_wave_front=target_wave_front,
            source_node_query=self._source_node_query,
            destination_node_query=self._destination_node_query,
            edge_query=self._edge_query
        )

        if self._name is not None:
            out_g = out_g.edges(out_g._edges.assign(**{self._name: True}))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('/CALL EDGE END {%s} ===>\nnodes:\n%s\nedges:\n%s\n', self, out_g._nodes, out_g._edges)
            logger.debug('----------------------------------------')

        return out_g

    def reverse(self) -> 'ASTEdge':
        # updates both edges and nodes
        if self._direction == 'reverse':
            direction = 'forward'
        elif self._direction == 'forward':
            direction = 'reverse'
        else:
            direction = 'undirected'
        return ASTEdge(
            direction=direction,
            edge_match=self._edge_match,
            hops=self._hops,
            to_fixed_point=self._to_fixed_point,
            source_node_match=self._destination_node_match,
            destination_node_match=self._source_node_match,
            source_node_query=self._destination_node_query,
            destination_node_query=self._source_node_query,
            edge_query=self._edge_query
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
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None
    ):
        super().__init__(
            direction='forward',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name,
            source_node_query=source_node_query,
            destination_node_query=destination_node_query,
            edge_query=edge_query
        )

    def __repr__(self) -> str:
        return f'ASTEdgeForward(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name}, source_node_query={self._source_node_query}, destination_node_query={self._destination_node_query}, edge_query={self._edge_query})'

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
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None
    ):
        super().__init__(
            direction='reverse',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name,
            source_node_query=source_node_query,
            destination_node_query=destination_node_query,
            edge_query=edge_query
        )
    
    def __repr__(self) -> str:
        return f'ASTEdgeReverse(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name}, source_node_query={self._source_node_query}, destination_node_query={self._destination_node_query}, edge_query={self._edge_query})'

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
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None
    ):
        super().__init__(
            direction='undirected',
            edge_match=edge_match,
            hops=hops,
            source_node_match=source_node_match,
            destination_node_match=destination_node_match,
            to_fixed_point=to_fixed_point,
            name=name,
            source_node_query=source_node_query,
            destination_node_query=destination_node_query,
            edge_query=edge_query
        )

    def __repr__(self) -> str:
        return f'ASTEdgeUndirected(edge_match={self._edge_match}, hops={self._hops}, source_node_match={self._source_node_match}, destination_node_match={self._destination_node_match}, to_fixed_point={self._to_fixed_point}, name={self._name}, source_node_query={self._source_node_query}, destination_node_query={self._destination_node_query}, edge_query={self._edge_query})'

e_undirected = ASTEdgeUndirected  # noqa: E305
