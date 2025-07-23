from abc import abstractmethod
import logging
from typing import Any, TYPE_CHECKING, Dict, Optional, Union, cast
from typing_extensions import Literal
import pandas as pd
from graphistry.Engine import Engine

from graphistry.Plottable import Plottable
from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.util import setup_logger
from graphistry.utils.json import JSONVal, is_json_serializable
from .predicates.ASTPredicate import ASTPredicate
from .predicates.from_json import from_json as predicates_from_json

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
    is_year_end, IsYearEnd,
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
from .typing import DataFrameT


logger = setup_logger(__name__)

##############################################################################


class ASTObject(ASTSerializable):
    """
    Internal, not intended for use outside of this module.
    These are operator-level expressions used as g.chain(List<ASTObject>)
    """
    def __init__(self, name: Optional[str] = None):
        self._name = name
        pass

    @abstractmethod
    def __call__(
        self,
        g: Plottable,
        prev_node_wavefront: Optional[DataFrameT],
        target_wave_front: Optional[DataFrameT],
        engine: Engine
    ) -> Plottable:
        raise RuntimeError('__call__ not implemented')
        
    @abstractmethod
    def reverse(self) -> 'ASTObject':
        raise RuntimeError('reverse not implemented')


##############################################################################


def assert_record_match(d: Dict) -> None:
    assert isinstance(d, dict)
    for k, v in d.items():
        assert isinstance(k, str)
        assert isinstance(v, ASTPredicate) or is_json_serializable(v)

def maybe_filter_dict_from_json(d: Dict, key: str) -> Optional[Dict]:
    if key not in d:
        return None
    if key in d and isinstance(d[key], dict):
        return {
            k: predicates_from_json(v) if isinstance(v, dict) else v
            for k, v in d[key].items()
        }
    elif key in d and d[key] is not None:
        raise ValueError('filter_dict must be a dict or None')
    else:
        return None

##############################################################################


class ASTNode(ASTObject):
    """
    Internal, not intended for use outside of this module.
    """
    def __init__(self, filter_dict: Optional[dict] = None, name: Optional[str] = None, query: Optional[str] = None):

        super().__init__(name)

        if filter_dict == {}:
            filter_dict = None
        self.filter_dict = filter_dict
        self.query = query

    def __repr__(self) -> str:
        return f'ASTNode(filter_dict={self.filter_dict}, name={self._name})'
    
    def _validate_fields(self) -> None:
        """Validate node fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        # Validate filter_dict
        if self.filter_dict is not None:
            if not isinstance(self.filter_dict, dict):
                raise GFQLTypeError(
                    ErrorCode.E201,
                    "filter_dict must be a dictionary",
                    field="filter_dict",
                    value=type(self.filter_dict).__name__,
                    suggestion="Use filter_dict={'column': 'value'}",
                )

            # Validate each key in filter_dict
            for key, value in self.filter_dict.items():
                if not isinstance(key, str):
                    raise GFQLTypeError(
                        ErrorCode.E102,
                        "Filter keys must be strings",
                        field=f"filter_dict.{key}",
                        value=key,
                        suggestion="Use string column names as keys",
                    )

                # Validate value is either ASTPredicate or json-serializable
                if not (isinstance(value, ASTPredicate) or is_json_serializable(value)):
                    raise GFQLTypeError(
                        ErrorCode.E201,
                        "Filter values must be predicates or JSON-serializable",
                        field=f"filter_dict.{key}",
                        value=type(value).__name__,
                        suggestion="Use predicates like gt(5) or simple values",
                    )

        # Validate name
        if self._name is not None and not isinstance(self._name, str):
            raise GFQLTypeError(ErrorCode.E204, "name must be a string", field="name", value=type(self._name).__name__)

        # Validate query
        if self.query is not None and not isinstance(self.query, str):
            raise GFQLTypeError(
                ErrorCode.E205, "query must be a string", field="query", value=type(self.query).__name__
            )

    def _get_child_validators(self) -> list:
        """Return predicates that need validation."""
        children = []
        if self.filter_dict:
            for value in self.filter_dict.values():
                if isinstance(value, ASTPredicate):
                    children.append(value)
        return children

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Node',
            'filter_dict': {
                k: v.to_json() if isinstance(v, ASTPredicate) else v
                for k, v in self.filter_dict.items()
                if v is not None
            } if self.filter_dict is not None else {},
            **({'name': self._name} if self._name is not None else {}),
            **({'query': self.query } if self.query is not None else {})
        }
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTNode':
        out = ASTNode(
            filter_dict=maybe_filter_dict_from_json(d, 'filter_dict'),
            name=d['name'] if 'name' in d else None,
            query=d['query'] if 'query' in d else None
        )
        if validate:
            out.validate()
        return out

    def __call__(
        self,
        g: Plottable,
        prev_node_wavefront: Optional[DataFrameT],
        target_wave_front: Optional[DataFrameT],
        engine: Engine
    ) -> Plottable:
        out_g = (g
            .nodes(prev_node_wavefront if prev_node_wavefront is not None else g._nodes)
            .filter_nodes_by_dict(self.filter_dict)
            .nodes(lambda g_dynamic: g_dynamic._nodes.query(self.query) if self.query is not None else g_dynamic._nodes)
            .edges(g._edges[:0])
        )
        if target_wave_front is not None:
            assert g._node is not None
            reduced_nodes = cast(DataFrameT, out_g._nodes).merge(target_wave_front[[g._node]], on=g._node, how='inner')
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

Direction = Literal['forward', 'reverse', 'undirected']

DEFAULT_HOPS = 1
DEFAULT_FIXED_POINT = False
DEFAULT_DIRECTION: Direction = 'forward'
DEFAULT_FILTER_DICT = None

class ASTEdge(ASTObject):
    """
    Internal, not intended for use outside of this module.
    """

    def __init__(
        self,
        direction: Optional[Direction] = DEFAULT_DIRECTION,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
        name: Optional[str] = None,
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

        self.hops = hops
        self.to_fixed_point = to_fixed_point
        self.direction : Direction = direction
        self.source_node_match = source_node_match
        self.edge_match = edge_match
        self.destination_node_match = destination_node_match
        self.source_node_query = source_node_query
        self.destination_node_query = destination_node_query
        self.edge_query = edge_query

    def __repr__(self) -> str:
        return f'ASTEdge(direction={self.direction}, edge_match={self.edge_match}, hops={self.hops}, to_fixed_point={self.to_fixed_point}, source_node_match={self.source_node_match}, destination_node_match={self.destination_node_match}, name={self._name}, source_node_query={self.source_node_query}, destination_node_query={self.destination_node_query}, edge_query={self.edge_query})'

    def _validate_fields(self) -> None:
        """Validate edge fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLSyntaxError

        # Validate hops
        if self.hops is not None:
            if not isinstance(self.hops, int) or self.hops < 1:
                raise GFQLTypeError(
                    ErrorCode.E103,
                    "hops must be a positive integer or None",
                    field="hops",
                    value=self.hops,
                    suggestion="Use hops=2 for specific count, or to_fixed_point=True for unbounded",
                )

        # Validate to_fixed_point
        if not isinstance(self.to_fixed_point, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "to_fixed_point must be a boolean",
                field="to_fixed_point",
                value=type(self.to_fixed_point).__name__,
            )

        # Validate direction
        if self.direction not in ['forward', 'reverse', 'undirected']:
            raise GFQLSyntaxError(
                ErrorCode.E104,
                f"Invalid edge direction: {self.direction}",
                field="direction",
                value=self.direction,
                suggestion='Use "forward", "reverse", or "undirected"',
            )

        # Validate filter dicts
        for filter_name, filter_dict in [
            ('source_node_match', self.source_node_match),
            ('edge_match', self.edge_match),
            ('destination_node_match', self.destination_node_match),
        ]:
            if filter_dict is not None:
                if not isinstance(filter_dict, dict):
                    raise GFQLTypeError(
                        ErrorCode.E201,
                        f"{filter_name} must be a dictionary",
                        field=filter_name,
                        value=type(filter_dict).__name__,
                    )

                for key, value in filter_dict.items():
                    if not isinstance(key, str):
                        raise GFQLTypeError(
                            ErrorCode.E102, "Filter keys must be strings", field=f"{filter_name}.{key}", value=key
                        )

                    if not (isinstance(value, ASTPredicate) or is_json_serializable(value)):
                        raise GFQLTypeError(
                            ErrorCode.E201,
                            "Filter values must be predicates or JSON-serializable",
                            field=f"{filter_name}.{key}",
                            value=type(value).__name__,
                        )

        # Validate name
        if self._name is not None and not isinstance(self._name, str):
            raise GFQLTypeError(ErrorCode.E204, "name must be a string", field="name", value=type(self._name).__name__)

        # Validate query strings
        for query_name, query_value in [
            ("source_node_query", self.source_node_query),
            ("destination_node_query", self.destination_node_query),
            ("edge_query", self.edge_query),
        ]:
            if query_value is not None and not isinstance(query_value, str):
                raise GFQLTypeError(
                    ErrorCode.E205, f"{query_name} must be a string", field=query_name, value=type(query_value).__name__
                )

    def _get_child_validators(self) -> list:
        """Return predicates that need validation."""
        children = []
        for filter_dict in [self.source_node_match, self.edge_match, self.destination_node_match]:
            if filter_dict:
                for value in filter_dict.values():
                    if isinstance(value, ASTPredicate):
                        children.append(value)
        return children

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Edge',
            'hops': self.hops,
            'to_fixed_point': self.to_fixed_point,
            'direction': self.direction,
            **({'source_node_match': {
                k: v.to_json() if isinstance(v, ASTPredicate) else v
                for k, v in self.source_node_match.items()
                if v is not None
            }} if self.source_node_match is not None else {}),
            **({'edge_match': {
                k: v.to_json() if isinstance(v, ASTPredicate) else v
                for k, v in self.edge_match.items()
                if v is not None
            }} if self.edge_match is not None else {}),
            **({'destination_node_match': {
                k: v.to_json() if isinstance(v, ASTPredicate) else v
                for k, v in self.destination_node_match.items()
                if v is not None
            }} if self.destination_node_match is not None else {}),
            **({'name': self._name} if self._name is not None else {}),
            **({'source_node_query': self.source_node_query} if self.source_node_query is not None else {}),
            **({'destination_node_query': self.destination_node_query} if self.destination_node_query is not None else {}),
            **({'edge_query': self.edge_query} if self.edge_query is not None else {})
        }
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTEdge':
        out = ASTEdge(
            direction=d['direction'] if 'direction' in d else None,
            edge_match=maybe_filter_dict_from_json(d, 'edge_match'),
            hops=d['hops'] if 'hops' in d else None,
            to_fixed_point=d['to_fixed_point'] if 'to_fixed_point' in d else DEFAULT_FIXED_POINT,
            source_node_match=maybe_filter_dict_from_json(d, 'source_node_match'),
            destination_node_match=maybe_filter_dict_from_json(d, 'destination_node_match'),
            source_node_query=d['source_node_query'] if 'source_node_query' in d else None,
            destination_node_query=d['destination_node_query'] if 'destination_node_query' in d else None,
            edge_query=d['edge_query'] if 'edge_query' in d else None,
            name=d['name'] if 'name' in d else None
        )
        if validate:
            out.validate()
        return out

    def __call__(
        self,
        g: Plottable,
        prev_node_wavefront: Optional[DataFrameT],
        target_wave_front: Optional[DataFrameT],
        engine: Engine
    ) -> Plottable:

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
            hops=self.hops,
            to_fixed_point=self.to_fixed_point,
            direction=self.direction,
            source_node_match=self.source_node_match,
            edge_match=self.edge_match,
            destination_node_match=self.destination_node_match,
            return_as_wave_front=True,
            target_wave_front=target_wave_front,
            source_node_query=self.source_node_query,
            destination_node_query=self.destination_node_query,
            edge_query=self.edge_query
        )

        if self._name is not None:
            out_g = out_g.edges(out_g._edges.assign(**{self._name: True}))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('/CALL EDGE END {%s} ===>\nnodes:\n%s\nedges:\n%s\n', self, out_g._nodes, out_g._edges)
            logger.debug('----------------------------------------')

        return out_g

    def reverse(self) -> 'ASTEdge':
        # updates both edges and nodes
        direction : Direction
        if self.direction == 'reverse':
            direction = 'forward'
        elif self.direction == 'forward':
            direction = 'reverse'
        else:
            direction = 'undirected'
        return ASTEdge(
            direction=direction,
            edge_match=self.edge_match,
            hops=self.hops,
            to_fixed_point=self.to_fixed_point,
            source_node_match=self.destination_node_match,
            destination_node_match=self.source_node_match,
            source_node_query=self.destination_node_query,
            destination_node_query=self.source_node_query,
            edge_query=self.edge_query
        )

class ASTEdgeForward(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """

    def __init__(
        self,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
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
            edge_query=edge_query,
        )

    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTEdge':
        out = ASTEdgeForward(
            edge_match=maybe_filter_dict_from_json(d, 'edge_match'),
            hops=d['hops'] if 'hops' in d else None,
            to_fixed_point=d['to_fixed_point'] if 'to_fixed_point' in d else DEFAULT_FIXED_POINT,
            source_node_match=maybe_filter_dict_from_json(d, 'source_node_match'),
            destination_node_match=maybe_filter_dict_from_json(d, 'destination_node_match'),
            source_node_query=d['source_node_query'] if 'source_node_query' in d else None,
            destination_node_query=d['destination_node_query'] if 'destination_node_query' in d else None,
            edge_query=d['edge_query'] if 'edge_query' in d else None,
            name=d['name'] if 'name' in d else None
        )
        if validate:
            out.validate()
        return out


e_forward = ASTEdgeForward  # noqa: E305

class ASTEdgeReverse(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """

    def __init__(
        self,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
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
            edge_query=edge_query,
        )

    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTEdge':
        out = ASTEdgeReverse(
            edge_match=maybe_filter_dict_from_json(d, 'edge_match'),
            hops=d['hops'] if 'hops' in d else None,
            to_fixed_point=d['to_fixed_point'] if 'to_fixed_point' in d else DEFAULT_FIXED_POINT,
            source_node_match=maybe_filter_dict_from_json(d, 'source_node_match'),
            destination_node_match=maybe_filter_dict_from_json(d, 'destination_node_match'),
            source_node_query=d['source_node_query'] if 'source_node_query' in d else None,
            destination_node_query=d['destination_node_query'] if 'destination_node_query' in d else None,
            edge_query=d['edge_query'] if 'edge_query' in d else None,
            name=d['name'] if 'name' in d else None
        )
        if validate:
            out.validate()
        return out


e_reverse = ASTEdgeReverse  # noqa: E305

class ASTEdgeUndirected(ASTEdge):
    """
    Internal, not intended for use outside of this module.
    """

    def __init__(
        self,
        edge_match: Optional[dict] = DEFAULT_FILTER_DICT,
        hops: Optional[int] = DEFAULT_HOPS,
        source_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        destination_node_match: Optional[dict] = DEFAULT_FILTER_DICT,
        to_fixed_point: bool = DEFAULT_FIXED_POINT,
        name: Optional[str] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
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
            edge_query=edge_query,
        )

    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTEdge':
        out = ASTEdgeUndirected(
            edge_match=maybe_filter_dict_from_json(d, 'edge_match'),
            hops=d['hops'] if 'hops' in d else None,
            to_fixed_point=d['to_fixed_point'] if 'to_fixed_point' in d else DEFAULT_FIXED_POINT,
            source_node_match=maybe_filter_dict_from_json(d, 'source_node_match'),
            destination_node_match=maybe_filter_dict_from_json(d, 'destination_node_match'),
            source_node_query=d['source_node_query'] if 'source_node_query' in d else None,
            destination_node_query=d['destination_node_query'] if 'destination_node_query' in d else None,
            edge_query=d['edge_query'] if 'edge_query' in d else None,
            name=d['name'] if 'name' in d else None
        )
        if validate:
            out.validate()
        return out


e_undirected = ASTEdgeUndirected  # noqa: E305
e = ASTEdgeUndirected  # noqa: E305

###

def from_json(o: JSONVal, validate: bool = True) -> Union[ASTNode, ASTEdge]:
    from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError

    if not isinstance(o, dict):
        raise GFQLSyntaxError(ErrorCode.E101, "AST JSON must be a dictionary", value=type(o).__name__)

    if 'type' not in o:
        raise GFQLSyntaxError(
            ErrorCode.E105, "AST JSON missing required 'type' field", suggestion="Add 'type' field: 'Node' or 'Edge'"
        )

    out: Union[ASTNode, ASTEdge]
    if o['type'] == 'Node':
        out = ASTNode.from_json(o, validate=validate)
    elif o['type'] == 'Edge':
        if 'direction' in o:
            if o['direction'] == 'forward':
                out = ASTEdgeForward.from_json(o, validate=validate)
            elif o['direction'] == 'reverse':
                out = ASTEdgeReverse.from_json(o, validate=validate)
            elif o['direction'] == 'undirected':
                out = ASTEdgeUndirected.from_json(o, validate=validate)
            else:
                raise GFQLSyntaxError(
                    ErrorCode.E104,
                    f"Edge has unknown direction: {o['direction']}",
                    field="direction",
                    value=o['direction'],
                    suggestion='Use "forward", "reverse", or "undirected"',
                )
        else:
            raise GFQLSyntaxError(
                ErrorCode.E105,
                "Edge missing required 'direction' field",
                suggestion="Add 'direction' field: 'forward', 'reverse', or 'undirected'",
            )
    else:
        raise GFQLSyntaxError(
            ErrorCode.E101,
            f"Unknown AST type: {o['type']}",
            field="type",
            value=o["type"],
            suggestion="Use 'Node' or 'Edge'",
        )
    return out
