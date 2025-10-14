from abc import abstractmethod
import logging
from typing import (
    Any, TYPE_CHECKING, Dict, List, Optional, Sequence, Union, cast
)
from typing_extensions import Literal

if TYPE_CHECKING:
    from graphistry.compute.chain import Chain

from graphistry.Engine import Engine, EngineAbstract

from graphistry.Plottable import Plottable
from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLSyntaxError
from graphistry.compute.gfql.identifiers import validate_column_references
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
    fullmatch, Fullmatch,
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

            # Validate that filter_dict doesn't reference internal columns
            validate_column_references(self.filter_dict, "n()")

        # Validate name
        if self._name is not None and not isinstance(self._name, str):
            raise GFQLTypeError(ErrorCode.E204, "name must be a string", field="name", value=type(self._name).__name__)

        # Validate query
        if self.query is not None and not isinstance(self.query, str):
            raise GFQLTypeError(
                ErrorCode.E205, "query must be a string", field="query", value=type(self.query).__name__
            )

    def _get_child_validators(self) -> Sequence['ASTSerializable']:
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

        # Validate that filter dicts don't reference internal columns
        validate_column_references(self.source_node_match, f"e_{self.direction}() source_node_match")
        validate_column_references(self.edge_match, f"e_{self.direction}() edge_match")
        validate_column_references(self.destination_node_match, f"e_{self.direction}() destination_node_match")

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

    def _get_child_validators(self) -> Sequence['ASTSerializable']:
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


##############################################################################


class ASTLet(ASTObject):
    """Let-bindings for named graph operations in a DAG.
    
    Allows defining reusable graph operations that can reference each other,
    forming a directed acyclic graph (DAG) of computations.
    
    :param bindings: Dictionary mapping names to graph operations
    :type bindings: Dict[str, Union[ASTObject, Chain, Plottable]]
    
    :raises GFQLTypeError: If bindings is not a dict or contains invalid keys/values
    
    **Example::**

        # Matchers now supported directly (operate on root graph)
        dag = ASTLet({
            'persons': n({'type': 'person'}),
            'friends': ASTRef('persons', [e_forward({'rel': 'friend'})])
        })
    """
    bindings: Dict[str, Union['ASTObject', 'Chain', Plottable]]
    
    def __init__(self, bindings: Dict[str, Union['ASTObject', 'Chain', Plottable, Dict[str, Any]]], validate: bool = True) -> None:
        """Initialize Let with named bindings.

        :param bindings: Dictionary mapping names to GraphOperation instances or JSON dicts.
                        JSON dicts must have a 'type' field indicating the AST object type.
        :type bindings: Dict[str, Union[ASTObject, Chain, Plottable, Dict[str, Any]]]
        :param validate: Whether to validate the bindings immediately
        :type validate: bool
        """
        super().__init__()
        
        # Process mixed JSON/native objects
        processed_bindings: Dict[str, Any] = {}
        for name, value in bindings.items():
            if isinstance(value, dict):
                # JSON dict - check type and convert if valid
                if 'type' not in value:
                    raise ValueError(f"JSON binding '{name}' missing 'type' field")
                
                obj_type = value.get('type')
                # Check if it's a valid GraphOperation type
                if obj_type == 'Chain':
                    # Import Chain here due to circular dependency
                    from graphistry.compute.chain import Chain  # noqa: F401, F811
                    chain_obj = Chain.from_json(value, validate=False)
                    processed_bindings[name] = chain_obj  # type: ignore
                else:
                    # Convert other AST types
                    ast_obj = from_json(value, validate=False)
                    processed_bindings[name] = ast_obj  # type: ignore
            else:
                # Native object - use as-is
                processed_bindings[name] = value
        
        self.bindings = processed_bindings  # type: ignore
        
        if validate:
            self.validate()
    
    def _validate_fields(self) -> None:
        """Validate Let fields."""
        if not isinstance(self.bindings, dict):
            raise GFQLTypeError(
                ErrorCode.E201,
                "bindings must be a dictionary",
                field="bindings",
                value=type(self.bindings).__name__
            )
        
        for k, v in self.bindings.items():
            if not isinstance(k, str):
                raise GFQLTypeError(
                    ErrorCode.E102,
                    "binding key must be string",
                    field=f"bindings.{k}",
                    value=type(k).__name__
                )
            # Check if value is a valid GraphOperation type
            # Import Chain here due to circular dependency
            from graphistry.compute.chain import Chain as ChainClass  # noqa: F401

            # GraphOperation now includes all AST types
            # ASTNode/ASTEdge are now allowed and will operate on the root graph
            if not isinstance(v, (ASTNode, ASTEdge, ASTRef, ASTCall, ASTRemoteGraph, ASTLet, Plottable, ChainClass)):
                raise GFQLTypeError(
                    ErrorCode.E201,
                    "binding value must be a valid operation (ASTNode, ASTEdge, Chain, ASTRef, ASTCall, ASTRemoteGraph, ASTLet, or Plottable)",
                    field=f"bindings.{k}",
                    value=type(v).__name__,
                    suggestion="Use a valid graph operation or matcher"
                )
        # TODO: Check for cycles in DAG
        return None
    
    def _get_child_validators(self) -> Sequence['ASTSerializable']:
        """Return child AST nodes that need validation."""
        # Only return objects that inherit from ASTSerializable
        # Plottable instances don't need validation
        children = []
        for v in self.bindings.values():
            if isinstance(v, ASTSerializable):
                children.append(v)
        return children
    
    def to_json(self, validate: bool = True) -> dict:
        """Convert Let to JSON representation.
        
        :param validate: Whether to validate before serialization
        :type validate: bool
        :returns: JSON-serializable dictionary
        :rtype: dict
        """
        if validate:
            self.validate()
        bindings_json = {}
        for k, v in self.bindings.items():
            if hasattr(v, 'to_json'):
                bindings_json[k] = v.to_json()  # type: ignore
            else:
                # Plottable doesn't have to_json
                raise ValueError(f"Cannot serialize {type(v).__name__} to JSON")
        return {
            'type': 'Let',
            'bindings': bindings_json
        }
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTLet':
        """Create ASTLet from JSON representation.
        
        :param d: JSON dictionary with 'bindings' field
        :type d: dict
        :param validate: Whether to validate after creation
        :type validate: bool
        :returns: New ASTLet instance
        :rtype: ASTLet
        :raises AssertionError: If 'bindings' field is missing
        """
        assert 'bindings' in d, "Let missing bindings"

        # Import Chain here due to circular dependency
        from graphistry.compute.chain import Chain  # noqa: F401, F811

        bindings: Dict[str, Any] = {}
        for k, v in d['bindings'].items():
            # Handle Chain objects specially
            if isinstance(v, dict) and v.get('type') == 'Chain':
                bindings[k] = Chain.from_json(v, validate=validate)
            else:
                # Regular AST objects
                bindings[k] = from_json(v, validate=validate)
        
        out = cls(bindings=bindings, validate=validate)  # type: ignore
        return out
    
    def __call__(self, g: Plottable, prev_node_wavefront: Optional[DataFrameT],
                 target_wave_front: Optional[DataFrameT], engine: Engine) -> Plottable:
        # Let bindings don't use wavefronts - execute via chain_let_impl
        # Import here due to circular dependency
        from graphistry.compute.chain_let import chain_let_impl  # noqa: F401, F811
        return chain_let_impl(g, self, EngineAbstract(engine.value))
    
    def reverse(self) -> 'ASTLet':
        raise NotImplementedError("Let reversal not supported")


class ASTRemoteGraph(ASTObject):
    """Load a graph from Graphistry server.
    
    Allows fetching previously uploaded graphs by dataset ID,
    optionally with an authentication token.
    
    :param dataset_id: Unique identifier of the dataset on the server
    :type dataset_id: str
    :param token: Optional authentication token
    :type token: Optional[str]
    
    :raises GFQLTypeError: If dataset_id is not a string or is empty
    
    **Example::**
    
        # Fetch public dataset
        remote = ASTRemoteGraph('my-dataset-id')
        
        # Fetch private dataset with token
        remote = ASTRemoteGraph('private-dataset', token='auth-token')
    """
    def __init__(self, dataset_id: str, token: Optional[str] = None) -> None:
        """Initialize RemoteGraph with dataset ID and optional token.
        
        :param dataset_id: Unique identifier of the dataset
        :type dataset_id: str
        :param token: Optional authentication token
        :type token: Optional[str]
        """
        super().__init__()
        self.dataset_id = dataset_id
        self.token = token
    
    def _validate_fields(self) -> None:
        """Validate RemoteGraph fields."""
        if not isinstance(self.dataset_id, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "dataset_id must be a string",
                field="dataset_id",
                value=type(self.dataset_id).__name__
            )
        
        if len(self.dataset_id) == 0:
            raise GFQLTypeError(
                ErrorCode.E106,
                "dataset_id cannot be empty",
                field="dataset_id",
                value=self.dataset_id
            )
        
        if self.token is not None and not isinstance(self.token, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "token must be string or None",
                field="token",
                value=type(self.token).__name__
            )
    
    def to_json(self, validate: bool = True) -> dict:
        """Convert RemoteGraph to JSON representation.
        
        :param validate: Whether to validate before serialization
        :type validate: bool
        :returns: JSON-serializable dictionary
        :rtype: dict
        """
        if validate:
            self.validate()
        result = {
            'type': 'RemoteGraph',
            'dataset_id': self.dataset_id
        }
        if self.token is not None:
            result['token'] = self.token
        return result
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTRemoteGraph':
        """Create ASTRemoteGraph from JSON representation.
        
        :param d: JSON dictionary with 'dataset_id' field
        :type d: dict
        :param validate: Whether to validate after creation
        :type validate: bool
        :returns: New ASTRemoteGraph instance
        :rtype: ASTRemoteGraph
        :raises AssertionError: If 'dataset_id' field is missing
        """
        assert 'dataset_id' in d, "RemoteGraph missing dataset_id"
        out = cls(
            dataset_id=d['dataset_id'],
            token=d.get('token')
        )
        if validate:
            out.validate()
        return out
    
    def __call__(self, g: Plottable, prev_node_wavefront: Optional[DataFrameT],
                 target_wave_front: Optional[DataFrameT], engine: Engine) -> Plottable:
        # Implementation in PR 1.3
        raise NotImplementedError("RemoteGraph loading will be implemented in PR 1.3")
    
    def reverse(self) -> 'ASTRemoteGraph':
        raise NotImplementedError("RemoteGraph reversal not supported")


class ASTRef(ASTObject):
    """Execute a chain of operations starting from a DAG binding reference.
    
    Allows building graph operations that start from a named binding
    defined in an ASTLet (DAG) and apply additional operations.
    
    :param ref: Name of the binding to reference from the DAG
    :type ref: str
    :param chain: List of operations to apply to the referenced graph
    :type chain: List[ASTObject]
    
    :raises GFQLTypeError: If ref is not a string or chain is not a list
    
    **Example::**
    
        # Reference 'persons' binding and find their friends
        friends = ASTRef('persons', [e_forward({'rel': 'friend'})])
    """
    def __init__(self, ref: str, chain: List['ASTObject']) -> None:
        """Initialize Ref with reference name and operation chain.
        
        :param ref: Name of the binding to reference
        :type ref: str
        :param chain: List of operations to apply
        :type chain: List[ASTObject]
        """
        super().__init__()
        self.ref = ref
        self.chain = chain
    
    def _validate_fields(self) -> None:
        """Validate Ref fields."""
        if not isinstance(self.ref, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "ref must be a string",
                field="ref",
                value=type(self.ref).__name__
            )
        
        if len(self.ref) == 0:
            raise GFQLTypeError(
                ErrorCode.E106,
                "ref cannot be empty",
                field="ref",
                value=self.ref
            )
        
        if not isinstance(self.chain, list):
            raise GFQLTypeError(
                ErrorCode.E201,
                "chain must be a list",
                field="chain",
                value=type(self.chain).__name__
            )
        
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                raise GFQLTypeError(
                    ErrorCode.E201,
                    f"chain[{i}] must be ASTObject",
                    field=f"chain[{i}]",
                    value=type(op).__name__
                )
    
    def _get_child_validators(self) -> Sequence['ASTSerializable']:
        """Return child AST nodes that need validation."""
        # ASTObject inherits from ASTSerializable, so this is safe
        return self.chain
    
    def to_json(self, validate: bool = True) -> dict:
        """Convert Ref to JSON representation.
        
        :param validate: Whether to validate before serialization
        :type validate: bool
        :returns: JSON-serializable dictionary
        :rtype: dict
        """
        if validate:
            self.validate()
        return {
            'type': 'Ref',
            'ref': self.ref,
            'chain': [op.to_json() for op in self.chain]
        }
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTRef':
        """Create ASTRef from JSON representation.
        
        :param d: JSON dictionary with 'ref' and 'chain' fields
        :type d: dict
        :param validate: Whether to validate after creation
        :type validate: bool
        :returns: New ASTRef instance
        :rtype: ASTRef
        :raises AssertionError: If 'ref' or 'chain' fields are missing
        """
        assert 'ref' in d, "Ref missing ref"
        assert 'chain' in d, "Ref missing chain"
        out = cls(
            ref=d['ref'],
            chain=[from_json(op, validate=validate) for op in d['chain']]
        )
        if validate:
            out.validate()
        return out
    
    def __call__(self, g: Plottable, prev_node_wavefront: Optional[DataFrameT],
                 target_wave_front: Optional[DataFrameT], engine: Engine) -> Plottable:
        raise NotImplementedError(
            "ASTRef cannot be used directly in chain(). "
            "It must be used within an ASTLet/chain_let() context."
        )
    
    def reverse(self) -> 'ASTRef':
        # Reverse the chain operations
        return ASTRef(self.ref, [op.reverse() for op in reversed(self.chain)])


class ASTCall(ASTObject):
    """Call a method on the current graph with validated parameters.
    
    Allows safe execution of Plottable methods through GFQL with parameter
    validation and schema checking.
    
    Attributes:
        function: Name of the method to call (must be in safelist)
        params: Dictionary of parameters to pass to the method
    """
    def __init__(self, function: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a Call operation.

        Args:
            function: Name of the Plottable method to call
            params: Optional dictionary of parameters for the method
        """
        super().__init__()
        self.function = function
        self.params = params or {}

    def _validate_fields(self) -> None:
        """Validate Call fields."""
        if not isinstance(self.function, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "function must be a string",
                field="function",
                value=type(self.function).__name__
            )
        
        if len(self.function) == 0:
            raise GFQLTypeError(
                ErrorCode.E106,
                "function name cannot be empty",
                field="function",
                value=self.function
            )
        
        if not isinstance(self.params, dict):
            raise GFQLTypeError(
                ErrorCode.E201,
                "params must be a dictionary",
                field="params",
                value=type(self.params).__name__
            )

        # Validate filter_*_by_dict calls for internal column references
        if self.function in ('filter_nodes_by_dict', 'filter_edges_by_dict'):
            # For these functions, the filter_dict is passed as a parameter
            if 'filter_dict' in self.params:
                validate_column_references(
                    self.params.get('filter_dict'),
                    f"call('{self.function}')"
                )

        # Validate output column name parameters to prevent __gfql_*__ internal column conflicts
        from graphistry.compute.gfql.identifiers import validate_column_name

        # Map function names to their output column parameter names
        output_col_params = {
            'get_degrees': ['col', 'degree_in', 'degree_out'],
            'get_indegrees': ['col'],
            'get_outdegrees': ['col'],
            'get_topological_levels': ['level_col'],
            'compute_cugraph': ['out_col'],
            'compute_igraph': ['out_col'],
            'encode_point_color': ['column'],
            'encode_edge_color': ['column'],
            'encode_point_size': ['column'],
            'encode_point_icon': ['column'],
            'layout_igraph': ['x_out_col', 'y_out_col'],
            'layout_cugraph': ['x_out_col', 'y_out_col'],
            'layout_graphviz': ['x_out_col', 'y_out_col'],
            'collapse': ['column'],
        }

        if self.function in output_col_params:
            for param in output_col_params[self.function]:
                if param in self.params:
                    validate_column_name(self.params[param], f"call('{self.function}') {param} parameter")

    def to_json(self, validate: bool = True) -> dict:
        """Convert Call to JSON representation.
        
        Args:
            validate: If True, validate before serialization
            
        Returns:
            Dictionary with type, function, and params fields
        """
        if validate:
            self.validate()
        return {
            'type': 'Call',
            'function': self.function,
            'params': self.params
        }
    
    @classmethod
    def from_json(cls, d: dict, validate: bool = True) -> 'ASTCall':
        """Create ASTCall from JSON representation.
        
        :param d: JSON dictionary with 'function' field and optional 'params'
        :type d: dict
        :param validate: Whether to validate after creation
        :type validate: bool
        :returns: New ASTCall instance
        :rtype: ASTCall
        :raises AssertionError: If 'function' field is missing
        
        **Example::**
        
            call_json = {'type': 'Call', 'function': 'hop', 'params': {'steps': 2}}
            call = ASTCall.from_json(call_json)
        """
        assert 'function' in d, "Call missing function"
        out = cls(
            function=d['function'],
            params=d.get('params', {})
        )
        if validate:
            out.validate()
        return out
    
    def __call__(self, g: Plottable, prev_node_wavefront: Optional[DataFrameT],
                 target_wave_front: Optional[DataFrameT], engine: Engine) -> Plottable:
        """Execute the method call on the graph.
        
        Args:
            g: Graph to operate on
            prev_node_wavefront: Previous node wavefront (unused)
            target_wave_front: Target wavefront (unused)
            engine: Execution engine (pandas/cudf)
            
        Returns:
            New Plottable with method results
            
        Raises:
            GFQLTypeError: If method not in safelist or parameters invalid
        """
        # For chain_let, we don't use wavefronts, just execute the call
        # Import here due to circular dependency
        from graphistry.compute.gfql.call_executor import execute_call  # noqa: F401, F811
        return execute_call(g, self.function, self.params, engine)
    
    def reverse(self) -> 'ASTCall':
        # Method calls are transformations that don't need reversal
        # Return self to act as identity in the reverse pass
        return self


###

def from_json(o: JSONVal, validate: bool = True) -> Union[ASTNode, ASTEdge, ASTLet, ASTRemoteGraph, ASTRef, ASTCall]:
    if not isinstance(o, dict):
        raise GFQLSyntaxError(ErrorCode.E101, "AST JSON must be a dictionary", value=type(o).__name__)

    if 'type' not in o:
        raise GFQLSyntaxError(
            ErrorCode.E105, "AST JSON missing required 'type' field", suggestion="Add 'type' field: 'Node', 'Edge', 'Let', 'RemoteGraph', or 'ChainRef'"
        )

    out: Union[ASTNode, ASTEdge, ASTLet, ASTRemoteGraph, ASTRef, ASTCall]
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
    elif o['type'] == 'Let':
        out = ASTLet.from_json(o, validate=validate)
    elif o['type'] == 'QueryDAG':
        # For backward compatibility
        out = ASTLet.from_json(o, validate=validate)
    elif o['type'] == 'RemoteGraph':
        out = ASTRemoteGraph.from_json(o, validate=validate)
    elif o['type'] == 'ChainRef':
        out = ASTRef.from_json(o, validate=validate)
    elif o['type'] == 'Ref':
        out = ASTRef.from_json(o, validate=validate)
    elif o['type'] == 'Call':
        out = ASTCall.from_json(o, validate=validate)
    else:
        raise GFQLSyntaxError(
            ErrorCode.E101,
            f"Unknown AST type: {o['type']}",
            field="type",
            value=o["type"],
            suggestion="Use 'Node', 'Edge', 'Let', 'RemoteGraph', 'ChainRef', 'Ref', or 'Call'",
        )
    return out


###############################################################################
# User-friendly aliases for public API

let = ASTLet  # noqa: E305
remote = ASTRemoteGraph  # noqa: E305
ref = ASTRef  # noqa: E305

# Import type-safe call() function from models/gfql/types/call
# This provides overloaded signatures for IDE autocomplete and type checking
from graphistry.models.gfql.types.call import call  # noqa: E305 E402 F401
