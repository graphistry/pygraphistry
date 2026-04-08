"""
Logical type layer for the GFQL compiler IR.

These types represent the *semantic* kinds of Cypher variables and pattern
elements — distinct from the AST representation in cypher/ast.py.

All types are frozen dataclasses so they are hashable and safe to use as
dict keys / set members.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Union

from graphistry.compute.gfql.cypher.ast import (
    CypherGraphQuery,
    CypherQuery,
    CypherUnionQuery,
)

CypherAST = Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]


@dataclass(frozen=True)
class NodeRef:
    """A node variable binding.

    ``labels`` is the *conjunction* of required labels (all must match).
    An empty frozenset means unconstrained — matches any node.
    Supports ``(n:Person:Admin)`` as ``NodeRef(labels=frozenset({'Person', 'Admin'}))``.
    """

    labels: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EdgeRef:
    """An edge variable binding (direction is NOT stored here — see RelSpec).

    ``types`` is the disjunction of allowed relationship types (any may match).
    An empty frozenset means unconstrained.
    """

    types: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RelSpec:
    """Pattern-level relationship specification — carries direction.

    Unlike ``EdgeRef`` (which represents a *bound variable*), ``RelSpec``
    describes the structural pattern constraint including direction.
    """

    types: FrozenSet[str] = field(default_factory=frozenset)
    direction: Literal["forward", "reverse", "undirected"] = "undirected"


@dataclass(frozen=True)
class ScalarType:
    """A scalar (literal or computed) variable binding."""

    pass


@dataclass(frozen=True)
class PathType:
    """A path variable binding (from shortestPath / allShortestPaths)."""

    pass


# The universe of logical types a Cypher variable can have.
LogicalType = Union[NodeRef, EdgeRef, ScalarType, PathType]


@dataclass(frozen=True)
class QueryGraph:
    """Placeholder for the pattern graph built during binding.

    Full shape is defined in a later compiler stage.
    """

    pass
