"""GFQL reserved identifiers and validation."""

import re
from typing import Optional, Dict, Any, Final, FrozenSet, Set

from graphistry.compute.util.generate_safe_column_name import generate_safe_column_name_from

#: The openCypher identifier form, shared by every surface that has to recognize one.
IDENTIFIER_TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def is_bare_identifier(text: str) -> bool:
    """The text is exactly one identifier: no dots, calls, operators or whitespace."""
    return IDENTIFIER_TOKEN.fullmatch(text) is not None


def identifier_tokens(text: str) -> Set[str]:
    """Every identifier-shaped token in an expression, ignoring where it appears."""
    return set(IDENTIFIER_TOKEN.findall(text))


# Internal column pattern for temporary GFQL columns
INTERNAL_COLUMN_PATTERN: Final[str] = '__gfql_*__'
INTERNAL_COLUMN_PREFIX: Final[str] = '__gfql_'
INTERNAL_COLUMN_SUFFIX: Final[str] = '__'

# BASE names only: render through generate_safe_column_name(_from) against the user's frame.
ROW_EDGE_IDENTITY_BASE: str = 'edge_ident'
EDGE_INDEX_BASE: str = 'edge_index'

#: The collision-free rendering against an empty frame, for callers that build their own
#: pair frames; every caller holding real edge columns must resolve against those instead.
DEFAULT_ROW_EDGE_IDENTITY_COL: Final[str] = generate_safe_column_name_from(
    ROW_EDGE_IDENTITY_BASE, ()
)

#: Prefix of the internal columns that carry an alias's value under a hidden name.
HIDDEN_ALIAS_COLUMN_PREFIX: Final[str] = '__gfql_hidden_'

#: Prefix of the internal name given to an otherwise-anonymous connected-join arm edge.
TRAIL_ARM_EDGE_ALIAS_PREFIX: Final[str] = '__gfql_trail_arm_'

#: Source endpoint of an ORIENTED edge row (``EdgeSemantics.orient_edges`` output).
WALK_FROM_COL: Final[str] = '__from__'

#: Destination endpoint of an ORIENTED edge row.
WALK_TO_COL: Final[str] = '__to__'

#: The node a row of the path bag currently stands on.
WALK_CURRENT_COL: Final[str] = '__current__'

#: The node a path-bag row just left, so a hop can drop an immediate backtrack.
WALK_PREV_COL: Final[str] = '__gfql_prev__'

#: Stable per-edge identity: openCypher TRAIL semantics bind a relationship at most once per path.
TRAIL_EDGE_IDENT_COL: Final[str] = '__gfql_edge_ident__'

#: Prefix of the per-hop column recording WHICH relationship that hop bound.
TRAIL_COLUMN_PREFIX: Final[str] = '__gfql_trail_'

#: Prefix of the hidden hop-count column a ``shortestPath`` pattern binds for its path alias.
SHORTEST_PATH_HOPS_COLUMN_PREFIX: Final[str] = '__cypher_shortest_path_hops__'

#: Every scratch column the row-binding walk may introduce; none may survive into a result.
WALK_SCRATCH_COLUMNS: FrozenSet[str] = frozenset({
    WALK_FROM_COL, WALK_TO_COL, WALK_CURRENT_COL, WALK_PREV_COL, TRAIL_EDGE_IDENT_COL,
})


def trail_column_name(hop_index: int) -> str:
    """Name of the trail column recording the relationship bound at ``hop_index``."""
    return f'{TRAIL_COLUMN_PREFIX}{hop_index}{INTERNAL_COLUMN_SUFFIX}'


def is_trail_column(name: str) -> bool:
    """Whether ``name`` is a per-hop trail column produced by :func:`trail_column_name`."""
    return (isinstance(name, str)
            and name.startswith(TRAIL_COLUMN_PREFIX)
            and name.endswith(INTERNAL_COLUMN_SUFFIX)
            and name[len(TRAIL_COLUMN_PREFIX):-len(INTERNAL_COLUMN_SUFFIX)].isdigit())


def shortest_path_hops_column(alias: str) -> str:
    """Name of the hidden hop-count column bound by the ``shortestPath`` path alias ``alias``."""
    return f'{SHORTEST_PATH_HOPS_COLUMN_PREFIX}{alias}'


def is_shortest_path_hops_column(name: object) -> bool:
    """Whether ``name`` is a ``shortestPath`` hop-count column, i.e. selects shortestPath mode."""
    return isinstance(name, str) and name.startswith(SHORTEST_PATH_HOPS_COLUMN_PREFIX)


def is_walk_scratch_column(name: str) -> bool:
    """Whether ``name`` is any walk scratch column (fixed vocabulary or a trail column)."""
    return name in WALK_SCRATCH_COLUMNS or is_trail_column(name)


def is_internal_column(name: str) -> bool:
    """Check if name matches internal column pattern __gfql_*__."""
    return (isinstance(name, str)
            and name.startswith(INTERNAL_COLUMN_PREFIX)
            and name.endswith(INTERNAL_COLUMN_SUFFIX)
            and len(name) > len(INTERNAL_COLUMN_PREFIX) + len(INTERNAL_COLUMN_SUFFIX))


def validate_column_name(name: str, context: str = "Column") -> None:
    """Validate output column name doesn't use internal pattern.

    Used for operation output parameters like get_degrees(col='...').
    """
    if is_internal_column(name):
        raise ValueError(
            f"{context} cannot use column name '{name}'. "
            f"Pattern '{INTERNAL_COLUMN_PATTERN}' is reserved for internal use. "
            f"Choose a different name."
        )


def validate_column_references(
    col_dict: Optional[Dict[str, Any]],
    context: str = "Operation"
) -> None:
    """Validate dict keys don't reference internal columns.

    Internal columns are temporary and won't work with gfql_remote().
    """
    if not col_dict:
        return

    for key in col_dict.keys():
        if is_internal_column(key):
            raise ValueError(
                f"{context} cannot use column '{key}'. "
                f"Pattern '{INTERNAL_COLUMN_PATTERN}' is reserved for internal use. "
                f"These columns are temporary and won't work with gfql_remote()."
            )


# Legacy reserved names (server-side only, not enforced client-side)
RESERVED_COLUMN_NAMES_LEGACY: Set[str] = {'id'}

# Recommended to avoid (may cause confusion in some contexts)
RECOMMENDED_AVOID: Set[str] = {
    'node', 'edge', 'graph',
    'id', 'idx', 'index',
    'src', 'source', 'from',
    'dst', 'dest', 'destination', 'to', 'target',
    'type', 'label', 'name'
}
