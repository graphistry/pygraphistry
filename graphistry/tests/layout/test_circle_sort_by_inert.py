"""Characterization pin: circle_layout's sort parameters do not affect positions.

The docstring on ``circle_layout`` states plainly that ``sort_by`` / ``ascending`` /
``na_position`` / ``ignore_index`` have no effect on the layout. That is a consequence of
the unconditional re-sort by node id which fixes ring order before any angle is assigned.
These tests lock the documented behavior so a future change that makes ``sort_by`` real
must update the docs in the same commit.
"""

from typing import Any, Dict, List

import pandas as pd
import pytest

import graphistry


def _graph() -> Any:
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd', 'e'],
        'k': [5, 4, 3, 2, 1],
        'grp': ['x', 'x', 'y', 'y', 'y'],
    })
    edges = pd.DataFrame({'s': ['a', 'b', 'c', 'd'], 'd': ['b', 'c', 'd', 'e']})
    return graphistry.edges(edges, 's', 'd').nodes(nodes, 'id')


def _positions(g: Any) -> Dict[str, Any]:
    nodes = g._nodes
    return {
        row['id']: (round(float(row['x']), 9), round(float(row['y']), 9))
        for _, row in nodes.iterrows()
    }


_VARIANTS: List[Dict[str, Any]] = [
    {'sort_by': 'k'},
    {'sort_by': 'k', 'ascending': False},
    {'sort_by': ['k'], 'ascending': [False]},
    {'sort_by': 'k', 'na_position': 'first'},
    {'sort_by': 'k', 'ignore_index': False},
    {'sort_by': ['grp', 'k'], 'ascending': False},
]


@pytest.mark.parametrize('variant', _VARIANTS)
def test_sort_params_do_not_change_positions(variant: Dict[str, Any]) -> None:
    g = _graph()
    baseline = _positions(g.circle_layout(bounding_box=(0, 0, 10, 10)))
    got = _positions(g.circle_layout(bounding_box=(0, 0, 10, 10), **variant))
    assert got == baseline, f'{variant} changed positions; docs say it cannot'


# Under partition_by the discarded sort prepends the partition columns to `by`, so a
# list-valued `ascending` of the caller's own length is rejected by pandas before it is
# thrown away. Excluded here; the raising behavior is covered separately.
_PARTITIONED_VARIANTS = [v for v in _VARIANTS if not isinstance(v.get('ascending'), list)]


@pytest.mark.parametrize('variant', _PARTITIONED_VARIANTS)
def test_sort_params_do_not_change_positions_when_partitioned(
    variant: Dict[str, Any]
) -> None:
    g = _graph()
    bbox = pd.DataFrame({
        'grp': ['x', 'y'],
        'cx': [0.0, 20.0],
        'cy': [0.0, 0.0],
        'w': [10.0, 10.0],
        'h': [10.0, 10.0],
    })
    kwargs: Dict[str, Any] = {'bounding_box': bbox, 'partition_by': 'grp'}
    baseline = _positions(g.circle_layout(**kwargs))
    got = _positions(g.circle_layout(**kwargs, **variant))
    assert got == baseline, f'{variant} changed positions; docs say it cannot'


def test_ring_order_is_by_node_id() -> None:
    """The documented ordering rule: position follows node id, not any sort key."""
    g = _graph()
    # Reversing the node table must not move any node.
    g_rev = g.nodes(g._nodes.iloc[::-1].reset_index(drop=True), 'id')
    assert _positions(g.circle_layout(bounding_box=(0, 0, 10, 10))) == _positions(
        g_rev.circle_layout(bounding_box=(0, 0, 10, 10))
    )


def test_sort_by_none_attaches_degree_columns() -> None:
    """The one documented residual effect of sort_by: degree columns on the output."""
    g = _graph()
    default_cols = set(g.circle_layout(bounding_box=(0, 0, 10, 10))._nodes.columns)
    sorted_cols = set(
        g.circle_layout(bounding_box=(0, 0, 10, 10), sort_by='k')._nodes.columns
    )
    assert {'degree', 'degree_in', 'degree_out'} <= default_cols
    assert not ({'degree', 'degree_in', 'degree_out'} & sorted_cols)


def test_unknown_sort_by_still_raises() -> None:
    """Also documented: the discarded sort still validates its column."""
    g = _graph()
    with pytest.raises(KeyError):
        g.circle_layout(bounding_box=(0, 0, 10, 10), sort_by='nope')
