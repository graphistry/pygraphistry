"""Entry guards for circle_layout: edge-only graphs, missing positions, duplicate
bounding_box partition keys (#1968)."""

import pandas as pd
import pytest

import graphistry


def test_edge_only_graph_materializes_nodes() -> None:
    e = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
    g = graphistry.edges(e, 's', 'd').circle_layout(bounding_box=(0, 0, 10, 10))
    assert len(g._nodes) == 3
    assert 'x' in g._nodes.columns and 'y' in g._nodes.columns
    assert not g._nodes['x'].isna().any()


def test_no_bounding_box_without_positions_raises_value_error() -> None:
    e = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
    n = pd.DataFrame({'id': [0, 1, 2]})
    g = graphistry.edges(e, 's', 'd').nodes(n, 'id')
    with pytest.raises(ValueError, match='bounding_box'):
        g.circle_layout()


def test_no_bounding_box_with_positions_still_works() -> None:
    e = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
    n = pd.DataFrame({'id': [0, 1, 2], 'x': [0., 1., 2.], 'y': [0., 1., 2.]})
    g = graphistry.edges(e, 's', 'd').nodes(n, 'id').circle_layout()
    assert len(g._nodes) == 3


def test_duplicate_bounding_box_partition_keys_raise_value_error() -> None:
    e = pd.DataFrame({'s': [0, 1, 2, 3], 'd': [1, 2, 0, 0]})
    n = pd.DataFrame({'id': [0, 1, 2, 3], 'p': ['a', 'a', 'b', 'b']})
    bb = pd.DataFrame({
        'partition_key': ['a', 'a', 'b', 'b'],
        'cx': [0., 5., 100., 105.],
        'cy': [0., 0., 100., 100.],
        'w': [10.] * 4,
        'h': [10.] * 4,
    })
    g = graphistry.edges(e, 's', 'd').nodes(n, 'id')
    with pytest.raises(ValueError, match='partition_key'):
        g.circle_layout(partition_by='p', bounding_box=bb)


def test_unique_bounding_box_partition_keys_still_work() -> None:
    e = pd.DataFrame({'s': [0, 1, 2, 3], 'd': [1, 2, 0, 0]})
    n = pd.DataFrame({'id': [0, 1, 2, 3], 'p': ['a', 'a', 'b', 'b']})
    bb = pd.DataFrame({
        'partition_key': ['a', 'b'],
        'cx': [0., 100.],
        'cy': [0., 100.],
        'w': [10., 10.],
        'h': [10., 10.],
    })
    g = graphistry.edges(e, 's', 'd').nodes(n, 'id').circle_layout(partition_by='p', bounding_box=bb)
    assert len(g._nodes) == 4
