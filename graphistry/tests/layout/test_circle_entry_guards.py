"""Entry guards for circle_layout: edge-only graphs, missing positions, duplicate
bounding_box partition keys, null partition keys (#1968)."""

import os

import pandas as pd
import pytest

import graphistry

TRIANGLE_EDGES = pd.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
PARTITIONED_EDGES = pd.DataFrame({'s': [0, 1, 2, 3], 'd': [1, 2, 0, 0]})
PARTITIONED_NODES = pd.DataFrame({'id': [0, 1, 2, 3], 'p': ['a', 'a', 'b', 'b']})
BOUNDING_BOX_AB = pd.DataFrame({
    'partition_key': ['a', 'b'],
    'cx': [0., 100.],
    'cy': [0., 100.],
    'w': [10., 10.],
    'h': [10., 10.],
})


def _partitioned_graph() -> graphistry.Plottable:
    return graphistry.edges(PARTITIONED_EDGES, 's', 'd').nodes(PARTITIONED_NODES, 'id')


def test_edge_only_graph_materializes_nodes() -> None:
    g = graphistry.edges(TRIANGLE_EDGES, 's', 'd').circle_layout(bounding_box=(0, 0, 10, 10))
    assert len(g._nodes) == 3
    assert 'x' in g._nodes.columns and 'y' in g._nodes.columns
    assert not g._nodes['x'].isna().any()


def test_nodes_without_node_binding_counts_the_materialized_nodes() -> None:
    n = pd.DataFrame({'id': [0, 1]})
    g = graphistry.edges(TRIANGLE_EDGES, 's', 'd').nodes(n).circle_layout(bounding_box=(0, 0, 10, 10))
    assert len(g._nodes) == 3
    assert not g._nodes['x'].isna().any()


def test_no_bounding_box_without_positions_raises_value_error() -> None:
    n = pd.DataFrame({'id': [0, 1, 2]})
    g = graphistry.edges(TRIANGLE_EDGES, 's', 'd').nodes(n, 'id')
    with pytest.raises(ValueError, match='bounding_box'):
        g.circle_layout()


def test_no_bounding_box_with_positions_still_works() -> None:
    n = pd.DataFrame({'id': [0, 1, 2], 'x': [0., 1., 2.], 'y': [0., 1., 2.]})
    g = graphistry.edges(TRIANGLE_EDGES, 's', 'd').nodes(n, 'id').circle_layout()
    assert len(g._nodes) == 3


def test_duplicate_bounding_box_partition_keys_raise_value_error() -> None:
    bb = pd.DataFrame({
        'partition_key': ['a', 'a', 'b', 'b'],
        'cx': [0., 5., 100., 105.],
        'cy': [0., 0., 100., 100.],
        'w': [10.] * 4,
        'h': [10.] * 4,
    })
    with pytest.raises(ValueError, match=r"duplicate partition_key values: \['a', 'b'\]"):
        _partitioned_graph().circle_layout(partition_by='p', bounding_box=bb)


def test_unique_bounding_box_partition_keys_still_work() -> None:
    g = _partitioned_graph().circle_layout(partition_by='p', bounding_box=BOUNDING_BOX_AB)
    assert len(g._nodes) == 4
    assert not g._nodes['x'].isna().any()


def test_null_partition_key_names_the_partition_column() -> None:
    n = pd.DataFrame({'id': [0, 1, 2, 3], 'p': ['a', 'a', None, 'b']})
    g = graphistry.edges(PARTITIONED_EDGES, 's', 'd').nodes(n, 'id')
    with pytest.raises(ValueError, match=r"partition_by columns with nulls: \['p'\]"):
        g.circle_layout(partition_by='p', bounding_box=BOUNDING_BOX_AB)


def test_bounding_box_without_partition_key_column_raises_value_error() -> None:
    bb = pd.DataFrame({'cx': [0.], 'cy': [0.], 'w': [1.], 'h': [1.]})
    with pytest.raises(ValueError, match=r"missing columns \['partition_key'\]"):
        _partitioned_graph().circle_layout(partition_by='p', bounding_box=bb)


def test_bounding_box_without_extent_columns_raises_value_error() -> None:
    bb = pd.DataFrame({'partition_key': ['a', 'b'], 'cy': [0., 1.], 'w': [1., 1.], 'h': [1., 1.]})
    with pytest.raises(ValueError, match=r"missing columns \['cx'\]"):
        _partitioned_graph().circle_layout(partition_by='p', bounding_box=bb)


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1")
def test_cudf_positions_match_pandas() -> None:
    import cudf

    for kwargs in [
        {'bounding_box': (0., 0., 10., 10.)},
        {'partition_by': 'p', 'bounding_box': BOUNDING_BOX_AB},
    ]:
        g_pd = _partitioned_graph().circle_layout(**kwargs)
        gpu_kwargs = dict(kwargs)
        if isinstance(gpu_kwargs['bounding_box'], pd.DataFrame):
            gpu_kwargs['bounding_box'] = cudf.from_pandas(gpu_kwargs['bounding_box'])
        g_gdf = (
            graphistry
            .edges(cudf.from_pandas(PARTITIONED_EDGES), 's', 'd')
            .nodes(cudf.from_pandas(PARTITIONED_NODES), 'id')
            .circle_layout(**gpu_kwargs)
        )
        assert isinstance(g_gdf._nodes, cudf.DataFrame)
        got = g_gdf._nodes.to_pandas().sort_values('id').reset_index(drop=True)
        want = g_pd._nodes.sort_values('id').reset_index(drop=True)
        pd.testing.assert_series_equal(got['x'], want['x'], check_dtype=False, rtol=1e-9)
        pd.testing.assert_series_equal(got['y'], want['y'], check_dtype=False, rtol=1e-9)


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1")
def test_cudf_duplicate_bounding_box_partition_keys_raise_value_error() -> None:
    import cudf

    bb = cudf.DataFrame({
        'partition_key': ['a', 'a', 'b', 'b'],
        'cx': [0., 5., 100., 105.],
        'cy': [0., 0., 100., 100.],
        'w': [10.] * 4,
        'h': [10.] * 4,
    })
    g = (
        graphistry
        .edges(cudf.from_pandas(PARTITIONED_EDGES), 's', 'd')
        .nodes(cudf.from_pandas(PARTITIONED_NODES), 'id')
    )
    with pytest.raises(ValueError, match=r"duplicate partition_key values: \['a', 'b'\]"):
        g.circle_layout(partition_by='p', bounding_box=bb)


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1")
def test_cudf_edge_only_graph_materializes_nodes() -> None:
    import cudf

    g = graphistry.edges(cudf.from_pandas(TRIANGLE_EDGES), 's', 'd').circle_layout(bounding_box=(0, 0, 10, 10))
    assert isinstance(g._nodes, cudf.DataFrame)
    assert len(g._nodes) == 3
    assert not g._nodes['x'].isna().any()


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1")
def test_cudf_null_partition_key_names_the_partition_column() -> None:
    import cudf

    n = cudf.DataFrame({'id': [0, 1, 2, 3], 'p': ['a', 'a', None, 'b']})
    g = graphistry.edges(cudf.from_pandas(PARTITIONED_EDGES), 's', 'd').nodes(n, 'id')
    with pytest.raises(ValueError, match=r"partition_by columns with nulls: \['p'\]"):
        g.circle_layout(partition_by='p', bounding_box=cudf.from_pandas(BOUNDING_BOX_AB))
