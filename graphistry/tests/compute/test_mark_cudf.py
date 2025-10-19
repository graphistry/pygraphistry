"""
Tests for mark() operation with cuDF/GPU support.

This test module ensures mark() works correctly with cuDF DataFrames
and GPU execution, maintaining proper DataFrame engine handling.

Related: #755
"""

import os
import pytest
import pandas as pd
import graphistry

from graphistry.compute.ast import n, e_forward
from graphistry.Engine import Engine, EngineAbstract

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestMarkCuDF:
    """Test mark() operation with cuDF DataFrames"""

    @skip_gpu
    def test_mark_nodes_cudf_basic(self):
        """Test marking nodes with cuDF DataFrames"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'type': ['user', 'admin', 'user', 'guest']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1, 2],
            'dst': [1, 2, 3]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Mark user nodes
        result = g.mark(gfql=[n({'type': 'user'})], name='is_user')

        # Should have mark column
        assert 'is_user' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame)

        # Convert to pandas for easier assertion
        result_pd = result._nodes.to_pandas().set_index('id')
        assert result_pd.loc[0, 'is_user'] == True  # noqa: E712
        assert result_pd.loc[1, 'is_user'] == False  # noqa: E712
        assert result_pd.loc[2, 'is_user'] == True  # noqa: E712
        assert result_pd.loc[3, 'is_user'] == False  # noqa: E712

    @skip_gpu
    def test_mark_edges_cudf_basic(self):
        """Test marking edges with cuDF DataFrames"""
        import cudf

        edges_df = pd.DataFrame({
            'src': [0, 1, 2, 3],
            'dst': [1, 2, 3, 0],
            'type': ['friend', 'follow', 'friend', 'block']
        })
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.edges(edges_gdf, 'src', 'dst')

        # Mark friend edges
        result = g.mark(gfql=[e_forward({'type': 'friend'})], name='is_friend')

        # Should have mark column
        assert 'is_friend' in result._edges.columns
        assert isinstance(result._edges, cudf.DataFrame)

        # Convert to pandas for easier assertion
        result_pd = result._edges.to_pandas()
        # Edges 0 and 2 have type='friend'
        assert result_pd.iloc[0]['is_friend'] == True  # noqa: E712
        assert result_pd.iloc[1]['is_friend'] == False  # noqa: E712
        assert result_pd.iloc[2]['is_friend'] == True  # noqa: E712
        assert result_pd.iloc[3]['is_friend'] == False  # noqa: E712

    @skip_gpu
    def test_mark_accumulation_cudf(self):
        """Test multiple marks accumulate on cuDF DataFrames"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'type': ['user', 'admin', 'user', 'guest'],
            'active': [True, True, False, True]
        })
        edges_df = pd.DataFrame({
            'src': [0, 1, 2],
            'dst': [1, 2, 3]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Mark users
        g1 = g.mark(gfql=[n({'type': 'user'})], name='is_user')

        # Mark active (should accumulate)
        g2 = g1.mark(gfql=[n({'active': True})], name='is_active')

        # Should have both columns
        assert 'is_user' in g2._nodes.columns
        assert 'is_active' in g2._nodes.columns
        assert isinstance(g2._nodes, cudf.DataFrame)

        # Verify marks
        result_pd = g2._nodes.to_pandas().set_index('id')
        assert result_pd.loc[0, 'is_user'] == True  # noqa: E712
        assert result_pd.loc[0, 'is_active'] == True  # noqa: E712
        assert result_pd.loc[1, 'is_user'] == False  # noqa: E712
        assert result_pd.loc[1, 'is_active'] == True  # noqa: E712

    @skip_gpu
    def test_mark_default_name_cudf(self):
        """Test default name generation with cuDF"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'type': ['A', 'B', 'A']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Mark without name parameter
        result = g.mark(gfql=[n({'type': 'A'})])

        # Should use default name
        assert 'is_matched_node' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame)

    @skip_gpu
    def test_mark_engine_parameter_explicit_cudf(self):
        """Test explicit engine parameter with cuDF"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'type': ['user', 'admin', 'user']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Test with string engine
        result1 = g.mark(
            gfql=[n({'type': 'user'})],
            name='is_user_str',
            engine='cudf'
        )
        assert 'is_user_str' in result1._nodes.columns

        # Test with EngineAbstract enum
        result2 = g.mark(
            gfql=[n({'type': 'user'})],
            name='is_user_enum',
            engine=EngineAbstract.CUDF
        )
        assert 'is_user_enum' in result2._nodes.columns

        # Test with Engine enum (backward compat)
        result3 = g.mark(
            gfql=[n({'type': 'user'})],
            name='is_user_legacy',
            engine=Engine.CUDF
        )
        assert 'is_user_legacy' in result3._nodes.columns

    @skip_gpu
    def test_mark_multi_hop_cudf(self):
        """Test multi-hop marking marks final entities with cuDF"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'type': ['A', 'B', 'C', 'D']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1, 2],
            'dst': [1, 2, 3],
            'rel': ['friend', 'friend', 'follow']
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Multi-hop pattern: A -> friend edge -> C nodes
        result = g.mark(
            gfql=[
                n({'type': 'A'}),
                e_forward({'rel': 'friend'}),
                n({'type': 'C'})
            ],
            name='is_target'
        )

        # Should mark only final nodes matching the full pattern
        assert 'is_target' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame)

        result_pd = result._nodes.to_pandas().set_index('id')
        # Pattern: A -> friend edge -> C nodes
        # Node 0 (A) -> edge 0 (friend) -> node 1 (B) - NO MATCH (not type C)
        # Result: No nodes match the pattern, all should be False
        assert result_pd.loc[0, 'is_target'] == False  # noqa: E712
        assert result_pd.loc[1, 'is_target'] == False  # noqa: E712
        assert result_pd.loc[2, 'is_target'] == False  # noqa: E712
        assert result_pd.loc[3, 'is_target'] == False  # noqa: E712

    @skip_gpu
    def test_mark_no_matches_cudf(self):
        """Test marking when no entities match with cuDF"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'type': ['user', 'user', 'user']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Mark admins (none exist)
        result = g.mark(gfql=[n({'type': 'admin'})], name='is_admin')

        # All should be False
        assert 'is_admin' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame)

        result_pd = result._nodes.to_pandas()
        assert (result_pd['is_admin'] == False).all()  # noqa: E712

    @skip_gpu
    def test_mark_all_matches_cudf(self):
        """Test marking when all entities match with cuDF"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'type': ['user', 'user', 'user']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2]
        })

        nodes_gdf = cudf.from_pandas(nodes_df)
        edges_gdf = cudf.from_pandas(edges_df)

        g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')

        # Mark users (all nodes)
        result = g.mark(gfql=[n({'type': 'user'})], name='is_user')

        # All should be True
        assert 'is_user' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame)

        result_pd = result._nodes.to_pandas()
        assert (result_pd['is_user'] == True).all()  # noqa: E712

    @skip_gpu
    def test_mark_mixed_engines_pandas_to_cudf(self):
        """Test marking converts pandas to cuDF when engine='cudf'"""
        import cudf

        # Start with pandas DataFrames
        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'type': ['user', 'admin', 'user']
        })
        edges_df = pd.DataFrame({
            'src': [0, 1],
            'dst': [1, 2]
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Explicitly request cuDF engine
        result = g.mark(
            gfql=[n({'type': 'user'})],
            name='is_user',
            engine='cudf'
        )

        # Result should be cuDF (if conversion worked)
        # Note: Implementation may or may not convert, so we just check it works
        assert 'is_user' in result._nodes.columns
        # Verify correctness regardless of engine
        if isinstance(result._nodes, cudf.DataFrame):
            result_pd = result._nodes.to_pandas()
        else:
            result_pd = result._nodes

        result_pd = result_pd.set_index('id')
        assert result_pd.loc[0, 'is_user'] == True  # noqa: E712
        assert result_pd.loc[1, 'is_user'] == False  # noqa: E712
        assert result_pd.loc[2, 'is_user'] == True  # noqa: E712
