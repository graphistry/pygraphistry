"""Test that Let bindings support matchers (ASTNode/ASTEdge)."""

import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tests.test_compute import CGFull
from graphistry.compute.ast import n, e_forward, let, ref, ge, ASTNode, ASTEdge
from graphistry.compute.chain import Chain


class TestLetMatchers:
    """Test Let bindings with ASTNode and ASTEdge matchers."""

    def test_node_matcher_in_let(self):
        """Test that ASTNode can be used as Let binding."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd'], 'type': ['x', 'y', 'x']}),
            's', 'd'
        ).nodes(
            pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'kind': ['person', 'person', 'company', 'company']}),
            'id'
        )

        # ASTNode should now be accepted
        result = g.gfql(let({
            'persons': n({'kind': 'person'})
        }))

        # Check result has the named column
        assert 'persons' in result._nodes.columns
        assert result._nodes['persons'].sum() == 2  # 'a' and 'b' are persons

    def test_edge_matcher_in_let(self):
        """Test that ASTEdge can be used as Let binding."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd'], 'type': ['x', 'y', 'x']}),
            's', 'd'
        )

        # ASTEdge should now be accepted
        result = g.gfql(let({
            'x_edges': e_forward({'type': 'x'})
        }))

        # Check result has the named column
        assert 'x_edges' in result._edges.columns
        assert result._edges['x_edges'].sum() == 2  # Two edges with type 'x'

    def test_mixed_matchers_and_refs(self):
        """Test Let with both matchers and refs."""
        g = CGFull().edges(
            pd.DataFrame({
                's': [1, 2, 3, 4, 5],
                'd': [2, 3, 4, 5, 1],
                'type': ['knows', 'knows', 'works', 'knows', 'knows']
            }),
            's', 'd'
        ).nodes(
            pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'type': ['person', 'person', 'person', 'company', 'person'],
                'age': [25, 30, 17, 0, 40]
            }),
            'id'
        )

        result = g.gfql(let({
            'persons': n({'type': 'person'}),
            'adults': ref('persons', [n({'age': ge(18)})]),
            'knows_edges': e_forward({'type': 'knows'})
        }))

        # Check all named results exist
        assert 'persons' in result._nodes.columns
        assert 'adults' in result._nodes.columns
        assert 'knows_edges' in result._edges.columns

        # Verify counts
        assert result._nodes['persons'].sum() == 4  # 4 persons
        assert result._nodes['adults'].sum() == 3  # 3 adults (age >= 18)
        assert result._edges['knows_edges'].sum() == 4  # 4 knows edges

    def test_matchers_operate_on_root_graph(self):
        """Test that matchers in Let operate on the root graph, not on previous bindings."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a'], 'd': ['b']}),
            's', 'd'
        ).nodes(
            pd.DataFrame({
                'id': ['a', 'b', 'c', 'd'],
                'type': ['person', 'person', 'company', 'company']
            }),
            'id'
        )

        result = g.gfql(let({
            'persons': n({'type': 'person'}),
            'companies': n({'type': 'company'})  # Should find companies from root, not from persons
        }))

        # Both should succeed independently
        assert 'persons' in result._nodes.columns
        assert 'companies' in result._nodes.columns
        assert result._nodes['persons'].sum() == 2  # 2 persons
        assert result._nodes['companies'].sum() == 2  # 2 companies

    def test_chain_still_works(self):
        """Test that Chain still works as Let binding."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}),
            's', 'd'
        ).nodes(
            pd.DataFrame({'id': ['a', 'b', 'c'], 'type': ['x', 'y', 'z']}),
            'id'
        )

        # Chain should still work
        result = g.gfql(let({
            'pattern': Chain([n({'type': 'x'}), e_forward(), n()])
        }))

        assert result is not None
        # Chain execution would create the full pattern result

    def test_validate_matcher_types(self):
        """Test that Let validates matcher types properly."""
        g = CGFull()

        # Valid matchers should work
        dag = let({
            'node': ASTNode(),
            'edge': ASTEdge(direction='forward', hops=1)
        })
        assert dag is not None

        # Invalid type should still fail
        from graphistry.compute.exceptions import GFQLTypeError

        class InvalidMatcher:
            pass

        with pytest.raises(GFQLTypeError) as exc_info:
            let({'invalid': InvalidMatcher()})

        assert 'valid operation' in str(exc_info.value)

    def test_backwards_compatibility(self):
        """Test that existing code with Chain wrapper still works."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a'], 'd': ['b']}),
            's', 'd'
        )

        # Old style with Chain wrapper should still work
        result = g.gfql(let({
            'pattern': Chain([n()])
        }))

        assert result is not None