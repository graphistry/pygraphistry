"""Test that Let bindings support matchers (ASTNode/ASTEdge)."""

import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tests.test_compute import CGFull  # noqa: E402
from graphistry.compute.ast import n, e_forward, let, ref, ge, ASTNode, ASTEdge  # noqa: E402
from graphistry.compute.chain import Chain  # noqa: E402


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

        # FILTER semantics: result should only contain persons
        assert len(result._nodes) == 2  # Only 'a' and 'b'
        assert all(result._nodes['kind'] == 'person')

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

        # FILTER semantics: result should only contain x edges
        assert len(result._edges) == 2  # Two edges with type 'x'
        assert all(result._edges['type'] == 'x')

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

        # FILTER semantics: last executed binding in topological order is 'adults'
        # 'adults' depends on 'persons', so executes after 'knows_edges'
        # Result is adult persons (nodes only, no edges since n() filters to nodes)
        assert len(result._nodes) == 3  # Adults: nodes 1, 2, 5
        assert all(result._nodes['age'] >= 18)
        assert len(result._edges) == 0  # n() returns just nodes

    def test_ref_uses_binding_graph(self):
        """Test that ASTRef chains operate on the referenced graph, not the root."""
        g = CGFull().edges(
            pd.DataFrame({'s': ['a'], 'd': ['b']}),
            's', 'd'
        ).nodes(
            pd.DataFrame({'id': ['a', 'b'], 'type': ['person', 'person']}),
            'id'
        )

        result = g.gfql(let({
            'persons': n({'type': 'person'}),
            'neighbors': ref('persons', [e_forward(), n()])
        }), output='neighbors')

        assert len(result._edges) == 0
        assert len(result._nodes) == 0

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

        # FILTER semantics: last binding (companies) should return only companies
        assert len(result._nodes) == 2  # 2 companies
        assert all(result._nodes['type'] == 'company')

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
