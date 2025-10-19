#!/usr/bin/env python3
"""
Generate JSON wire protocol examples for common GFQL patterns.

This script creates concrete examples of how GFQL Python code translates to JSON.
"""

import json
import sys
import os

# Add parent directory to path so we can import graphistry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from graphistry.compute.ast import (
    n, e, e_forward, e_reverse, e_undirected,
    ASTCall, ASTLet, ASTRef, call
)
from graphistry.compute.chain import Chain
from graphistry.compute.predicates.is_in import is_in
from graphistry.compute.predicates.numeric import gt, lt, ge, le
from graphistry.compute.predicates.str import contains, startswith


def print_example(title, python_code, ast_object):
    """Print a formatted example with Python and JSON forms."""
    print(f"\n{'='*80}")
    print(f"Example: {title}")
    print(f"{'='*80}")
    print(f"\nPython form:")
    print(f"```python")
    print(python_code)
    print(f"```")
    print(f"\nJSON wire protocol:")
    print(f"```json")
    print(json.dumps(ast_object.to_json(), indent=2))
    print(f"```")


def main():
    """Generate all examples."""

    print("# GFQL JSON Wire Protocol Examples")
    print("Generated from actual AST objects using .to_json() method")

    # ========================================================================
    # EXAMPLE 1: Simple Node Matcher
    # ========================================================================
    print_example(
        "Simple Node Matcher",
        "n({'type': 'person'})",
        n({'type': 'person'})
    )

    # ========================================================================
    # EXAMPLE 2: Node Matcher with Predicate
    # ========================================================================
    print_example(
        "Node Matcher with Predicate",
        "n({'age': gt(18), 'country': is_in(['USA', 'Canada'])})",
        n({'age': gt(18), 'country': is_in(['USA', 'Canada'])})
    )

    # ========================================================================
    # EXAMPLE 3: Edge Matcher (Forward)
    # ========================================================================
    print_example(
        "Forward Edge Matcher",
        "e_forward(edge_match={'weight': gt(0.5)}, hops=2)",
        e_forward(edge_match={'weight': gt(0.5)}, hops=2)
    )

    # ========================================================================
    # EXAMPLE 4: Edge Matcher (Undirected with predicates)
    # ========================================================================
    print_example(
        "Undirected Edge with Node Predicates",
        """e_undirected(
    source_node_match={'type': 'user'},
    edge_match={'weight': ge(0.8)},
    destination_node_match={'type': 'post'},
    hops=1
)""",
        e_undirected(
            source_node_match={'type': 'user'},
            edge_match={'weight': ge(0.8)},
            destination_node_match={'type': 'post'},
            hops=1
        )
    )

    # ========================================================================
    # EXAMPLE 5: Simple Chain (Graph Traversal Pattern)
    # ========================================================================
    print_example(
        "Simple Chain - Find Friends of Friends",
        """Chain([
    n({'type': 'person', 'name': 'Alice'}),
    e_forward(edge_match={'relationship': 'friend'}),
    n({'type': 'person'}),
    e_forward(edge_match={'relationship': 'friend'}),
    n({'type': 'person'})
])""",
        Chain([
            n({'type': 'person', 'name': 'Alice'}),
            e_forward(edge_match={'relationship': 'friend'}),
            n({'type': 'person'}),
            e_forward(edge_match={'relationship': 'friend'}),
            n({'type': 'person'})
        ])
    )

    # ========================================================================
    # EXAMPLE 6: Call Operation - Hypergraph
    # ========================================================================
    print_example(
        "Call Operation - Hypergraph Transformation",
        """ASTCall('hypergraph', {
    'entity_types': ['user', 'post', 'comment'],
    'direct': True,
    'opts': {'USE_FEAT_V2': True}
})""",
        ASTCall('hypergraph', {
            'entity_types': ['user', 'post', 'comment'],
            'direct': True,
            'opts': {'USE_FEAT_V2': True}
        })
    )

    # ========================================================================
    # EXAMPLE 7: Call Operation - UMAP
    # ========================================================================
    print_example(
        "Call Operation - UMAP Layout",
        """ASTCall('umap', {
    'kind': 'nodes',
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2
})""",
        ASTCall('umap', {
            'kind': 'nodes',
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2
        })
    )

    # ========================================================================
    # EXAMPLE 8: Let Binding (DAG) - Multi-step Query
    # ========================================================================
    print_example(
        "Let Binding (DAG) - Multi-step Query",
        """ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'active_users': Chain([n({'type': 'user', 'last_login': gt(30)})]),
    'with_degrees': ASTCall('get_degrees', {'col': 'degree'}),
    'high_degree': Chain([n({'degree': gt(10)})])
})""",
        ASTLet({
            'users': Chain([n({'type': 'user'})]),
            'active_users': Chain([n({'type': 'user', 'last_login': gt(30)})]),
            'with_degrees': ASTCall('get_degrees', {'col': 'degree'}),
            'high_degree': Chain([n({'degree': gt(10)})])
        })
    )

    # ========================================================================
    # EXAMPLE 9: Let Binding with References
    # ========================================================================
    print_example(
        "Let Binding with References",
        """ASTLet({
    'users': Chain([n({'type': 'user'})]),
    'user_friends': ASTRef('users', [
        e_forward(edge_match={'type': 'friend'}),
        n()
    ])
})""",
        ASTLet({
            'users': Chain([n({'type': 'user'})]),
            'user_friends': ASTRef('users', [
                e_forward(edge_match={'type': 'friend'}),
                n()
            ])
        })
    )

    # ========================================================================
    # EXAMPLE 10: String Predicates
    # ========================================================================
    print_example(
        "String Predicates",
        """n({
    'name': contains('smith'),
    'email': startswith('@example.com')
})""",
        n({
            'name': contains('smith'),
            'email': startswith('@example.com')
        })
    )

    # ========================================================================
    # EXAMPLE 11: Complex Chain with Multiple Predicates
    # ========================================================================
    print_example(
        "Complex Chain - E-commerce Pattern",
        """Chain([
    n({'type': 'user', 'country': is_in(['USA', 'Canada'])}),
    e_forward(edge_match={'action': 'purchased'}),
    n({'type': 'product', 'price': gt(100)}),
    e_reverse(edge_match={'action': 'also_purchased'}),
    n({'type': 'product'})
])""",
        Chain([
            n({'type': 'user', 'country': is_in(['USA', 'Canada'])}),
            e_forward(edge_match={'action': 'purchased'}),
            n({'type': 'product', 'price': gt(100)}),
            e_reverse(edge_match={'action': 'also_purchased'}),
            n({'type': 'product'})
        ])
    )

    # ========================================================================
    # EXAMPLE 12: Graph Algorithm Call
    # ========================================================================
    print_example(
        "Graph Algorithm - PageRank",
        """ASTCall('compute_cugraph', {
    'alg': 'pagerank',
    'out_col': 'pagerank_score',
    'directed': True
})""",
        ASTCall('compute_cugraph', {
            'alg': 'pagerank',
            'out_col': 'pagerank_score',
            'directed': True
        })
    )

    # ========================================================================
    # EXAMPLE 13: Layout Call
    # ========================================================================
    print_example(
        "Layout - Force Atlas 2",
        """ASTCall('fa2_layout', {
    'fa2_params': {'iterations': 1000, 'scalingRatio': 2.0}
})""",
        ASTCall('fa2_layout', {
            'fa2_params': {'iterations': 1000, 'scalingRatio': 2.0}
        })
    )

    print(f"\n{'='*80}")
    print("End of Examples")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
