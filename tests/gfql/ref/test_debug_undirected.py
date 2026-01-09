"""Debug test for undirected multi-step chain bug."""
import pandas as pd
import graphistry
from graphistry.compute.ast import n, e_undirected
from graphistry.compute.chain import Chain


def test_debug_undirected_chain():
    """Debug why multi-step undirected chain loses node 'c'."""
    # Linear graph: a -> b -> c -> d
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
    })
    edges = pd.DataFrame({
        'src': ['a', 'b', 'c'],
        'dst': ['b', 'c', 'd'],
        'eid': [0, 1, 2],
    })

    g = graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')

    # Single step should work
    chain1 = Chain([n({'id': 'b'}, name='n1'), e_undirected(name='e1'), n(name='n2')])
    result1 = g.gfql(chain1)
    nodes1 = set(result1._nodes['id'].tolist())
    print(f"\nSingle step from b: {nodes1}")
    print(f"  Expected: {{a, b, c}}")
    assert nodes1 == {'a', 'b', 'c'}, f"Single step failed: {nodes1}"

    # Two step is buggy
    chain2 = Chain([
        n({'id': 'b'}, name='n1'),
        e_undirected(name='e1'),
        n(name='n2'),
        e_undirected(name='e2'),
        n(name='n3')
    ])
    result2 = g.gfql(chain2)
    nodes2 = set(result2._nodes['id'].tolist())
    edges2 = result2._edges

    print(f"\nTwo step from b: {nodes2}")
    print(f"  Expected (with edge uniqueness): {{b, c, d}}")
    print(f"  Actual edges: {edges2[['src', 'dst', 'eid']].to_dict('records') if len(edges2) > 0 else 'empty'}")

    # The valid path with edge uniqueness is: b -[e1:b->c]- c -[e2:c->d]- d
    # So we should get nodes {b, c, d}
    # But we're getting {a, b, d} - missing c!

    # Let's check step by step what the forward pass produces
    print("\n--- Checking forward pass ---")
    # Step 0: start at b
    # Step 1: from b, undirected -> {a, c} via edges e0, e1
    # Step 2: from {a, c}, undirected -> ...
    #   from a: only edge e0 (a->b), reaches b
    #   from c: edges e1 (b->c) reaches b, e2 (c->d) reaches d
    # So step 2 should reach {b, d}

    # But in backward pass, we need to prune:
    # - Paths that reuse edges (Cypher edge uniqueness)
    # OR if not enforcing edge uniqueness:
    # - Just validate that paths exist

    # Current bug: 'c' is missing
    # This suggests the backward pass is incorrectly pruning 'c'

    assert 'c' in nodes2, f"BUG: 'c' should be in result, got {nodes2}"


if __name__ == "__main__":
    test_debug_undirected_chain()
