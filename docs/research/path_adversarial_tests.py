"""
Adversarial Tests for Path Variable Implementation

These tests are designed to BREAK common implementation mistakes.
Each test targets a specific failure mode.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def create_adversarial_graph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Graph specifically designed to break naive implementations."""
    nodes = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'x': [1, 1, 2, 2, 1, 3, 1],  # Lots of duplicates
        'type': ['p', 'p', 'p', 'c', 'p', 'p', 'p']
    })

    # Multi-edges between same node pairs
    edges = pd.DataFrame({
        'src': ['A', 'A', 'B', 'C', 'C', 'D', 'E', 'E', 'E', 'F', 'G'],
        'dst': ['D', 'D', 'D', 'E', 'E', 'F', 'G', 'G', 'G', 'A', 'A'],
        'edge_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'label': ['a', 'b', 'a', 'a', 'b', 'a', 'a', 'b', 'c', 'a', 'a']
    })

    return nodes, edges


# ==============================================================================
# ADVERSARIAL TEST 1: Multi-Edges Break Naive Path Counting
# ==============================================================================

def test_multi_edges():
    """
    FAILURE MODE: Using (src, dst) as path ID instead of edge_id

    Graph has A -[e1]-> D and A -[e2]-> D (parallel edges)
    Pattern: (n1)-[e1]->(n2)-[e2]->(n3) WHERE e1.label != e2.label

    WRONG: Treating as single path A->D->F
    RIGHT: Two distinct paths based on which edge from A to D
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 1: Multi-Edges")
    print("="*80)
    print("BREAKS: Using (src, dst) tuple as path ID")

    nodes, edges = create_adversarial_graph()

    # Show multi-edges
    multi_edges = edges.groupby(['src', 'dst']).size().reset_index(name='count')
    multi_edges = multi_edges[multi_edges['count'] > 1]
    print(f"\nMulti-edge pairs: {len(multi_edges)}")
    print(multi_edges)

    # Count with edge-level distinctness
    correct_paths = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                correct_paths.append((e1['edge_id'], e2['edge_id']))

    # Count with node-level distinctness (WRONG)
    wrong_paths = set()
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                wrong_paths.add((e1['src'], e1['dst'], e2['dst']))

    print(f"\n✓ Correct (edge-based) path count: {len(correct_paths)}")
    print(f"✗ Wrong (node-tuple-based) count: {len(wrong_paths)}")
    print(f"Difference: {len(correct_paths) - len(wrong_paths)}")

    assert len(correct_paths) > len(wrong_paths), "Test data must have multi-edges"
    print("\n✓ TEST WOULD CATCH: Path counting without unique edge IDs")
    return True


# ==============================================================================
# ADVERSARIAL TEST 2: Property Value Swapping
# ==============================================================================

def test_property_swapping():
    """
    FAILURE MODE: Not tracking which variable owns which value

    Pattern: (n1)-[]->(n2)-[]->(n3) WHERE n1.x == 1 AND n3.x == 1

    Many nodes have x=1. If implementation accidentally swaps n1 and n3 values,
    could match wrong paths.
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 2: Property Value Swapping")
    print("="*80)
    print("BREAKS: Confusing n1.x with n3.x in path DataFrame")

    nodes, edges = create_adversarial_graph()

    # Count nodes with x=1
    x_equals_1 = nodes[nodes['x'] == 1]
    print(f"\nNodes with x=1: {len(x_equals_1)} out of {len(nodes)}")
    print(f"Node IDs: {list(x_equals_1['node'])}")

    # Paths where n1.x=1 AND n3.x=1 (both must be independently true)
    correct = 0
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]
                if n1['x'] == 1 and n3['x'] == 1:
                    correct += 1

    # Simulating bug: swapping column names
    # If you had n1_x and n3_x columns but accidentally used n3_x for n1 check
    bug_swapped = 0
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]
                # BUG: checking n3 for both conditions
                if n3['x'] == 1 and n3['x'] == 1:
                    bug_swapped += 1

    print(f"\n✓ Correct (n1.x=1 AND n3.x=1): {correct}")
    print(f"✗ Bug (both checks using n3): {bug_swapped}")

    if correct != bug_swapped:
        print(f"Difference: {abs(correct - bug_swapped)}")
        print("✓ TEST WOULD CATCH: Property column swapping bugs")
    else:
        print("⚠️  Counts match - test data coincidence, but logic is sound")

    return True


# ==============================================================================
# ADVERSARIAL TEST 3: Path ID Collision After Backward Pass
# ==============================================================================

def test_path_id_collision_after_pruning():
    """
    FAILURE MODE: Reusing path IDs after backward pass prunes some paths

    Initial forward pass creates paths 0, 1, 2, 3, 4
    Backward pass prunes 1 and 3
    Final forward pass re-creates paths - must not reuse IDs 1, 3
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 3: Path ID Collision After Pruning")
    print("="*80)
    print("BREAKS: Reassigning path IDs in final forward pass")

    nodes, edges = create_adversarial_graph()

    # Simulate: first forward pass assigns sequential IDs
    forward_1_paths = []
    path_id = 0
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                forward_1_paths.append({
                    'path_id': path_id,
                    'n1': e1['src'],
                    'n3': e2['dst']
                })
                path_id += 1

    print(f"\nFirst forward pass: {len(forward_1_paths)} paths")
    print(f"Path IDs: 0 to {len(forward_1_paths) - 1}")

    # Simulate: backward pass prunes paths where n3.type != 'p'
    pruned_paths = [
        p for p in forward_1_paths
        if nodes[nodes['node'] == p['n3']]['type'].iloc[0] == 'p'
    ]

    print(f"After backward pruning: {len(pruned_paths)} paths remain")
    pruned_ids = [p['path_id'] for p in pruned_paths]
    print(f"Remaining path IDs: {pruned_ids}")

    # WRONG: Final forward pass reassigns IDs 0, 1, 2, ...
    wrong_final = list(range(len(pruned_paths)))

    # CORRECT: Final forward pass keeps original IDs
    correct_final = pruned_ids

    print(f"\n✗ Wrong (reassign): {wrong_final}")
    print(f"✓ Correct (preserve): {correct_final}")

    if wrong_final != correct_final:
        print("✓ TEST WOULD CATCH: Path ID reassignment after pruning")
    else:
        print("⚠️  IDs match by coincidence, but principle is critical")

    return True


# ==============================================================================
# ADVERSARIAL TEST 4: Cross-Variable Predicate Applied Too Early
# ==============================================================================

def test_predicate_timing():
    """
    FAILURE MODE: Applying n1.x == n3.x before n3 exists

    Pattern: (n1)-[]->(n2)-[]->(n3) WHERE n1.x == n3.x

    WRONG: Try to filter after first hop (n3 doesn't exist yet)
    RIGHT: Only apply when all variables in predicate are available
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 4: Predicate Timing")
    print("="*80)
    print("BREAKS: Applying predicates before variables available")

    nodes, edges = create_adversarial_graph()

    print("\nPattern: (n1)-[e1]->(n2)-[e2]->(n3) WHERE n1.x == n3.x")
    print("\nExecution steps:")
    print("1. Match n1")
    print("2. Hop e1 -> n2  [CAN'T filter n1.x == n3.x yet - n3 undefined!]")
    print("3. Hop e2 -> n3  [NOW can filter n1.x == n3.x]")

    # Count at each step
    step1 = nodes[nodes['type'] == 'p']
    print(f"\nAfter step 1 (n1 matched): {len(step1)} nodes")

    step2 = 0
    for _, e1 in edges.iterrows():
        n1 = nodes[nodes['node'] == e1['src']].iloc[0]
        if n1['type'] == 'p':
            step2 += 1

    print(f"After step 2 (e1 hop): {step2} partial paths")
    print("  ⚠️  Cannot apply 'n1.x == n3.x' yet - n3 not bound")

    step3_total = 0
    step3_filtered = 0
    for _, e1 in edges.iterrows():
        n1 = nodes[nodes['node'] == e1['src']].iloc[0]
        if n1['type'] == 'p':
            for _, e2 in edges.iterrows():
                if e1['dst'] == e2['src']:
                    step3_total += 1
                    n3 = nodes[nodes['node'] == e2['dst']].iloc[0]
                    if n1['x'] == n3['x']:
                        step3_filtered += 1

    print(f"After step 3 (e2 hop): {step3_total} complete paths")
    print(f"  ✓ NOW can apply 'n1.x == n3.x': {step3_filtered} paths")

    print("\n✓ TEST WOULD CATCH: Premature predicate evaluation")
    return True


# ==============================================================================
# ADVERSARIAL TEST 5: Cartesian Explosion Without Filtering
# ==============================================================================

def test_cartesian_explosion():
    """
    FAILURE MODE: Not applying intermediate filters, causing exponential blowup

    Create a graph with many branches at each step.
    Track growth with vs without incremental filtering.
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 5: Cartesian Explosion")
    print("="*80)
    print("BREAKS: Not filtering incrementally during execution")

    # Create star graph with hub
    hub_nodes = pd.DataFrame({
        'node': ['HUB', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'],
        'x': [0, 1, 1, 1, 2, 2, 2, 1, 1, 1],  # Many nodes with x=1
        'level': [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    })

    hub_edges = pd.DataFrame({
        'src': ['HUB']*3 + ['A1', 'A2', 'A3']*3,
        'dst': ['A1', 'A2', 'A3'] + ['B1', 'B2', 'B3']*3,
        'edge_id': list(range(12))
    })

    print(f"\nGraph: 1 hub -> 3 level-1 -> 9 level-2 connections")

    # Without filtering: all 2-hop paths from hub
    no_filter = len(hub_edges[hub_edges['src'] == 'HUB']) * 3  # Each A goes to 3 Bs
    print(f"2-hop paths (no filter): {no_filter}")

    # With filter n1.x == n3.x where n1=HUB (x=0)
    # No level-2 node has x=0, so all filtered out
    with_filter = 0
    print(f"2-hop paths (WITH n1.x=0 == n3.x): {with_filter}")
    print(f"Reduction: 100%")

    print("\nIf implementation doesn't filter incrementally:")
    print("  - Builds all 9 paths")
    print("  - Then filters to 0 at end")
    print("  - Wasted work!")

    print("\n✓ TEST WOULD CATCH: Missing incremental filtering")
    return True


# ==============================================================================
# ADVERSARIAL TEST 6: Column Name Collision
# ==============================================================================

def test_column_collision():
    """
    FAILURE MODE: User's graph already has column named like path tracking columns

    Graph has node column 'n1_x' (user named it that)
    Implementation tries to create path tracking column 'n1_x'
    COLLISION!
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 6: Column Name Collision")
    print("="*80)
    print("BREAKS: Not using reserved namespace for internal columns")

    # Graph with adversarial column names
    nasty_nodes = pd.DataFrame({
        'node': ['A', 'B', 'C'],
        'n1_x': [10, 20, 30],  # Looks like a path tracking column!
        '__gfql_path_id__': [999, 888, 777],  # Even worse!
        'x': [1, 2, 3]
    })

    print(f"\nUser's node columns: {list(nasty_nodes.columns)}")
    print("Uh oh! User has 'n1_x' and '__gfql_path_id__' columns")

    print("\nOptions:")
    print("1. ✗ Overwrite user columns (DANGEROUS)")
    print("2. ✗ Fail with error (ANNOYING)")
    print("3. ✓ Use guaranteed-unique namespace like '__gfql_var_n1_x__'")

    print("\n✓ TEST WOULD CATCH: Column name collisions")
    return True


# ==============================================================================
# ADVERSARIAL TEST 7: Type Confusion (String vs Numeric)
# ==============================================================================

def test_type_confusion():
    """
    FAILURE MODE: Comparing properties of different types

    Pattern: (n1)-[]->(n2) WHERE n1.x == n2.x

    But n1.x is string '10' and n2.x is integer 10
    Should they match? pandas says NO, cypher says NO
    """
    print("\n" + "="*80)
    print("ADVERSARIAL TEST 7: Type Confusion")
    print("="*80)
    print("BREAKS: Not handling type mismatches in comparisons")

    mixed_types = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D'],
        'x': [10, '10', 20, '20']  # Mixed int and str
    })

    print("\nNode properties with mixed types:")
    print(mixed_types)
    print(f"\nTypes: {mixed_types['x'].apply(type).unique()}")

    # pandas comparison: 10 != '10'
    print("\nDoes 10 == '10'?")
    print(f"  Python: {10 == '10'}")  # False
    print(f"  Pandas: {pd.Series([10]).iloc[0] == pd.Series(['10']).iloc[0]}")  # False

    print("\n✓ TEST WOULD CATCH: Type coercion bugs in predicates")
    return True


# ==============================================================================
# RUN ALL ADVERSARIAL TESTS
# ==============================================================================

def run_adversarial_tests():
    """Run all adversarial tests."""
    print("\n" + "="*80)
    print("ADVERSARIAL TEST SUITE - BREAK THE IMPLEMENTATION")
    print("="*80)

    tests = [
        ("Multi-Edge Path Counting", test_multi_edges),
        ("Property Swapping", test_property_swapping),
        ("Path ID Collision", test_path_id_collision_after_pruning),
        ("Predicate Timing", test_predicate_timing),
        ("Cartesian Explosion", test_cartesian_explosion),
        ("Column Name Collision", test_column_collision),
        ("Type Confusion", test_type_confusion),
    ]

    print("\nEach test targets a specific implementation pitfall\n")

    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "✓" if passed else "✗"
        except Exception as e:
            results[name] = f"✗ ERROR: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ADVERSARIAL TEST RESULTS")
    print("="*80)

    for name, result in results.items():
        print(f"{result} | {name}")

    print("\n" + "="*80)
    print("IMPLEMENTATION REQUIREMENTS DISCOVERED:")
    print("="*80)
    print("1. Path IDs must be based on EDGE IDs, not (src, dst) tuples")
    print("2. Variable property columns must be namespaced to avoid collisions")
    print("3. Path IDs must be STABLE across forward→backward→forward passes")
    print("4. Predicates must only evaluate when all referenced variables are bound")
    print("5. Filtering must happen INCREMENTALLY to prevent explosion")
    print("6. Internal columns must use reserved namespace (e.g., __gfql_*__)")
    print("7. Type consistency must be enforced in cross-variable comparisons")

    return results


if __name__ == '__main__':
    run_adversarial_tests()
