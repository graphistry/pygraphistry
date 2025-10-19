"""
False Positive Detection Tests

Tests designed to catch implementations that INCORRECTLY PASS when they should FAIL.
Each test has paths that LOOK like they match but DON'T.

Focus: Correctness over functionality
Goal: Zero false positives
"""
import pandas as pd
from typing import List, Tuple, Set
import sys


def create_trap_graph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Graph with many false-positive traps."""
    nodes = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
        'x': [10, 15, 20, 10, 25, 10, 30, 20, 10, 15, 20, 10],  # Repeated values
        'y': [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12],  # Unique values
        'type': ['p', 'p', 'p', 'c', 'p', 'c', 'p', 'c', 'p', 'p', 'p', 'c']
    })

    # Directed edges with loops and chains
    edges = pd.DataFrame({
        'src':  ['A', 'B', 'C', 'A', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'B', 'C', 'E', 'G', 'A', 'D'],
        'dst':  ['B', 'C', 'D', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'A', 'B', 'D', 'F', 'E', 'A'],
        'edge_id': list(range(18)),
        'weight': [1, 2, 3, 5, 1, 2, 3, 1, 2, 3, 1, 2, 10, 20, 15, 5, 8, 12]
    })

    return nodes, edges


# ==============================================================================
# TOPOLOGY TEST 1: Dead-End Chain (Partial Match Failure)
# ==============================================================================

def test_dead_end_chain():
    """
    FALSE POSITIVE TRAP: Path starts matching but dead-ends before completion.

    Pattern: (n1:x=10)-[]->(n2)-[]->(n3)-[]->(n4:x=10)

    A(x=10) -> B -> C -> D(x=10)  ✓ SHOULD MATCH
    A(x=10) -> D -> E -> ?        ✗ Dead end (E has no outgoing to x=10)

    NAIVE BUG: Counting (A, D, E) as match because A.x=10 and forgetting n4.x=10
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 1: Dead-End Chain")
    print("="*80)
    print("TRAP: Partial matches that don't complete")

    nodes, edges = create_trap_graph()

    # Ground truth: manually verified correct paths
    correct_paths = []

    # Check all 3-hop paths
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] != e2['src']:
                continue
            for _, e3 in edges.iterrows():
                if e2['dst'] != e3['src']:
                    continue

                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n4 = nodes[nodes['node'] == e3['dst']].iloc[0]

                # BOTH n1.x == 10 AND n4.x == 10
                if n1['x'] == 10 and n4['x'] == 10:
                    correct_paths.append((e1['src'], e2['src'], e3['src'], e3['dst']))

    # NAIVE BUG: Only checking n1.x == 10, forgetting about n4
    naive_bug_paths = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] != e2['src']:
                continue
            for _, e3 in edges.iterrows():
                if e2['dst'] != e3['src']:
                    continue

                n1 = nodes[nodes['node'] == e1['src']].iloc[0]

                # BUG: Only checking n1
                if n1['x'] == 10:
                    naive_bug_paths.append((e1['src'], e2['src'], e3['src'], e3['dst']))

    print(f"\n✓ Correct paths (n1.x=10 AND n4.x=10): {len(correct_paths)}")
    print(f"✗ Naive bug (only n1.x=10): {len(naive_bug_paths)}")
    print(f"FALSE POSITIVES: {len(naive_bug_paths) - len(correct_paths)}")

    if len(naive_bug_paths) > len(correct_paths):
        print("\n⚠️  CRITICAL: Implementation must check ALL predicates!")
        for path in naive_bug_paths[:3]:
            n4 = nodes[nodes['node'] == path[3]].iloc[0]
            if n4['x'] != 10:
                print(f"  False positive: {path[0]}->...->{path[3]} (n4.x={n4['x']}, not 10)")
        return True
    else:
        print("⚠️  Test data needs more dead ends")
        return False


# ==============================================================================
# TOPOLOGY TEST 2: Loop Detection
# ==============================================================================

def test_loop_false_positive():
    """
    FALSE POSITIVE TRAP: Counting loops that shouldn't match.

    Pattern: (n1)-[]->(n2)-[]->(n3) WHERE n1 != n3 (no self-loops)

    A -> B -> A  ✗ SHOULD NOT MATCH (loop back to start)
    A -> B -> C  ✓ Should match

    NAIVE BUG: Not checking node identity, only properties
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 2: Loop Detection")
    print("="*80)
    print("TRAP: Paths that loop back to start node")

    nodes, edges = create_trap_graph()

    # Correct: n1 != n3 (no loops)
    correct_no_loops = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src'] and e1['src'] != e2['dst']:
                correct_no_loops.append((e1['src'], e1['dst'], e2['dst']))

    # Naive: Forgets to check n1 != n3
    naive_with_loops = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                naive_with_loops.append((e1['src'], e1['dst'], e2['dst']))

    loops = [p for p in naive_with_loops if p[0] == p[2]]

    print(f"\n✓ Correct (no loops): {len(correct_no_loops)}")
    print(f"✗ Naive (with loops): {len(naive_with_loops)}")
    print(f"FALSE POSITIVES (loops): {len(loops)}")

    if loops:
        print("\nLoop examples (should be REJECTED):")
        for path in loops[:3]:
            print(f"  {path[0]} -> {path[1]} -> {path[2]} (LOOP!)")
        return True
    else:
        print("⚠️  No loops in test data")
        return False


# ==============================================================================
# TOPOLOGY TEST 3: Direction Matters
# ==============================================================================

def test_direction_violation():
    """
    FALSE POSITIVE TRAP: Following edges backwards when pattern says forward.

    Pattern: (n1)-[:forward]->(n2)-[:forward]->(n3)

    Edge: A -> B (forward)

    A -[forward]-> B  ✓ Correct
    B -[backward]-> A ✗ WRONG DIRECTION

    NAIVE BUG: Using undirected traversal for directed pattern
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 3: Edge Direction")
    print("="*80)
    print("TRAP: Following edges in wrong direction")

    nodes, edges = create_trap_graph()

    # Correct: Respect direction
    correct_forward = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:  # Forward only
                correct_forward.append((e1['src'], e2['src'], e2['dst']))

    # Naive bug: Treat as undirected
    naive_undirected = []
    edge_pairs = set()
    for _, e in edges.iterrows():
        edge_pairs.add((e['src'], e['dst']))
        edge_pairs.add((e['dst'], e['src']))  # Add reverse

    for src1, dst1 in edge_pairs:
        for src2, dst2 in edge_pairs:
            if dst1 == src2:
                naive_undirected.append((src1, src2, dst2))

    print(f"\n✓ Correct (directed): {len(correct_forward)}")
    print(f"✗ Naive (undirected): {len(naive_undirected)}")
    print(f"FALSE POSITIVES: {len(naive_undirected) - len(correct_forward)}")

    if len(naive_undirected) > len(correct_forward):
        print("\n⚠️  CRITICAL: Must respect edge direction!")
        return True
    else:
        return False


# ==============================================================================
# PREDICATE TEST 1: Inequality Operators
# ==============================================================================

def test_inequality_correctness():
    """
    FALSE POSITIVE TRAP: Using wrong comparison operator.

    Pattern: (n1)-[]->(n2)-[]->(n3) WHERE n1.x < n3.x

    A(x=10) -> B -> C(x=20)  ✓ 10 < 20 CORRECT
    A(x=10) -> B -> F(x=10)  ✗ 10 < 10 FALSE
    A(x=10) -> B -> D(x=10)  ✗ 10 < 10 FALSE

    NAIVE BUG: Using <= instead of <, or == instead of !=
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 4: Inequality Operators (<, >, !=)")
    print("="*80)
    print("TRAP: Using wrong comparison operator")

    nodes, edges = create_trap_graph()

    # Correct: STRICT less-than
    correct_lt = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]

                if n1['x'] < n3['x']:  # STRICT <
                    correct_lt.append((e1['src'], e2['dst'], n1['x'], n3['x']))

    # Bug: Using <= instead of <
    bug_lte = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]

                if n1['x'] <= n3['x']:  # BUG: <=
                    bug_lte.append((e1['src'], e2['dst'], n1['x'], n3['x']))

    false_positives = [p for p in bug_lte if p[2] == p[3]]  # Cases where x equals

    print(f"\n✓ Correct (n1.x < n3.x): {len(correct_lt)}")
    print(f"✗ Bug (n1.x <= n3.x): {len(bug_lte)}")
    print(f"FALSE POSITIVES (equals): {len(false_positives)}")

    if false_positives:
        print("\nFalse positive examples (x values equal, should be REJECTED):")
        for path in false_positives[:3]:
            print(f"  {path[0]} (x={path[2]}) -> {path[1]} (x={path[3]}) - EQUAL!")
        return True
    else:
        print("⚠️  Need more equal-value cases in test data")
        return False


# ==============================================================================
# PREDICATE TEST 2: Multiple Field Conjunction (AND)
# ==============================================================================

def test_multi_field_and():
    """
    FALSE POSITIVE TRAP: Using OR when should be AND.

    Pattern: WHERE n1.x == n3.x AND n1.y < n3.y

    A(x=10,y=1) -> B -> I(x=10,y=9)  ✓ Both conditions true
    A(x=10,y=1) -> B -> F(x=10,y=6)  ✓ Both conditions true
    A(x=10,y=1) -> B -> C(x=20,y=3)  ✗ x mismatch (10 != 20)
    E(x=25,y=5) -> F -> G(x=30,y=7)  ✗ x mismatch (25 != 30)

    NAIVE BUG: Checking OR instead of AND
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 5: Multi-Field AND")
    print("="*80)
    print("TRAP: Using OR when should be AND")

    nodes, edges = create_trap_graph()

    # Correct: BOTH conditions must be true
    correct_and = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]

                if n1['x'] == n3['x'] and n1['y'] < n3['y']:  # AND
                    correct_and.append((e1['src'], e2['dst'], n1['x'], n1['y'], n3['x'], n3['y']))

    # Bug: Using OR instead of AND
    bug_or = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src']:
                n1 = nodes[nodes['node'] == e1['src']].iloc[0]
                n3 = nodes[nodes['node'] == e2['dst']].iloc[0]

                if n1['x'] == n3['x'] or n1['y'] < n3['y']:  # BUG: OR
                    bug_or.append((e1['src'], e2['dst'], n1['x'], n1['y'], n3['x'], n3['y']))

    print(f"\n✓ Correct (x match AND y less): {len(correct_and)}")
    print(f"✗ Bug (x match OR y less): {len(bug_or)}")
    print(f"FALSE POSITIVES: {len(bug_or) - len(correct_and)}")

    if len(bug_or) > len(correct_and):
        # Show cases where only one condition is true
        for path in bug_or[:5]:
            x_match = path[2] == path[4]
            y_less = path[3] < path[5]
            if x_match and not y_less:
                print(f"  {path[0]} -> {path[1]}: x match but y NOT less ({path[3]} >= {path[5]})")
            elif y_less and not x_match:
                print(f"  {path[0]} -> {path[1]}: y less but x NOT match ({path[2]} != {path[4]})")
        return True
    else:
        return False


# ==============================================================================
# PREDICATE TEST 3: Edge Property Comparison
# ==============================================================================

def test_edge_weight_comparison():
    """
    FALSE POSITIVE TRAP: Comparing wrong edge weights.

    Pattern: (n1)-[e1]->(n2)-[e2]->(n3) WHERE e1.weight < e2.weight

    A -[w=1]-> D -[w=5]-> E  ✓ 1 < 5 CORRECT
    B -[w=10]-> A -[w=1]-> D ✗ 10 < 1 FALSE

    NAIVE BUG: Comparing e2.weight < e1.weight (reversed)
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 6: Edge Weight Comparison")
    print("="*80)
    print("TRAP: Comparing edges in wrong order")

    nodes, edges = create_trap_graph()

    # Correct: e1.weight < e2.weight
    correct_order = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src'] and e1['weight'] < e2['weight']:
                correct_order.append((e1['src'], e2['src'], e2['dst'], e1['weight'], e2['weight']))

    # Bug: Reversed comparison
    bug_reversed = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] == e2['src'] and e2['weight'] < e1['weight']:  # REVERSED
                bug_reversed.append((e1['src'], e2['src'], e2['dst'], e1['weight'], e2['weight']))

    print(f"\n✓ Correct (e1.w < e2.w): {len(correct_order)}")
    print(f"✗ Bug (e2.w < e1.w): {len(bug_reversed)}")

    # These should be disjoint sets
    overlap = set(correct_order) & set(bug_reversed)
    assert len(overlap) == 0, "Bug: Sets should be disjoint!"

    print(f"Sets correctly disjoint: {len(overlap) == 0}")

    if len(bug_reversed) > 0:
        print("\nBug would return these (wrong direction):")
        for path in bug_reversed[:3]:
            print(f"  {path[0]} -[{path[3]}]-> {path[1]} -[{path[4]}]-> {path[2]} (reversed!)")
        return True
    else:
        return False


# ==============================================================================
# TOPOLOGY TEST 4: Long Chain Correctness
# ==============================================================================

def test_long_chain():
    """
    FALSE POSITIVE TRAP: Losing track of constraints in long chains.

    Pattern: 5-hop chain where FIRST.x == LAST.x
    (n1)-[]->(n2)-[]->(n3)-[]->(n4)-[]->(n5)-[]->(n6) WHERE n1.x == n6.x

    Must track n1.x through entire 5-hop chain.

    NAIVE BUG: Forgetting n1.x by hop 5
    """
    print("\n" + "="*80)
    print("FALSE POSITIVE TEST 7: Long Chain (5-hop)")
    print("="*80)
    print("TRAP: Losing constraint tracking in long chains")

    nodes, edges = create_trap_graph()

    # Find 5-hop paths
    hop5_paths = []
    for _, e1 in edges.iterrows():
        for _, e2 in edges.iterrows():
            if e1['dst'] != e2['src']:
                continue
            for _, e3 in edges.iterrows():
                if e2['dst'] != e3['src']:
                    continue
                for _, e4 in edges.iterrows():
                    if e3['dst'] != e4['src']:
                        continue
                    for _, e5 in edges.iterrows():
                        if e4['dst'] != e5['src']:
                            continue

                        hop5_paths.append({
                            'n1': e1['src'],
                            'n6': e5['dst'],
                            'path': f"{e1['src']}->{e2['src']}->{e3['src']}->{e4['src']}->{e5['src']}->{e5['dst']}"
                        })

    # Correct: n1.x == n6.x
    correct = [
        p for p in hop5_paths
        if nodes[nodes['node'] == p['n1']]['x'].iloc[0] == nodes[nodes['node'] == p['n6']]['x'].iloc[0]
    ]

    print(f"\nTotal 5-hop paths: {len(hop5_paths)}")
    print(f"✓ Matching n1.x == n6.x: {len(correct)}")
    print(f"✗ Would be false positives without check: {len(hop5_paths) - len(correct)}")

    if len(correct) < len(hop5_paths):
        print(f"\n⚠️  CRITICAL: Must track constraints through {5}-hop chain!")
        return True
    else:
        print("All paths happen to match - need more diverse data")
        return False


# ==============================================================================
# RUN ALL FALSE POSITIVE TESTS
# ==============================================================================

def run_false_positive_tests():
    """Run all false positive detection tests."""
    print("\n" + "="*80)
    print("FALSE POSITIVE DETECTION TEST SUITE")
    print("="*80)
    print("Goal: Catch implementations that return WRONG results\n")

    tests = [
        ("Dead-End Chain", test_dead_end_chain),
        ("Loop Detection", test_loop_false_positive),
        ("Edge Direction", test_direction_violation),
        ("Inequality (<, >)", test_inequality_correctness),
        ("Multi-Field AND", test_multi_field_and),
        ("Edge Weight Order", test_edge_weight_comparison),
        ("Long Chain (5-hop)", test_long_chain),
    ]

    results = {}
    for name, test_func in tests:
        try:
            detected = test_func()
            results[name] = "✓ TRAP DETECTED" if detected else "⚠️ weak"
        except Exception as e:
            results[name] = f"✗ ERROR: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("FALSE POSITIVE TEST RESULTS")
    print("="*80)

    for name, result in results.items():
        print(f"{result:20} | {name}")

    detected_count = sum(1 for r in results.values() if "DETECTED" in r)
    total_count = len(tests)

    print(f"\nTraps detected: {detected_count}/{total_count}")

    if detected_count == total_count:
        print("\n✓ All false positive traps validated!")
    else:
        print("\n⚠️  Some tests need better data to expose traps")

    print("\n" + "="*80)
    print("KEY TAKEAWAY:")
    print("="*80)
    print("Implementation must:")
    print("  1. Check ALL predicates (not just some)")
    print("  2. Respect edge direction")
    print("  3. Use correct operators (<, not <=)")
    print("  4. Use AND (not OR) for conjunctions")
    print("  5. Track constraints through long chains")
    print("  6. Reject loops when specified")
    print("  7. Compare edges in correct order")

    return results


if __name__ == '__main__':
    results = run_false_positive_tests()

    # Exit with error if any test failed to detect its trap
    if any("weak" in r or "ERROR" in r for r in results.values()):
        print("\n⚠️  Some traps not fully validated - consider enriching test data")
        sys.exit(1)
