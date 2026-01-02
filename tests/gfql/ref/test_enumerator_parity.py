import pandas as pd
import pytest

from graphistry.compute import e_forward, e_reverse, e_undirected, n
from graphistry.compute.ast import ASTEdge, ASTNode
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull


def _to_pandas(df):
    if df is None:
        return None
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _alias_bindings(df, id_col, alias):
    if df is None or alias not in df.columns:
        return set()
    return set(df.loc[df[alias].astype(bool), id_col])


def _run_parity_case(nodes, edges, ops, check_hop_labels=False):
    g = (
        CGFull()
        .nodes(pd.DataFrame(nodes), "id")
        .edges(pd.DataFrame(edges), "src", "dst", edge="edge_id")
    )
    gfql_result = g.gfql(ops)
    oracle = enumerate_chain(g, ops, caps=OracleCaps(max_nodes=50, max_edges=50))

    gfql_nodes = _to_pandas(gfql_result._nodes)
    gfql_edges = _to_pandas(gfql_result._edges)

    assert gfql_nodes is not None
    assert set(gfql_nodes[g._node]) == set(oracle.nodes[g._node])

    if g._edge is not None and gfql_edges is not None and not gfql_edges.empty:
        assert set(gfql_edges[g._edge]) == set(oracle.edges[g._edge])
    else:
        assert oracle.edges.empty

    for op in ops:
        alias = getattr(op, "_name", None)
        if not alias:
            continue
        if isinstance(op, ASTNode):
            assert oracle.tags.get(alias, set()) == _alias_bindings(
                gfql_nodes, g._node, alias
            )
        elif isinstance(op, ASTEdge):
            assert oracle.tags.get(alias, set()) == _alias_bindings(
                gfql_edges, g._edge, alias
            )

    # Check hop labels if requested
    if check_hop_labels:
        for op in ops:
            if not isinstance(op, ASTEdge):
                continue
            node_hop_col = getattr(op, 'label_node_hops', None)
            edge_hop_col = getattr(op, 'label_edge_hops', None)
            label_seeds = getattr(op, 'label_seeds', False)

            if node_hop_col and gfql_nodes is not None and node_hop_col in gfql_nodes.columns:
                # Compare node hop labels
                gfql_node_hops = {
                    row[g._node]: int(row[node_hop_col])
                    for _, row in gfql_nodes.iterrows()
                    if pd.notna(row[node_hop_col])
                }
                oracle_node_hops = oracle.node_hop_labels or {}
                # Oracle should match GFQL for non-seed nodes
                for nid, hop in gfql_node_hops.items():
                    if hop == 0 and not label_seeds:
                        continue  # Skip seeds when label_seeds=False
                    assert nid in oracle_node_hops, f"Node {nid} with hop {hop} not in oracle"
                    assert oracle_node_hops[nid] == hop, f"Node {nid}: oracle hop {oracle_node_hops[nid]} != gfql hop {hop}"

            if edge_hop_col and gfql_edges is not None and edge_hop_col in gfql_edges.columns:
                # Compare edge hop labels
                gfql_edge_hops = {
                    row[g._edge]: int(row[edge_hop_col])
                    for _, row in gfql_edges.iterrows()
                    if pd.notna(row[edge_hop_col])
                }
                oracle_edge_hops = oracle.edge_hop_labels or {}
                for eid, hop in gfql_edge_hops.items():
                    assert eid in oracle_edge_hops, f"Edge {eid} with hop {hop} not in oracle"
                    assert oracle_edge_hops[eid] == hop, f"Edge {eid}: oracle hop {oracle_edge_hops[eid]} != gfql hop {hop}"

    return oracle  # Return for additional assertions in specific tests


CASES = [
    (
        "forward",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "acct3", "type": "txn"},
            {"edge_id": "e3", "src": "acct3", "dst": "acct1", "type": "txn"},
        ],
        [n({"type": "account"}, name="start"), e_forward({"type": "txn"}, name="hop"),
n({"type": "account"}, name="end")],
    ),
    (
        "reverse",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ],
        [
            {"edge_id": "owns1", "src": "acct1", "dst": "user1", "type": "owns"},
            {"edge_id": "owns2", "src": "acct2", "dst": "user1", "type": "owns"},
        ],
        [n({"type": "user"}, name="u"), e_reverse({"type": "owns"}, name="owns_rev"),
n({"type": "account"}, name="acct")],
    ),
    (
        "two_hop",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
        ],
        [
            {"edge_id": "txn1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "owns1", "src": "acct2", "dst": "user1", "type": "owns"},
            {"edge_id": "owns2", "src": "acct2", "dst": "user2", "type": "owns"},
        ],
        [
            n({"type": "account"}, name="acct_start"),
            e_forward({"type": "txn"}, name="txn"),
            n({"type": "account"}, name="acct_mid"),
            e_forward({"type": "owns"}, name="owns"),
            n({"type": "user"}, name="user_end"),
        ],
    ),
    (
        "undirected",
        [
            {"id": "n1", "type": "node"},
            {"id": "n2", "type": "node"},
            {"id": "n3", "type": "node"},
        ],
        [
            {"edge_id": "e12", "src": "n1", "dst": "n2", "type": "path"},
            {"edge_id": "e23", "src": "n2", "dst": "n3", "type": "path"},
        ],
        [
            n({"type": "node"}, name="start"),
            e_undirected({"type": "path"}, name="hop"),
            n({"type": "node"}, name="end"),
        ],
    ),
    (
        "empty",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
        ],
        [{"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"}],
        [n({"type": "user"}, name="start"), e_forward({"type": "txn"}, name="hop"),
n({"type": "user"}, name="end")],
    ),
    (
        "cycle",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
        ],
        [
            {"edge_id": "e12", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e21", "src": "acct2", "dst": "acct1", "type": "txn"},
        ],
        [
            n({"type": "account"}, name="start"),
            e_forward({"type": "txn"}, name="hop1"),
            n({"type": "account"}, name="mid"),
            e_forward({"type": "txn"}, name="hop2"),
            n({"type": "account"}, name="end"),
        ],
    ),
    (
        "branch",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
            {"id": "acct4", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct1", "dst": "acct3", "type": "txn"},
            {"edge_id": "e3", "src": "acct3", "dst": "acct4", "type": "txn"},
        ],
        [n({"type": "account"}, name="root"), e_forward({"type": "txn"},
name="first_hop"), n({"type": "account"}, name="child")],
    ),
    (
        "forward_labels",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "acct3", "type": "txn"},
        ],
        [
            n({"type": "account"}, name="start"),
            e_forward(
                {"type": "txn"},
                name="hop",
                label_node_hops="node_hop",
                label_edge_hops="edge_hop",
                label_seeds=True,
            ),
            n({"type": "account"}, name="end"),
        ],
    ),
    (
        "reverse_two_hop",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ],
        [
            {"edge_id": "txn1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "owns1", "src": "acct2", "dst": "user1", "type": "owns"},
        ],
        [
            n({"type": "user"}, name="user_end"),
            e_reverse({"type": "owns"}, name="owns_rev"),
            n({"type": "account"}, name="acct_mid"),
            e_reverse({"type": "txn"}, name="txn_rev"),
            n({"type": "account"}, name="acct_start"),
        ],
    ),
]


@pytest.mark.parametrize("_, nodes, edges, ops", CASES, ids=[case[0] for case in CASES])
def test_enumerator_matches_gfql(_, nodes, edges, ops):
    _run_parity_case(nodes, edges, ops)


def test_enumerator_min_max_three_branch_unlabeled():
    nodes = [
        {"id": "a"},
        {"id": "b1"},
        {"id": "c1"},
        {"id": "d1"},
        {"id": "e1"},
        {"id": "b2"},
        {"id": "c2"},
    ]
    edges = [
        {"edge_id": "e1", "src": "a", "dst": "b1"},
        {"edge_id": "e2", "src": "b1", "dst": "c1"},
        {"edge_id": "e3", "src": "c1", "dst": "d1"},
        {"edge_id": "e4", "src": "d1", "dst": "e1"},
        {"edge_id": "e5", "src": "a", "dst": "b2"},
        {"edge_id": "e6", "src": "b2", "dst": "c2"},
    ]
    ops = [
        n({"id": "a"}),
        e_forward(min_hops=3, max_hops=3),
        n(),
    ]
    _run_parity_case(nodes, edges, ops)


# ============================================================================
# TRICKY PARITY TESTS - Exercise edge cases for hop bounds/labels
# ============================================================================


class TestTrickyHopBounds:
    """Test cases designed to catch subtle bugs in hop bounds and label logic."""

    def test_dead_end_branch_pruning(self):
        """min_hops should prune branches that don't reach the minimum.

        Graph:
          a -> b -> c -> d (3 edges, reaches hop 3)
          a -> x           (1 edge, dead end at hop 1)

        With min_hops=2, the a->x branch should be pruned.
        """
        nodes = [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
            {"id": "x"},
        ]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
            {"edge_id": "dead", "src": "a", "dst": "x"},
        ]
        ops = [
            n({"id": "a"}),
            e_forward(min_hops=2, max_hops=3, label_node_hops="hop", label_edge_hops="ehop"),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        # x and dead edge should not be in output
        assert "x" not in set(oracle.nodes["id"])
        assert "dead" not in set(oracle.edges["edge_id"])

    def test_output_slice_vs_traversal_bounds(self):
        """output_min/max should filter output without affecting traversal.

        Graph: a -> b -> c -> d -> e (linear, 4 edges)

        With min_hops=1, max_hops=4, output_min_hops=2, output_max_hops=3:
        - Traversal reaches all nodes
        - Output includes edges at hop 2-3 (e2, e3)
        - Output includes nodes that are endpoints of those edges (b, c, d)
        - Node hop labels only set for nodes within slice (c=2, d=3), others NA
        """
        nodes = [{"id": x} for x in ["a", "b", "c", "d", "e"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
            {"edge_id": "e4", "src": "d", "dst": "e"},
        ]
        ops = [
            n({"id": "a"}),
            e_forward(
                min_hops=1,
                max_hops=4,
                output_min_hops=2,
                output_max_hops=3,
                label_node_hops="hop",
                label_edge_hops="ehop",
            ),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)

        # Edges at hop 2-3 should be in output
        output_edges = set(oracle.edges["edge_id"])
        assert "e1" not in output_edges  # hop 1
        assert "e2" in output_edges      # hop 2
        assert "e3" in output_edges      # hop 3
        assert "e4" not in output_edges  # hop 4

        # Nodes that are endpoints of kept edges
        output_nodes = set(oracle.nodes["id"])
        assert "a" not in output_nodes  # not an endpoint of e2 or e3
        assert "b" in output_nodes      # source of e2
        assert "c" in output_nodes      # dst of e2, source of e3
        assert "d" in output_nodes      # dst of e3
        assert "e" not in output_nodes  # not an endpoint of kept edges

        # Only nodes at hop 2-3 have hop labels
        assert oracle.node_hop_labels is not None
        assert oracle.node_hop_labels.get("c") == 2
        assert oracle.node_hop_labels.get("d") == 3
        assert "b" not in oracle.node_hop_labels  # hop 1, outside slice

    def test_label_seeds_true(self):
        """label_seeds=True should label seed nodes with hop=0."""
        nodes = [{"id": x} for x in ["seed", "b", "c"]]
        edges = [
            {"edge_id": "e1", "src": "seed", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
        ]
        ops = [
            n({"id": "seed"}),
            e_forward(
                min_hops=1,
                max_hops=2,
                label_node_hops="hop",
                label_seeds=True,
            ),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        # Seed should have hop=0
        assert oracle.node_hop_labels is not None
        assert oracle.node_hop_labels.get("seed") == 0
        assert oracle.node_hop_labels.get("b") == 1
        assert oracle.node_hop_labels.get("c") == 2

    def test_label_seeds_false(self):
        """label_seeds=False should not label seed nodes (hop=NA)."""
        nodes = [{"id": x} for x in ["seed", "b", "c"]]
        edges = [
            {"edge_id": "e1", "src": "seed", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
        ]
        ops = [
            n({"id": "seed"}),
            e_forward(
                min_hops=1,
                max_hops=2,
                label_node_hops="hop",
                label_seeds=False,
            ),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        # Seed should NOT have hop label (or not be 0)
        assert oracle.node_hop_labels is not None
        assert "seed" not in oracle.node_hop_labels or oracle.node_hop_labels.get("seed") != 0

    def test_cycle_with_bounds(self):
        """Cycles should handle hop bounds correctly.

        Graph: a -> b -> c -> a (triangle cycle)

        With min_hops=2, max_hops=3, starting at a:
        - Can reach b at hop 1
        - Can reach c at hop 2
        - Can reach a again at hop 3
        """
        nodes = [{"id": x} for x in ["a", "b", "c"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "a"},
        ]
        ops = [
            n({"id": "a"}),
            e_forward(min_hops=2, max_hops=3, label_node_hops="hop", label_edge_hops="ehop"),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        # All nodes should be reachable
        assert set(oracle.nodes["id"]) == {"a", "b", "c"}

    def test_branching_path_lengths(self):
        """Test behavior with branching paths of different lengths.

        Graph:
          a -> b -> c -> d (3 hops to d via long path)
          a -> x -> d      (2 hops to d via short path)

        With min_hops=3, max_hops=3, d is reachable at hop 3 (via the long path).
        Both paths are explored during traversal, since:
        - a->b->c->d: 3 hops - meets min_hops=3 requirement
        - a->x->d: 2 hops - but x and d are still reachable in the graph

        Note: GFQL semantics include all reachable nodes/edges where at least
        one path satisfies the hop bounds. This is a parity test against GFQL.
        """
        nodes = [{"id": x} for x in ["a", "b", "c", "d", "x"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
            {"edge_id": "short1", "src": "a", "dst": "x"},
            {"edge_id": "short2", "src": "x", "dst": "d"},
        ]
        ops = [
            n({"id": "a"}),
            e_forward(min_hops=3, max_hops=3, label_node_hops="hop"),
            n(),
        ]
        # This is a parity test - just verify oracle matches GFQL
        _run_parity_case(nodes, edges, ops, check_hop_labels=True)

    def test_reverse_with_bounds(self):
        """Reverse traversal with bounds should work correctly.

        Graph: a -> b -> c -> d

        Starting at d, e_reverse, min_hops=2, max_hops=2:
        - Reverse traversal: d <- c <- b <- a
        - hop 1: c, hop 2: b, hop 3: a
        - Valid destination: b (at hop 2)
        - All paths to b are included: d->c->b, so c is included as intermediate
        - a is NOT included because it's hop 3 (beyond max_hops=2)
        """
        nodes = [{"id": x} for x in ["a", "b", "c", "d"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
        ]
        ops = [
            n({"id": "d"}),
            e_reverse(min_hops=2, max_hops=2, label_node_hops="hop"),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        output_nodes = set(oracle.nodes["id"])
        # b is reachable at exactly 2 reverse hops (valid destination)
        assert "b" in output_nodes
        # c is included as intermediate node on path to b
        assert "c" in output_nodes
        # a is at hop 3, beyond max_hops, not included
        assert "a" not in output_nodes

    def test_undirected_with_output_slice(self):
        """Undirected traversal with output slicing.

        Graph: a -- b -- c -- d (undirected)

        Starting at b, e_undirected, max_hops=2, output_min_hops=2:
        - Reaches a,c at hop 1
        - Reaches d at hop 2 (from c)
        - Edge e3 (c->d) is at hop 2, so it's kept
        - Output edges: e3
        - Output nodes: endpoints of e3 (c, d)
        - Node d has hop=2 (valid), c has hop=NA (outside slice)
        """
        nodes = [{"id": x} for x in ["a", "b", "c", "d"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
        ]
        ops = [
            n({"id": "b"}),
            e_undirected(max_hops=2, output_min_hops=2, label_node_hops="hop"),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops, check_hop_labels=True)
        output_nodes = set(oracle.nodes["id"])
        output_edges = set(oracle.edges["edge_id"])
        # Only edge e3 (hop 2) is in output
        assert "e3" in output_edges
        assert "e1" not in output_edges  # hop 1
        assert "e2" not in output_edges  # hop 1
        # Nodes: endpoints of kept edge e3
        assert "c" in output_nodes  # source of e3
        assert "d" in output_nodes  # dest of e3
        assert "a" not in output_nodes  # not endpoint of e3

    def test_empty_result_unreachable_bounds(self):
        """When bounds can't be satisfied, result should be empty.

        Graph: a -> b (1 edge)

        With min_hops=5, max_hops=10: nothing is reachable.
        """
        nodes = [{"id": x} for x in ["a", "b"]]
        edges = [{"edge_id": "e1", "src": "a", "dst": "b"}]
        ops = [
            n({"id": "a"}),
            e_forward(min_hops=5, max_hops=10),
            n(),
        ]
        oracle = _run_parity_case(nodes, edges, ops)
        assert oracle.nodes.empty or len(oracle.nodes) == 0
        assert oracle.edges.empty or len(oracle.edges) == 0

    def test_hop_label_uses_shortest_path_not_valid_path(self):
        """Hop labels should use minimum distance across ALL paths, not just valid paths.

        This is a regression test for a bug where hop labeling only considered
        paths that satisfied min_hops, causing incorrect minimum distances.

        Graph:
          a -> b -> c -> d (3 hops to d via long path)
          a -> x -> d      (2 hops to d via short path)

        With min_hops=3, max_hops=3:
        - Only the 3-hop path a->b->c->d satisfies min_hops
        - But node d's minimum hop distance is 2 (via the short path a->x->d)
        - The hop label for d should be 2, NOT 3

        The bug was: only saving paths >= min_hops caused d to get hop=3.
        """
        nodes = [{"id": x} for x in ["a", "b", "c", "d", "x"]]
        edges = [
            {"edge_id": "e1", "src": "a", "dst": "b"},
            {"edge_id": "e2", "src": "b", "dst": "c"},
            {"edge_id": "e3", "src": "c", "dst": "d"},
            {"edge_id": "short1", "src": "a", "dst": "x"},
            {"edge_id": "short2", "src": "x", "dst": "d"},
        ]
        g = (
            CGFull()
            .nodes(pd.DataFrame(nodes), "id")
            .edges(pd.DataFrame(edges), "src", "dst", edge="edge_id")
        )
        ops = [
            n({"id": "a"}),
            e_forward(min_hops=3, max_hops=3, label_node_hops="hop"),
            n(),
        ]

        # Get GFQL result
        gfql_result = g.gfql(ops)
        gfql_nodes = _to_pandas(gfql_result._nodes)
        gfql_node_hops = {
            row["id"]: int(row["hop"])
            for _, row in gfql_nodes.iterrows()
            if pd.notna(row["hop"])
        }

        # d should have hop=2 (minimum distance via short path)
        # even though only the 3-hop path satisfies min_hops
        assert gfql_node_hops.get("d") == 2, (
            f"Node d should have hop=2 (shortest path), got {gfql_node_hops.get('d')}"
        )

        # x should NOT be in output (short path doesn't satisfy min_hops=3)
        assert "x" not in set(gfql_nodes["id"]), (
            "Node x should not be in output (short path doesn't satisfy min_hops)"
        )

        # Now verify oracle matches
        oracle = enumerate_chain(g, ops, caps=OracleCaps(max_nodes=50, max_edges=50))
        oracle_node_hops = oracle.node_hop_labels or {}

        # Oracle should also have d at hop=2
        assert oracle_node_hops.get("d") == 2, (
            f"Oracle: node d should have hop=2, got {oracle_node_hops.get('d')}"
        )

        # Oracle should also exclude x
        assert "x" not in set(oracle.nodes["id"]), (
            "Oracle: node x should not be in output"
        )
