import pyarrow as pa

import graphistry, logging, pandas as pd, pytest, warnings
from graphistry.tests.common import NoAuthTestCase
from graphistry.constants import SRC, DST, NODE
from graphistry.plugins.igraph import SRC_IGRAPH, DST_IGRAPH, compute_algs, compute_igraph, layout_algs, layout_igraph

try:
    import igraph
    has_igraph = True
except:
    has_igraph = False

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


####################


edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)]
names = ["my", "list", "of", "five", "edges"]

edges_sparse = [(2, 3), (3, 4), (6, 2)]
edges_sparse_renamed = [(0, 1), (1, 2), (3, 0)]
names_sparse = ['ab', 'bc', 'da']
nodes_sparse = [2, 3, 4, 6]
nodes_sparse_renamed = [0, 1, 2, 3]
names_sparse_v = ['a', 'b', 'c', 'd']
names_dense_v = ['u0', 'u1', 'a', 'b', 'c', 'u5', 'd']

nodes = [0, 1, 2, 3, 4]
names_v = ["eggs", "spam", "ham", "bacon", "yello"]

edges2_df = pd.DataFrame({
    'a': ['c', 'd', 'a', 'b', 'b'],
    'b': ['d', 'a', 'b', 'c', 'c'],
    'v1': ['cc', 'dd', 'aa', 'bb', 'bb2'],
    'i': [2, 4, 6, 8, 10]
})
nodes2_df = pd.DataFrame({
    'n': ['a', 'c', 'b', 'd'],
    'v': ['aa', 'cc', 'bb', 'dd'],
    'i': [2, 4, 6, 8]
})

edges3_df = pd.DataFrame({
    'a': ['c', 'd', 'a'],
    'b': ['d', 'a', 'b'],
    'v1': ['cc', 'dd', 'aa'],
    'i': [2, 4, 6]
})
nodes3_df = pd.DataFrame({
    'n': ['a', 'b', 'c', 'd'],
    't': [0, 1, 0, 1]
})

edges4_df = pd.DataFrame({
    #no 0
    's': [5, 6, 2, 5, 4, 2, 9, 4, 7, 10],
    'd': [10, 1, 8, 7, 10, 10, 3, 10, 1, 3],
    'w': [5.58851127, 9.12320228, 4.58717668, 6.59665844, 8.62772521,
       2.48654683, 1.4533045 , 4.47252362, 3.38562727, 9.16188751]
})

@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_from_igraph(NoAuthTestCase):

    def test_minimal_edges(self):
        ig = igraph.Graph(edges)
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_minimal_edges_sparse(self):
        ig = igraph.Graph(edges_sparse)
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges_sparse)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges_sparse)
        assert len(g._edges[g._destination].dropna()) == len(edges_sparse)

    def test_minimal_attributed_edges(self):
        ig = igraph.Graph(edges)
        ig.es["name"] = names
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)
        assert (g._edges['name'] == pd.Series(names)).all()

    def test_minimal_attributed_edges_source_target_collision(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = [f"n{x}" for x in range(len(nodes))]
        ig.es["source"] = [f"src_attr_{x}" for x in range(len(edges))]
        ig.es["target"] = [f"dst_attr_{x}" for x in range(len(edges))]

        g = graphistry.from_igraph(ig, load_nodes=False)

        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

        assert '__attr_source_2__' in g._edges.columns
        assert '__attr_target_2__' in g._edges.columns
        assert g._edges['__attr_source_2__'].tolist() == [f"src_attr_{x}" for x in range(len(edges))]
        assert g._edges['__attr_target_2__'].tolist() == [f"dst_attr_{x}" for x in range(len(edges))]

    def test_attributed_edges_collision_with_custom_bindings(self):
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a'],
            'source': ['src_a', 'src_b', 'src_c'],
            'target': ['dst_a', 'dst_b', 'dst_c'],
        })
        g = graphistry.edges(edges_df, 's', 'd').materialize_nodes()

        g2 = g.from_igraph(g.to_igraph())

        assert g2._source == 's' and g2._destination == 'd'
        assert g2._edges.columns.tolist() == ['s', 'd', 'source', 'target']
        assert g2._edges.equals(edges_df)

    def test_attributed_edges_collision_with_occupied_generated_names(self):
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'a'],
            'source': ['src_a', 'src_b', 'src_c'],
            'target': ['dst_a', 'dst_b', 'dst_c'],
            '__attr_source_2__': ['kept_src_2_a', 'kept_src_2_b', 'kept_src_2_c'],
            '__attr_target_2__': ['kept_dst_2_a', 'kept_dst_2_b', 'kept_dst_2_c'],
        })
        g = graphistry.edges(edges_df, 's', 'd').materialize_nodes()

        g2 = g.from_igraph(g.to_igraph())

        assert g2._source == 's' and g2._destination == 'd'
        assert g2._edges.columns.tolist() == edges_df.columns.tolist()
        assert g2._edges.equals(edges_df)

    def test_source_collision_with_destination_named_source_fallback(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names
        ig.es["source"] = [f"src_attr_{x}" for x in range(len(edges))]

        g = graphistry.bind(
            source="source", destination="__attr_source_2__"
        ).from_igraph(ig, load_nodes=False)

        assert g._source == "source"
        assert g._destination == "__attr_source_2__"
        assert g._edges.columns.tolist() == [
            "source", "__attr_source_2__", "__attr_source_3__"
        ]
        assert g._edges["source"].tolist() == ["my", "my", "my", "list", "of"]
        assert g._edges["__attr_source_2__"].tolist() == [
            "list", "of", "five", "of", "edges"
        ]
        assert g._edges["__attr_source_3__"].tolist() == [
            f"src_attr_{x}" for x in range(len(edges))
        ]

    def test_target_collision_with_source_named_target_fallback(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names
        ig.es["target"] = [f"dst_attr_{x}" for x in range(len(edges))]

        g = graphistry.bind(
            source="__attr_target_2__", destination="target"
        ).from_igraph(ig, load_nodes=False)

        assert g._source == "__attr_target_2__"
        assert g._destination == "target"
        assert g._edges.columns.tolist() == [
            "__attr_target_2__", "target", "__attr_target_3__"
        ]
        assert g._edges["__attr_target_2__"].tolist() == ["my", "my", "my", "list", "of"]
        assert g._edges["target"].tolist() == ["list", "of", "five", "of", "edges"]
        assert g._edges["__attr_target_3__"].tolist() == [
            f"dst_attr_{x}" for x in range(len(edges))
        ]

    def test_attributed_edges_collision_with_multiple_occupied_generated_names(self):
        edges_df = pd.DataFrame({
            's': ['a', 'b'],
            'd': ['b', 'a'],
            'source': ['src_a', 'src_b'],
            '__attr_source_2__': ['kept_src_2_a', 'kept_src_2_b'],
            '__attr_source_3__': ['kept_src_3_a', 'kept_src_3_b'],
        })
        g = graphistry.edges(edges_df, 's', 'd').materialize_nodes()

        g2 = g.from_igraph(g.to_igraph())
        g3 = g2.from_igraph(g2.to_igraph())

        assert g2._source == 's' and g2._destination == 'd'
        assert g2._edges.columns.tolist() == edges_df.columns.tolist()
        assert g2._edges.equals(edges_df)
        assert g3._edges.equals(edges_df)

    def test_standalone_import_collision_with_final_binding(self):
        for endpoint in ["source", "target"]:
            with self.subTest(endpoint=endpoint):
                ig = igraph.Graph(edges)
                ig.vs["name"] = names
                attr = "s" if endpoint == "source" else "d"
                ig.es[attr] = [10 + x for x in range(len(edges))]

                g = graphistry.bind(source="s", destination="d").from_igraph(ig, load_nodes=False)
                fallback = f"__attr_{attr}_2__"
                assert g._edges.columns.tolist() == ["s", "d", fallback]
                assert g._edges[fallback].tolist() == [10 + x for x in range(len(edges))]
                assert g._source == "s" and g._destination == "d"

                g2 = g.from_igraph(g.to_igraph())
                assert g2._edges.columns.tolist() == ["s", "d", fallback]
                assert g2._edges.equals(g._edges)

    def test_final_binding_collision_with_occupied_fallback(self):
        edges_df = pd.DataFrame({
            "s": ["a", "b", "c"],
            "d": ["b", "c", "a"],
            "__attr_s_2__": ["kept_a", "kept_b", "kept_c"],
        })
        g = graphistry.edges(edges_df, "s", "d").materialize_nodes()
        ig = g.to_igraph()
        ig.es["s"] = [10, 11, 12]

        g2 = g.from_igraph(ig)
        assert g2._edges.columns.tolist() == ["s", "d", "__attr_s_2__", "__attr_s_3__"]
        assert g2._edges[["s", "d"]].values.tolist() == edges_df[["s", "d"]].values.tolist()
        assert g2._edges["__attr_s_2__"].tolist() == ["kept_a", "kept_b", "kept_c"]
        assert g2._edges["__attr_s_3__"].tolist() == [10, 11, 12]

    def test_collision_round_trip_by_edge_shape(self):
        edge_shapes = {
            "basic": [("a", "b", "src_a", "e1"), ("b", "c", "src_b", "e2")],
            "self-loop-parallel": [("a", "a", "src_a", "self1"), ("a", "b", "src_b", "e2"), ("a", "b", "src_c", "parallel")],
            "permuted": [("b", "a", "src_b", "e1"), ("a", "c", "src_a", "e2")],
        }
        for shape, edge_rows in edge_shapes.items():
            with self.subTest(shape=shape):
                edges_df = pd.DataFrame(edge_rows, columns=["s", "d", "source", "eid"])
                g = graphistry.edges(edges_df, "s", "d", edge="eid").materialize_nodes()

                g2 = g.from_igraph(g.to_igraph())
                assert g2._edges.columns.tolist() == ["s", "d", "source", "eid"]
                assert g2._edges.equals(edges_df)
                assert g2._source == "s" and g2._destination == "d"

    def test_collision_with_large_nullable_attr(self):
        big = 2**53 + 1
        edges_df = pd.DataFrame({
            "s": ["a", "b"],
            "d": ["b", "a"],
            "source": pd.Series([big, None], dtype="UInt64"),
            "eid": pd.Series([1, 2], dtype="Int64"),
        })
        g = graphistry.edges(edges_df, "s", "d", edge="eid").materialize_nodes()

        g2 = g.from_igraph(g.to_igraph())
        assert g2._edges.columns.tolist() == ["s", "d", "source", "eid"]
        assert g2._edges["source"].tolist() == [big, None]

    def test_collision_through_gfql_and_wire_call(self):
        edges_df = pd.DataFrame({
            "s": ["a", "b", "c", "d", "a"],
            "d": ["b", "c", "a", "a", "d"],
            "source": ["a", "b", "c", "d", "a"],
        })
        g = graphistry.edges(edges_df, "s", "d").materialize_nodes()
        chain = [
            {"type": "Call", "function": "compute_igraph", "params": {"alg": "pagerank"}},
            {"type": "Call", "function": "compute_igraph", "params": {"alg": "articulation_points"}},
        ]
        out = g.gfql(chain)
        assert out._edges.columns.tolist() == ["s", "d", "source"]
        assert out._edges.equals(edges_df)
        assert "pagerank" in out._nodes

    def test_node_name_attribute_round_trip(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"], "name": ["A", "B", "C", "D"]})
        edges_df = pd.DataFrame({"s": ["a", "a", "b"], "d": ["b", "c", "c"]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        g2 = g.from_igraph(g.to_igraph())
        assert g2._nodes.columns.tolist() == ["id", "name"]
        assert g2._nodes.equals(nodes_df)
        assert g2._node == "id"

    def test_node_name_attribute_round_trip_without_merge(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]})
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        g2 = g.from_igraph(g.to_igraph(), merge_if_existing=False)
        assert sorted(g2._nodes.columns.tolist()) == ["id", "name"]
        assert g2._nodes.sort_values("id")["name"].tolist() == ["A", "B", "C"]

    def test_node_name_attribute_with_occupied_generated_name(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"], "name": ["A", "B", "C"], "__attr_name_2__": [1, 2, 3],
        })
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        g2 = g.from_igraph(g.to_igraph())
        assert g2._nodes.columns.tolist() == ["id", "name", "__attr_name_2__"]
        assert g2._nodes.equals(nodes_df)

    def test_node_id_column_named_name(self):
        nodes_df = pd.DataFrame({"name": ["a", "b", "c"], "v": [1, 2, 3]})
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        g = graphistry.nodes(nodes_df, "name").edges(edges_df, "s", "d")

        g2 = g.from_igraph(g.to_igraph())
        assert g2._nodes.columns.tolist() == ["name", "v"]
        assert g2._nodes.equals(nodes_df)

    def test_node_name_attribute_kept_with_use_vids(self):
        nodes_df = pd.DataFrame({"id": [0, 1, 2], "name": ["A", "B", "C"]})
        edges_df = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        ig = g.to_igraph(use_vids=True)
        assert ig.vs["name"] == ["A", "B", "C"]
        assert ig.attributes() == []

        g2 = g.from_igraph(ig)
        assert g2._nodes.columns.tolist() == ["id", "name"]
        assert g2._nodes["name"].tolist() == ["A", "B", "C"]

    def test_node_name_attribute_with_node_id_dtypes(self):
        for dtype in ["Int64", "int32", "category"]:
            with self.subTest(dtype=dtype):
                nodes_df = pd.DataFrame({"id": [10, 20, 30, 40], "name": ["a", "b", "c", "d"]})
                nodes_df = nodes_df.astype({"id": dtype})
                edges_df = pd.DataFrame({"s": [10, 20, 30], "d": [20, 30, 40]})
                g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

                g2 = g.compute_igraph("pagerank")
                assert g2._nodes.columns.tolist() == ["id", "name", "pagerank"]
                assert g2._nodes["id"].tolist() == [10, 20, 30, 40]
                assert g2._nodes["name"].tolist() == ["a", "b", "c", "d"]
                assert g2._nodes["id"].dtype == nodes_df["id"].dtype
                assert g2._edges["s"].dtype == edges_df["s"].dtype

    def test_node_name_attribute_through_compute_and_cypher(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c", "d"], "name": ["A", "B", "C", "D"]})
        edges_df = pd.DataFrame({"s": ["a", "a", "b"], "d": ["b", "c", "c"]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        control = graphistry.nodes(nodes_df.rename(columns={"name": "label"}), "id").edges(edges_df, "s", "d")
        expected = control.compute_igraph("pagerank")._nodes.set_index("id")["pagerank"].to_dict()

        g2 = g.compute_igraph("pagerank")
        assert g2._nodes["name"].tolist() == ["A", "B", "C", "D"]
        assert g2._nodes.set_index("id")["pagerank"].to_dict() == pytest.approx(expected)

        out = g.gfql("CALL graphistry.igraph.pagerank() YIELD nodeId, pagerank RETURN nodeId, pagerank")
        assert out._nodes.set_index("nodeId")["pagerank"].to_dict() == pytest.approx(expected)

    def _attr_graph(self, attr, n, id_dtype):
        ids = list(range(10, 10 + 10 * n, 10)) if id_dtype == "int64" else [f"n{i}" for i in range(n)]
        nodes_df = pd.DataFrame({
            "id": pd.Series(ids, dtype=id_dtype),
            attr: pd.Series([f"v{i}" for i in range(n)], dtype="str"),
            "k": pd.Series(range(n), dtype="int64"),
        })
        edges_df = pd.DataFrame({
            "s": pd.Series(ids[:-1], dtype=id_dtype),
            "d": pd.Series(ids[1:], dtype=id_dtype),
        })
        return nodes_df, edges_df, graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

    def test_node_name_attribute_zero_one_and_populated_rows(self):
        for n in [0, 1, 4]:
            for id_dtype in ["str", "int64"]:
                for merge in [True, False]:
                    with self.subTest(n=n, id_dtype=id_dtype, merge=merge):
                        nodes_df, edges_df, g = self._attr_graph("name", n, id_dtype)

                        g2 = g.from_igraph(g.to_igraph(), merge_if_existing=merge)
                        out = g2._nodes.sort_values("id").reset_index(drop=True)[nodes_df.columns]
                        assert sorted(g2._nodes.columns.tolist()) == ["id", "k", "name"]
                        assert out.dtypes.to_dict() == nodes_df.dtypes.to_dict()
                        assert out.equals(nodes_df)
                        assert len(g2._edges) == len(edges_df)
                        assert g2._edges["s"].dtype == edges_df["s"].dtype
                        assert g2._edges["d"].dtype == edges_df["d"].dtype

                g3 = g.compute_igraph("pagerank")
                assert g3._nodes.columns.tolist() == ["id", "name", "k", "pagerank"]
                assert g3._nodes[nodes_df.columns].dtypes.to_dict() == nodes_df.dtypes.to_dict()
                assert g3._nodes["name"].tolist() == nodes_df["name"].tolist()
                assert len(g3._nodes) == n

    def test_node_attribute_control_zero_one_and_populated_rows(self):
        for n in [0, 1, 4]:
            for id_dtype in ["str", "int64"]:
                for merge in [True, False]:
                    with self.subTest(n=n, id_dtype=id_dtype, merge=merge):
                        nodes_df, edges_df, g = self._attr_graph("label", n, id_dtype)

                        g2 = g.from_igraph(g.to_igraph(), merge_if_existing=merge)
                        out = g2._nodes.sort_values("id").reset_index(drop=True)
                        assert out["id"].tolist() == nodes_df["id"].tolist()
                        assert out["label"].tolist() == nodes_df["label"].tolist()
                        assert len(g2._edges) == len(edges_df)

                g3 = g.compute_igraph("pagerank")
                assert g3._nodes["label"].tolist() == nodes_df["label"].tolist()
                assert len(g3._nodes) == n

    def test_node_name_and_edge_endpoint_attributes_round_trip(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"], "__attr_name_2__": [1, 2, 3]})
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "source": ["sa", "sb"], "target": ["ta", "tb"]})
        g = graphistry.nodes(nodes_df, "id").edges(edges_df, "s", "d")

        g2 = g.from_igraph(g.to_igraph())
        assert g2._nodes.equals(nodes_df)
        assert g2._edges[edges_df.columns.tolist()].equals(edges_df)

        g3 = g.compute_igraph("pagerank")
        assert g3._nodes["name"].tolist() == ["A", "B", "C"]
        assert g3._edges["source"].tolist() == ["sa", "sb"]

    def test_minimal_attributed_edges_sparse(self):
        ig = igraph.Graph(edges_sparse)
        ig.es["name"] = names_sparse
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges_sparse)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges_sparse)
        assert len(g._edges[g._destination].dropna()) == len(edges_sparse)
        assert (g._edges['name'] == pd.Series(names_sparse)).all()

    def test_minimal_nodes(self):
        ig = igraph.Graph(edges)
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert len(g._nodes) == len(nodes)
        assert (g._nodes[g._node].sort_values() == pd.Series(nodes)).all()
        assert g._nodes.columns == [ g._node ]
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_minimal_nodes_sparse(self):
        ig = igraph.Graph(edges_sparse)
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert len(g._nodes) == max(nodes_sparse) + 1
        assert len(g._nodes) == len(names_dense_v)
        assert g._nodes[g._node].sort_values().to_list() == list(range(max(nodes_sparse) + 1))
        assert g._nodes.columns == [ g._node ]
        assert len(g._edges) == len(edges_sparse)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges_sparse)
        assert len(g._edges[g._destination].dropna()) == len(edges_sparse)

    def test_minimal_nodes_attributed(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names_v
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert g._node == NODE
        assert len(g._nodes) == len(nodes)
        assert sorted(g._nodes.columns) == sorted([ NODE ])
        assert (g._nodes[g._node].sort_values() == pd.Series(names_v, name=NODE).sort_values()).all()
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_minimal_nodes_attributed_sparse(self):
        ig = igraph.Graph(edges_sparse)
        ig.vs["name"] = names_dense_v
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert g._node == NODE
        assert len(g._nodes) == max(nodes_sparse) + 1
        assert sorted(g._nodes.columns) == sorted([ NODE ])
        assert len(g._nodes) == len(names_dense_v)
        assert g._nodes[g._node].sort_values().to_list() == sorted(names_dense_v)
        assert len(g._edges) == len(edges_sparse)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges_sparse)
        assert len(g._edges[g._destination].dropna()) == len(edges_sparse)

    def test_merge_existing_nodes(self):
        ig = igraph.Graph(edges)
        ig.vs["idx"] = ['a', 'b', 'c', 'd', 'e']
        g = (graphistry
            .nodes(pd.DataFrame({
                'i': nodes,
                'v1': [f'1_{x}' for x in names_v]
            }), 'i')
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
                'i': ['aaa','bbb','ccc','ddd','eee']
            }), 's', 'd', 'i')
        )
        g2 = g.from_igraph(ig)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(['idx', 'i', 'v1'])
        assert sorted(g2._edges.columns) == sorted(g._edges.columns)

    def test_merge_existing_nodes_attributed(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names_v
        ig.vs["idx"] = ['a', 'b', 'c', 'd', 'e']
        g = (graphistry
            .nodes(pd.DataFrame({
                'i': nodes,
                'v1': [f'1_{x}' for x in names_v]
            }), 'i')
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
                'i': ['aaa','bbb','ccc','ddd','eee']
            }), 's', 'd', 'i')
        )
        g2 = g.from_igraph(ig)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(['idx', 'i', 'v1', 'name'])
        assert sorted(g2._edges.columns) == sorted(g._edges.columns)

    def test_merge_existing_edges(self):
        ig = igraph.Graph(edges)
        ig.es["name"] = names
        ig.es["idx"] = ['aa', 'bb', 'cc', 'dd', 'ee']
        g = (graphistry
            .nodes(pd.DataFrame({
                'i': nodes,
                'v1': [f'1_{x}' for x in names_v]
            }), 'i')
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
                'i': ['aaa','bbb','ccc','ddd','eee']
            }), 's', 'd', 'i')
        )
        g2 = g.from_igraph(ig)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(g._nodes.columns)
        assert sorted(g2._edges.columns) == sorted(['s', 'd', 'i', 'idx', 'name'])

    def test_nodes_str_ids(self):
        g = (graphistry
            .nodes(
                pd.DataFrame({
                    'n': ['a', 'b', 'c']
                }), 'n')
            .edges(
                pd.DataFrame({
                    's': ['a', 'b', 'c'],
                    'd': ['b', 'c', 'a']
                }), 's', 'd')
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph()
        ig.vs['spinglass'] = ig.community_spinglass(spins=3).membership
        g2 = g.from_igraph(ig)

        assert len(g2._nodes) == len(g._nodes)
        assert g2._node == g._node
        assert sorted(g2._nodes.columns) == sorted(['n', 'spinglass'])

        assert len(g2._edges) == len(g._edges)
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert sorted(g2._edges.columns) == sorted(['s', 'd'])

    def test_edges_named(self):
        g = graphistry.edges(edges2_df, 'a', 'b').nodes(nodes2_df, 'n')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert len(g2._nodes) == len(g._nodes)
        assert len(g2._edges) == len(g._edges)
        g2n = g2._nodes.sort_values(by='n').reset_index(drop=True)
        assert g2n.equals(pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'v': ['aa', 'bb', 'cc', 'dd'],
            'i': [2, 6, 4, 8]
        }))

    def test_edges_named_without_nodes(self):
        g = graphistry.edges(edges2_df, 'a', 'b')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert len(g2._nodes) == len(nodes2_df)
        assert len(g2._edges) == len(g._edges)
        g2n = g2._nodes.sort_values(by=g2._node).reset_index(drop=True)
        assert g2n.equals(pd.DataFrame({
            g2._node: ['a', 'b', 'c', 'd'],
        }))

@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_to_igraph(NoAuthTestCase):

    def test_minimal_edges(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes
        }))
        assert g2._node == NODE

    def test_sparse_edges_renamed(self):
        g = graphistry.edges(pd.DataFrame([{'s': s, 'd': d} for (s, d) in edges_sparse]), 's', 'd')
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert sorted(g2._nodes[g2._node].to_list()) == sorted(nodes_sparse)
        assert g2._node == NODE

    def test_swizzles_1_none(self):
        g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a'], 'v': ['aa', 'bb']}), 's', 'd')
        ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert g2._edges.equals(g._edges)
        
        gb = g.nodes(pd.DataFrame({'n': ['a', 'b'], 'v': ['aa', 'bb']}), 'n')
        ig = gb.to_igraph()
        gb2 = gb.from_igraph(ig)
        assert gb2._nodes.equals(gb._nodes)

        gc = g.nodes(pd.DataFrame({'n': ['b', 'a'], 'v': ['bb', 'aa']}), 'n')
        ig = gc.to_igraph()
        gc2 = gc.from_igraph(ig)
        assert gc2._nodes.equals(gc._nodes)

        gd = g.materialize_nodes()
        ig = gd.to_igraph()
        gd2 = gd.from_igraph(ig)
        assert gd2._nodes.equals(gd._nodes)

    def test_swizzles_1_none_numeric(self):
        g = graphistry.edges(pd.DataFrame({'s': [0, 1], 'd': [0, 1], 'v': ['aa', 'bb']}), 's', 'd')
        ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert g2._edges.equals(g._edges)
        
        gb = g.nodes(pd.DataFrame({'n': [0, 1], 'v': ['aa', 'bb']}), 'n')
        ig = gb.to_igraph()
        gb2 = gb.from_igraph(ig)
        assert gb2._nodes.equals(gb._nodes)

        gc = g.nodes(pd.DataFrame({'n': [1, 0], 'v': ['bb', 'aa']}), 'n')
        ig = gc.to_igraph()
        gc2 = gc.from_igraph(ig)
        assert gc2._nodes.equals(gc._nodes)

        gd = g.materialize_nodes()
        ig = gd.to_igraph()
        gd2 = gd.from_igraph(ig)
        assert gd2._nodes.equals(gd._nodes)

    def test_swizzles_2_sparse(self):
        g = graphistry.edges(pd.DataFrame({'s': [1, 2], 'd': [1, 2], 'v': ['11', '22']}), 's', 'd')
        ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert g2._edges.equals(g._edges)

        gb = g.nodes(pd.DataFrame({'n': [1, 2], 'v': ['11', '22']}), 'n')
        ig = gb.to_igraph()
        gb2 = gb.from_igraph(ig)
        assert gb2._nodes.equals(gb._nodes)

        gc = g.nodes(pd.DataFrame({'n': [2, 1], 'v': ['22', '11']}), 'n')
        ig = gc.to_igraph()
        gc2 = gc.from_igraph(ig)
        assert gc2._nodes.equals(gc._nodes)

        gd = g.materialize_nodes()
        ig = gd.to_igraph()
        gd2 = gd.from_igraph(ig)
        assert gd2._nodes.equals(gd._nodes)

    def test_swizzles_2_dense(self):
        g = graphistry.edges(pd.DataFrame({'s': [1, 0], 'd': [1, 0], 'v': ['11', '00']}), 's', 'd')
        ig = g.to_igraph()
        g2 = g.from_igraph(ig)
        assert g2._edges.equals(g._edges)

        gb = g.nodes(pd.DataFrame({'n': [1, 0], 'v': ['11', '00']}), 'n')
        ig = gb.to_igraph()
        gb2 = gb.from_igraph(ig)
        assert gb2._nodes.equals(gb._nodes)

        gc = g.nodes(pd.DataFrame({'n': [0, 1], 'v': ['00', '11']}), 'n')
        ig = gc.to_igraph()
        gc2 = gc.from_igraph(ig)
        assert gc2._nodes.equals(gc._nodes)

        gd = g.materialize_nodes()
        ig = gd.to_igraph()
        gd2 = gd.from_igraph(ig)
        assert gd2._nodes.equals(gd._nodes)


    def test_minimal_edges_renamed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 's'
        assert g2._destination == 'd'
        assert g2._edge is None
        assert g2._edges.equals(g._edges)
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes
        }))
        assert g2._node == NODE

    def test_minimal_edges_str(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }).astype(str), 's', 'd')
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 'source'
        assert g2._destination == 'target'
        assert g2._edge is None
        assert g2._edges.rename(columns={'source': 's', 'target': 'd'}).equals(g._edges)
        logger.debug('g2._nodes: %s', g2._nodes)
        logger.debug('g2._nodes dtypes: %s', g2._nodes.dtypes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: pd.Series(nodes).astype(str)
        }))
        assert g2._node == NODE

    def test_nodes(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'names': names_v
        }))
        assert g2._node == NODE

    def test_nodes_renamed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 's'
        assert g2._destination == 'd'
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': nodes,
            'names': names_v
        }))
        assert g2._node == 'n'

    def test_drop_nodes(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        ig = g.to_igraph(include_nodes=False)
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({NODE: nodes}))
        assert g2._node == NODE

    def test_nodes_undirected(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        ig = g.to_igraph(directed=False)
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': nodes,
            'names': names_v
        }))
        assert g2._node == 'n'

    def test_nodes_undirected_str(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': ['a', 'b', 'c'],
                'd': ['b', 'c', 'a'],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c'],
                'names': ['x', 'y', 'z']
            }), 'n')
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph(directed=False)
        logger.debug('ig: %s', ig)
        ig.vs['cluster'] = ig.community_infomap().membership
        g2 = g.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': ['a', 'b', 'c'],
            'names': ['x', 'y', 'z'],
            'cluster': [0, 0, 0]
        }))
        assert g2._node == 'n'

    def test_nodes_undirected_str_attributed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': ['a', 'b', 'c'],
                'd': ['b', 'c', 'a'],
                'v': ['x', 'y', 'z']
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c'],
                'names': ['x', 'y', 'z']
            }), 'n')
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ig = g.to_igraph(directed=False)
        logger.debug('ig: %s', ig)
        ig.vs['cluster'] = ig.community_infomap().membership
        g2 = g.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': ['a', 'b', 'c'],
            'names': ['x', 'y', 'z'],
            'cluster': [0, 0, 0]
        }))
        assert g2._node == 'n'


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_igraph_usage(NoAuthTestCase):

    def test_enrich_with_stat(self):
        g = graphistry.edges(pd.DataFrame({
            's': [x[0] for x in edges],
            'd': [x[1] for x in edges]
        }), 's', 'd')
        ig = g.to_igraph()
        ig.vs['spinglass'] = ig.community_spinglass(spins=3).membership
        g2 = g.from_igraph(ig)
        logger.debug('g2 nodes: %s', g2._nodes)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g2._edges.shape
        assert len(g2._nodes) == len(nodes)
        assert sorted(g2._nodes.columns) == sorted([g2._node, 'spinglass'])

    def test_enrich_with_stat_direct(self):
        g = (
            graphistry.edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges]
            }), 's', 'd')
            .materialize_nodes()
        )
        g2 = g.nodes(g._nodes.assign(
            #TODO this seems unsafe: no guarantee ._nodes order is ig vertices order?
            spinglass=g.to_igraph().community_spinglass(spins=3).membership
        ))
        logger.debug('g2 nodes: %s', g2._nodes)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g2._edges.shape
        assert len(g2._nodes) == len(nodes)
        assert sorted(g2._nodes.columns) == sorted([g2._node, 'spinglass'])


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_igraph_compute(NoAuthTestCase):

    def chain_1_rename(self, alg: str) -> None: 

        g = graphistry.edges(edges3_df, 'a', 'b').materialize_nodes()

        g2 = compute_igraph(g, alg)
        assert alg in g2._nodes

        g3 = compute_igraph(g2, alg, f'{alg}2')
        assert f'{alg}2' in g3._nodes
        assert g2._nodes[alg].equals(g3._nodes[alg])
        assert g2._nodes[alg].equals(g3._nodes[f'{alg}2'])

        g3b = compute_igraph(g2, alg)
        assert alg in g3b._nodes
        assert g3b._nodes.shape == g2._nodes.shape

    def test_chain_1_rename_pagerank(self):
        self.chain_1_rename('pagerank')

    def test_chain_2_rename_articulation_points(self):
        self.chain_1_rename('articulation_points')

    def test_chain_3_seq(self):
        g = graphistry.edges(edges3_df, 'a', 'b').materialize_nodes()
        g2 = compute_igraph(g, 'pagerank')
        g3 = compute_igraph(g2, 'articulation_points')
        assert 'pagerank' in g3._nodes
        assert 'articulation_points' in g3._nodes

    def test_chain_preserves_source_attribute(self):
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'a'],
            'd': ['b', 'c', 'a', 'a', 'd'],
            'source': ['a', 'b', 'c', 'd', 'a'],
        })
        g = graphistry.edges(edges_df, 's', 'd').materialize_nodes()
        original_edges = edges_df.copy()

        g2 = g.compute_igraph('pagerank')
        assert g2._edges.columns.tolist() == ['s', 'd', 'source']
        assert g2._edges.equals(edges_df)

        g3 = g2.compute_igraph('articulation_points')
        assert g3._edges.columns.tolist() == ['s', 'd', 'source']
        assert g3._edges.equals(edges_df)
        assert edges_df.equals(original_edges)
        assert g3._nodes['pagerank'].tolist() == g2._nodes['pagerank'].tolist()
        assert 'articulation_points' in g3._nodes

    def test_chain_4_sparse(self):

        #From https://github.com/graphistry/pygraphistry/pull/513#issuecomment-1784161313

        g = graphistry.edges(edges4_df, 's', 'd').materialize_nodes()
        g2 = g.compute_igraph('articulation_points')
        assert 'articulation_points' in g2._nodes
        g2b = g.compute_igraph('community_optimal_modularity')
        assert 'community_optimal_modularity' in g2b._nodes
        g3 = g2.compute_igraph('community_optimal_modularity')
        assert g3._nodes.community_optimal_modularity.equals(g2b._nodes.community_optimal_modularity)

    def test_all_calls(self):
        overrides = {
            'bipartite_projection': {
                'params': {'which': 0}
            },
            'community_leading_eigenvector': {
                'directed': False
            },
            'community_leiden': {
                'directed': False
            },
            'community_multilevel': {
                'directed': False
            },
            'gomory_hu_tree': {
                'directed': False
            }
        }

        deprecations = [ 'clusters' ]

        skiplist = [ 'eigenvector_centrality' ]

        g = graphistry.edges(edges3_df, 'a', 'b').materialize_nodes()
        for alg in [x for x in compute_algs]:
            if alg not in skiplist:
                opts = overrides[alg] if alg in overrides else {}
                #logger.debug('alg "%s", opts=(%s)', alg, opts)
                if alg in deprecations:
                    with warnings.catch_warnings(record=True) as w:
                        # Cause all warnings to always be triggered.
                        warnings.simplefilter("always")
                        g2 = compute_igraph(g, alg, **opts)
                        assert g2 is not None
                        assert g2._nodes is not None
                        assert g2._edges is not None
                        pa.Table.from_pandas(g2._nodes)
                        pa.Table.from_pandas(g2._edges)
                        #assert len(w) == 1
                        assert issubclass(w[-1].category, DeprecationWarning)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        g2 = compute_igraph(g, alg, **opts)
                        assert g2 is not None
                        assert g2._nodes is not None
                        assert g2._edges is not None
                        pa.Table.from_pandas(g2._nodes)
                        pa.Table.from_pandas(g2._edges)

@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_igraph_layouts(NoAuthTestCase):

    def test_all_calls(self):

        overrides = {
            'bipartite': {
                'params': {'types': 't'}
            },
            #FIXME some reason these complain about unexpected vertex types
            'lgl': None, 'large': None, 'large_graph': None,
            'star': None,

        }

        g = graphistry.edges(edges3_df, 'a', 'b').nodes(nodes3_df, 'n')
        for alg in layout_algs:
            opts = overrides[alg] if alg in overrides else {}
            if opts is None:
                logger.debug('skipping alg "%s"', alg)
                continue
            logger.debug('alg "%s", opts=(%s)', alg, opts)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                g2 = layout_igraph(g, alg, **opts)
            logger.debug('g._edges: %s', g._edges)
            logger.debug('2._edges: %s', g2._edges)
            assert len(g2._nodes) == len(g._nodes)
            assert g2._edges.equals(g._edges)
