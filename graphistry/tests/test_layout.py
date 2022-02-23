from pickle import dumps, loads, HIGHEST_PROTOCOL
import unittest, pytest
import pandas as pd
from graphistry.layout import Edge, Graph, Vertex, EdgeViewer, Layer, Rectangle, GraphBase, SugiyamaLayout, DummyVertex, route_with_splines, route_with_rounded_corners, Poset
from graphistry.compute import ComputeMixin
from graphistry.layouts import LayoutsMixin
from graphistry.plotter import PlotterBase


class LG(LayoutsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        LayoutsMixin.__init__(self, *args, **kwargs)


class LGFull(LayoutsMixin, ComputeMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('LGFull init')
        super(LGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)


def pickler(obj):
    return dumps(obj, HIGHEST_PROTOCOL)


def create_scenario():
    """
    Create something as:
    v4      v0
     *     * *
      *   v1 *
       * *   *
       v5   *
        *  *
        v2
        *
        v3
    """
    E = []
    data_to_vertex = {}

    vertices = []
    for i in range(6):
        data = 'v%s' % (i,)
        v = Vertex(data)
        data_to_vertex[data] = v
        v.view = Rectangle(100, 50)
        vertices.append(v)

    edge = Edge(vertices[0], vertices[1])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[0], vertices[2])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[1], vertices[5])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[2], vertices[3])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[4], vertices[5])
    edge.view = EdgeViewer()
    E.append(edge)

    edge = Edge(vertices[5], vertices[2])
    edge.view = EdgeViewer()
    E.append(edge)

    G = Graph(vertices, E)
    assert len(G.C) == 1
    gr = G.C[0]

    # not needed anymore...
    # r = filter(lambda x: len(x.e_in()) == 0, gr.verticesPoset)
    # if len(r) == 0:
    #    r = [gr.verticesPoset[0]]
    return gr, data_to_vertex


class CustomRankingSugiyamaLayout(SugiyamaLayout):

    def init_ranking(self, initial_ranking):
        """
            The initial ranking of each vertex if passed
        """
        self.initial_ranking = initial_ranking
        assert 0 in initial_ranking
        nblayers = max(initial_ranking.keys()) + 1
        self.layers = [Layer([]) for layer in range(nblayers)]

    def _rank_init(self, unranked):
        assert self.dag

        if not hasattr(self, 'initial_ranking'):
            super()._rank_init(unranked)
        else:
            for rank, vertices in sorted(self.initial_ranking.items()):
                for v in vertices:
                    self.layoutVertices[v].layer = rank
                    # add it to its layer:
                    self.layers[rank].append(v)


class TestLayout(unittest.TestCase):

    def sample_graph1(self):
        v = range(30)
        V = [Vertex(x) for x in map(str, v)]
        D = dict(zip(v, V))
        e = [(0, 5), (0, 29), (0, 6), (0, 20), (0, 4),
             (17, 3), (5, 2), (5, 10), (5, 14), (5, 26), (5, 4), (5, 3),
             (2, 23), (2, 8), (14, 10), (26, 18), (3, 4),
             (23, 9), (23, 24), (10, 27), (18, 13),
             (1, 12), (24, 28), (24, 12), (24, 15),
             (12, 9), (12, 6), (12, 19),
             (6, 9), (6, 29), (6, 25), (6, 21), (6, 13),
             (29, 25), (21, 22),
             (25, 11), (22, 9), (22, 8),
             (11, 9), (11, 16), (8, 20), (8, 16), (15, 16), (15, 27),
             (16, 27),
             (27, 19), (27, 13),
             (19, 9), (13, 20),
             (9, 28), (9, 4), (20, 4),
             (28, 7)
             ]
        E = [Edge(D[x], D[y]) for x, y in e]
        return (V, E)

    def sample_graph2(self):
        v = 'abc'
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['bc', 'ac']
        E = [Edge(D[xy[0]], D[xy[1]]) for xy in e]
        return (V, E)

    def sample_graph3(self):
        v = 'abc'
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'ac', 'bc']
        E = [Edge(D[xy[0]], D[xy[1]]) for xy in e]
        return (V, E)

    def test_longcycle(self):
        vertices = {}
        for x in 'abcdefgh':
            v = Vertex(x)
            vertices[x] = v
        edges = []
        for e in ('ab', 'bc', 'cd', 'de', 'eb', 'bf', 'dg', 'gh', 'fh'):
            edges.append(Edge(vertices[e[0]], vertices[e[1]]))
        g = Graph(vertices.values(), edges)
        layout = SugiyamaLayout(g.C[0])
        layout.init_all()
        assert len(layout.inverted_edges) == 1
        assert layout.inverted_edges[0] == edges[4]
        assert layout.layoutVertices[vertices['e']].layer == 4
        assert sum((v.dummy for v in layout.layoutVertices.values())) == 4
        print([v for v in layout.layoutVertices.values()])

    def test_vertex(self):
        v1 = Vertex()
        assert v1.degree() == 0
        assert len(v1.neighbors()) == 0
        v2 = Vertex("v2")
        assert v2.data == "v2"
        assert v1.e_to(v2) is None
        assert v1.c is None and v2.c is None
        x = pickler(v2)
        y = loads(x)
        assert y.data == "v2"

    def test_edge(self):
        v1 = Vertex("a")
        v2 = Vertex("b")
        e1 = Edge(v1, v2, data = "a->b")
        v1.e = [e1]
        v2.e = [e1]
        assert v1.degree() == v2.degree() == 1
        assert v2 in v1.neighbors()
        assert v1 in v2.neighbors()
        assert len(v1.neighbors(-1)) == 0
        assert len(v2.neighbors(+1)) == 0
        assert v2.e_from(v1) == e1
        x = pickler(e1)
        y = loads(x)
        assert y.w == 1
        assert len(y._v) == 2

    def test_graph_core(self):
        V, E = self.sample_graph2()
        g = GraphBase(V, E)
        assert g.order() == 3

    def test_graph(self):
        v = ('a', 'b', 'c', 'd')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'ac', 'bc', 'cd']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]
        g1 = GraphBase()
        assert g1.order() == 0
        assert g1.norm() == 0
        assert E[0].v[0] == V[0]
        assert V[0].data == "a"
        g1.add_single_vertex(E[0].v[0])
        assert g1.order() == 1
        assert g1.norm() == 0
        g1.add_edge(E[0])
        assert g1.order() == 2
        assert g1.norm() == 1
        assert V[0].c == V[1].c == g1
        g1.add_edge(E[2])
        assert g1.order() == 3
        assert g1.norm() == 2
        assert len(V[1].neighbors()) == 2
        g1.add_edge(E[3])
        assert g1.order() == 4
        assert g1.norm() == 3
        g1.add_edge(E[1])
        assert g1.order() == 4
        assert g1.norm() == 4
        p = g1.path(V[0], V[3])
        assert '->'.join([x.data for x in p]) == 'a->c->d'
        assert len(g1.roots()) == 1
        assert g1.roots()[0] == D['a']
        for v in g1.verticesPoset:
            v.detach()
        # ---------
        g2 = Graph(V, E)
        assert V[2] in g2
        p = g2.path(V[0], V[3])
        assert '->'.join([x.data for x in p]) == 'a->c->d'
        g2.add_edge(Edge(D['d'], D['a'], data = 'da'))

        assert p == g2.path(V[0], V[3], 1)
        x = pickler(g2)
        g3 = loads(x)
        assert len(g3.C) == 1
        assert ''.join([v.data for v in g3.C[0].verticesPoset]) == 'abcd'

    def test_remove(self):
        v1 = Vertex('a')
        v2 = Vertex('b')
        v3 = Vertex('c')
        e1 = Edge(v1, v2)
        e2 = Edge(v1, v3, w = 2)
        g = GraphBase([v1, v2, v3], [e1, e2])
        try:
            cont = False
            g.remove_vertex(v1)
        except ValueError as i:
            cont = True
            assert i.args[0] == v1
        assert cont
        assert v1 in g.verticesPoset
        assert e1 in g.edgesPoset
        assert e2 in g.edgesPoset
        g.remove_vertex(v3)
        assert e2 not in g.edgesPoset
        v4, v5 = Vertex(4), Vertex(5)
        g = Graph([v1, v2, v3, v4], [e1, e2])
        g.add_edge(Edge(v4, v5))
        g.add_edge(Edge(v3, v5))
        assert len(g.C) == 1
        g.remove_vertex(v1)
        assert len(g.C) == 2
        assert ''.join([v.data for v in g.C[0].verticesPoset]) == 'b'
        assert [v.data for v in g.C[1].verticesPoset] == ['c', 4, 5]
        x = pickler(g)
        y = loads(x)
        assert len(y.C) == 2
        assert [v.data for v in y.C[1].verticesPoset] == ['c', 4, 5]

    def test_cycles(self):
        v = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'bc', 'cd', 'be', 'ef', 'ea', 'fg', 'cg', 'gf', 'dh', 'hg', 'hd', 'dc', 'bf']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]
        g1 = GraphBase(V, E)
        scs = g1.get_scs_with_feedback([V[0]])
        assert len(scs) == 3
        assert [v.data for v in scs[0]] == ['g', 'f']
        assert [v.data for v in scs[1]] == ['c', 'd', 'h']
        assert [v.data for v in scs[2]] == ['a', 'b', 'e']

    def test_Matrix(self):
        V, E = self.sample_graph1()
        G = Graph(V, E, directed = True)
        assert len(G.C) == 1
        g = G.C[0]
        M = g.M()
        from numpy import matrix
        M = matrix(M)
        S = M + M.transpose()
        assert S.sum() == 0

    def test_splines(self):
        gr = GraphBase(*self.sample_graph3())
        for v in gr.V():
            v.view = Rectangle(10, 10)
        sug = SugiyamaLayout(gr)
        sug.init_all(roots = [gr.verticesPoset[0]], inverted_edges = [])
        i = 0
        for s in sug.draw_step():
            print('--- step %d ' % i + '-' * 20)
            for v, x in sug.layoutVertices.items():
                print(x, v.view.xy)
            i += 1
        for e in gr.E():
            e.view = EdgeViewer()
        sug.route_edge = route_with_splines
        sug.layout_edges()
        for e in sug.g.E():
            print('edge (%s -> %s) :' % e.v)
            print(e.view._pts)
            print(e.view.splines)

    def test_rounded_corners(self):
        gr = GraphBase(*self.sample_graph3())
        for v in gr.V():
            v.view = Rectangle(10, 10)
        sug = SugiyamaLayout(gr)
        sug.init_all(roots = [gr.verticesPoset[0]], inverted_edges = [])
        i = 0
        for s in sug.draw_step():
            print('--- step %d ' % i + '-' * 20)
            for v, x in sug.layoutVertices.items():
                print(x, v.view.xy)
            i += 1
        for e in gr.E():
            e.view = EdgeViewer()
        sug.route_edge = route_with_rounded_corners
        sug.layout_edges()
        for e in sug.g.E():
            print('edge (%s -> %s) :' % e.v)
            print(e.view._pts)

    @staticmethod
    def get_layer_data(layout):
        layer_data = {}
        for i, layer in enumerate(layout.layers):
            data = layer_data[i] = []
            for v in layer:
                if isinstance(v, DummyVertex):
                    continue
                data.append(v.data)
        return layer_data

    def test_sugiyama_ranking(self):
        gr, data_to_vertex = create_scenario()
        sug = SugiyamaLayout(gr)
        sug.route_edge = route_with_rounded_corners
        sug.init_all()
        # layer 0: v4      v0
        #          \     / |
        # layer 1:   \   v1 |
        #            \ /   |
        # layer 2:    v5   /
        #             \  /
        # layer 3:     v2
        #             |
        # layer 4:     v3
        rank_to_data = self.get_layer_data(sug)
        assert rank_to_data == {
            0: ['v4', 'v0'],
            1: ['v1'],
            2: ['v5'],
            3: ['v2'],
            4: ['v3'],
        }
        sug.layout(iteration_count = 10)

    def test_sugiyama_custom_ranking(self):
        gr, data_to_vertex = create_scenario()
        sug = CustomRankingSugiyamaLayout(gr)
        sug.route_edge = route_with_rounded_corners
        rank_to_data = {
            0: [data_to_vertex['v4'], data_to_vertex['v0']],
            1: [data_to_vertex['v1']],
            2: [data_to_vertex['v5']],
            3: [data_to_vertex['v2']],
            4: [data_to_vertex['v3']],
        }
        sug.init_ranking(rank_to_data)
        sug.init_all()
        # layer 0: v4      v0
        #          \     / |
        # layer 1:   \   v1 |
        #            \ /   |
        # layer 2:    v5   /
        #             \  /
        # layer 3:     v2
        #             |
        # layer 4:     v3
        rank_to_data = self.get_layer_data(sug)
        assert rank_to_data == {
            0: ['v4', 'v0'],
            1: ['v1'],
            2: ['v5'],
            3: ['v2'],
            4: ['v3'],
        }
        sug.layout(iteration_count = 10)

    def test_partitions(self):
        def mapping(b):
            b1, b2 = b
            return Edge(vertices[b1 - 1], vertices[b2 - 1])

        vertices = [Vertex("b%d" % n) for n in range(1, 16)]
        edges = map(mapping, [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (1, 5),
                              (5, 6), (6, 7), (6, 12), (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (10, 11),
                              (12, 13), (13, 14), (14, 13), (14, 15), (15, 6)])
        g = Graph(vertices, edges)
        g = g.C[0]
        P = g.partition()
        assert len(P) == 3
        assert sum([len(p) for p in P]) == g.order()

    def test_sugiyama_custom_ranking2(self):
        gr, data_to_vertex = create_scenario()
        sug = CustomRankingSugiyamaLayout(gr)
        sug.route_edge = route_with_rounded_corners
        rank_to_data = {
            0: [data_to_vertex['v4'], data_to_vertex['v0']],
            1: [data_to_vertex['v5'], data_to_vertex['v1']],
            2: [data_to_vertex['v2']],
            3: [data_to_vertex['v3']],
        }
        try:
            sug.init_ranking(rank_to_data)
            sug.init_all()
        except ValueError as e:
            assert e.message == 'bad ranking'

    def test_poset(self):
        p = Poset()
        assert len(p) == 0
        with self.assertRaises(ValueError):
            p.add(None)
        v = Vertex()
        ret = p.add(v)
        assert ret == v
        p.add(v)
        assert len(p) == 1
        assert v == p.get(v)

        # ordering
        v1 = Vertex()
        v2 = Vertex()
        v3 = Vertex()
        p1 = Poset([v1, v2, v3])
        p2 = Poset([v1, v2])
        assert p2 < p1
        assert p2.issubset(p1)
        assert len(p1.difference(p2)) == 1

    def test_cycles2(self):
        v = ('a', 'b', 'c', 'd')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'bc', 'cd', 'da']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]
        g = Graph(V, E)
        assert SugiyamaLayout.has_cycles(g)

        v = ('a', 'b', 'c', 'd')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'bc', 'cd']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]
        g = Graph(V, E)
        assert not SugiyamaLayout.has_cycles(g)

        # multiple components
        v = ('a', 'b', 'c', 'd', 'e')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'cd', 'de', 'ec']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]
        g = Graph(V, E)
        assert SugiyamaLayout.has_cycles(g)

    def test_data(self):

        v = ('a', 'b', 'c', 'd')
        V = [Vertex(x) for x in v]
        D = dict(zip(v, V))
        e = ['ab', 'ac', 'bc', 'cd']
        E = [Edge(D[xy[0]], D[xy[1]], data = xy) for xy in e]

        g = Graph(V, E)
        assert len(list(g.V())) == len(v)
        assert len(list(g.E())) == len(E)
        assert set(n.data for n in g.V()) == set(v)
        assert set(n.v[0].data + n.v[1].data for n in g.E()) == set(e)

    def test_tree_layout(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').tree_layout()
        print(g._nodes.to_dict(orient = 'records'))
        assert g._nodes.to_dict(orient = 'records') == [
            {'id': 'a', 'level': 0, 'x': 0.0, 'y': 1},
            {'id': 'b', 'level': 1, 'x': 0.0, 'y': 0}]

    def test_tree_layout_mt(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': [], 'd': []}), 's', 'd').tree_layout()
        assert g._edges is None or len(g._edges) == 0

    def test_tree_layout_levels_1(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').tree_layout()
        assert g._nodes.to_dict(orient = 'records') == [
            {'id': 'a', 'level': 0, 'x': 0.0, 'y': 1},
            {'id': 'b', 'level': 1, 'x': 0.0, 'y': 0}]

    def test_tree_layout_levels_1_aliasing(self):
        lg = LGFull()
        g = (lg
             .edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
             .nodes(pd.DataFrame({'n': ['a', 'b'], 'degree': ['x', 'y']}), 'n')
             .tree_layout())
        assert g._nodes.to_dict(orient = 'records') == [
            {'degree': 'x', 'level': 0, 'n': 'a', 'x': 0.0, 'y': 1},
            {'degree': 'y', 'level': 1, 'n': 'b', 'x': 0.0, 'y': 0}]

    def test_tree_layout_cycle_exn(self):
        lg = LGFull()
        with pytest.raises(ValueError):
            lg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd').tree_layout(allow_cycles = False)

    def test_tree_layout_cycle_override(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd').tree_layout(allow_cycles = True)
        assert g._nodes.to_dict(orient = 'records') == [
            {'id': 'a', 'level': 0, 'x': 0.0, 'y': 1},
            {'id': 'b', 'level': 1, 'x': 0.0, 'y': 0}
        ]

    def test_tree_layout_left_chain(self):
        lg = LGFull()
        g = lg.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').tree_layout(allow_cycles = True)
        assert g._nodes.to_dict(orient = 'records') == [
            {'id': 'a', 'level': 0, 'x': 0.0, 'y': 2},
            {'id': 'b', 'level': 1, 'x': 0.0, 'y': 1},
            {'id': 'c', 'level': 2, 'x': 0.0, 'y': 0}
        ]

    def test_tree_layout_center_tree(self):
        lg = LGFull()
        g = (lg
             .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
             .tree_layout(allow_cycles = True, level_align = 'center', width = 100, height = 100))
        assert g._nodes.to_dict(orient = 'records') == [
            {'id': 'a', 'level': 0, 'x': 50.0, 'y': 100},
            {'id': 'b', 'level': 1, 'x': 0.0, 'y': 0},
            {'id': 'c', 'level': 1, 'x': 100.0, 'y': 0}
        ]

    def test_tree_layout_sort_ascending(self):
        lg = LGFull()
        g = (lg
             .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
             .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 'v': [0, 1, 2]}), 'n')
             .tree_layout(level_sort_values_by = 'v'))
        assert g._nodes.to_dict(orient = 'records') == [
            {'level': 0, 'n': 'a', 'v': 0, 'x': 0.5, 'y': 1},
            {'level': 1, 'n': 'b', 'v': 1, 'x': 0.0, 'y': 0},
            {'level': 1, 'n': 'c', 'v': 2, 'x': 1.0, 'y': 0}
        ]

    def test_tree_layout_sort_descending(self):
        lg = LGFull()
        g = (lg
             .edges(pd.DataFrame({'s': ['a', 'a'], 'd': ['b', 'c']}), 's', 'd')
             .nodes(pd.DataFrame({'n': ['a', 'b', 'c'], 'v': [0, 1, 2]}), 'n')
             .tree_layout(level_sort_values_by = 'v', level_sort_values_by_ascending = False))
        assert g._nodes.to_dict(orient = 'records') == [
            {'level': 1, 'n': 'c', 'v': 2, 'x': 1.0, 'y': 0},
            {'level': 1, 'n': 'b', 'v': 1, 'x': 0.0, 'y': 0},
            {'level': 0, 'n': 'a', 'v': 0, 'x': 0.5, 'y': 1}
        ]
