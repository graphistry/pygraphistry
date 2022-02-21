# -*- coding: utf-8 -*-
import typing
from sys import getrecursionlimit, setrecursionlimit

from graphistry.layout import Rectangle, Vertex, GraphBase, size_median, Graph, Edge, DummyVertex
from graphistry.layout.utils import LayoutVertex
from graphistry.layout.utils.layer import Layer
import pandas as pd


class SugiyamaLayout(object):
    """

    The classic Sugiyama layout aka layered layout.

    - See https://en.wikipedia.org/wiki/Layered_graph_drawing
    - Excellent explanation: https://www.youtube.com/watch?v=Z0RGCWxvCxA


    **Attributes**
        - dirvh (int): the current aligment state for alignment policy:
                 dirvh=0 -> dirh=+1, dirv=-1: leftmost upper
                 dirvh=1 -> dirh=-1, dirv=-1: rightmost upper
                 dirvh=2 -> dirh=+1, dirv=+1: leftmost lower
                 dirvh=3 -> dirh=-1, dirv=+1: rightmost lower
        - order_iter (int): the default number of layer placement iterations
        - order_attr (str): set attribute name used for layer ordering
        - xspace (int): horizontal space between vertices in a layer
        - yspace (int): vertical space between layers
        - dw (int): default width of a vertex
        - dh (int): default height of a vertex
        - g (GraphBase): the graph component reference
        - layers (list[sugiyama.layer.Layer]): the list of layers
        - layoutVertices (dict): associate vertex (possibly dummy) with their sugiyama attributes
        - ctrls (dict): associate edge with all its vertices (including dummies)
        - dag (bool): the current acyclic state
        - init_done (bool): True if state is initialized (see init_all).

    **Example**

        ::

            g = nx.generators.connected_watts_strogatz_graph(1000, 2, 0.3)
            # render
            SugiyamaLayout.draw(g)
            # positions
            positions_dictionary = SugiyamaLayout.arrange(g)


    """
    ctrls: typing.Dict[Vertex, LayoutVertex]
    layers: typing.List[Layer]
    yspace: int
    xspace: int

    @property
    def dirvh(self):
        return self.__dirvh

    @dirvh.setter
    def dirvh(self, dirvh):
        assert dirvh in range(4)
        self.__dirvh = dirvh
        self.__dirh, self.__dirv = {0: (1, -1),
                                    1: (-1, -1),
                                    2: (1, 1),
                                    3: (-1, 1)}[dirvh]

    @property
    def dirv(self):
        return self.__dirv

    @dirv.setter
    def dirv(self, dirv):
        assert dirv in (-1, +1)
        dirvh = (dirv + 1) + (1 - self.__dirh) // 2
        self.dirvh = dirvh

    @property
    def dirh(self):
        return self.__dirh

    @dirh.setter
    def dirh(self, dirh):
        assert dirh in (-1, +1)
        dirvh = (self.__dirv + 1) + (1 - dirh) // 2
        self.dirvh = dirvh

    def __init__(self, g: GraphBase):
        self.dirvh = 0
        self.order_iter = 8
        self.order_attr = "pos"
        self.g = g
        self.layers = []
        self.layoutVertices = {}
        """The map from vertex to LayoutVertex."""
        self.ctrls = {}
        self.xspace = 20
        self.yspace = 20
        self.dw = 10
        self.dh = 10
        self.dag = False
        for v in self.g.V():
            # assert hasattr(v, "view")
            if not hasattr(v, "view"):
                v.view = Rectangle()
            self.layoutVertices[v] = LayoutVertex()
        self.dw, self.dh = size_median([v.view for v in self.g.V()])
        self.init_done = False

    def init_all(self, roots = None, inverted_edges = None, optimize = False):
        """
            Initializes the layout algorithm by computing roots (unless provided),
            inverted edges (unless provided), vertices ranks and creates all dummy
            vertices and layers.
             
             Parameters:
                - roots (list[Vertex]): set *root* vertices (layer 0)
                - inverted_edges (list[Edge]): set edges to invert to have a DAG.
                - optimize (bool): optimize ranking if True (default False)
        """
        if self.init_done:
            return
        if roots is None:
            # taking the ones with outgoing edges only for downflow
            roots = [v for v in self.g.verticesPoset if len(v.e_in()) == 0]
        if inverted_edges is None:
            _ = self.g.get_scs_with_feedback(roots)
            inverted_edges = [x for x in self.g.edgesPoset if x.feedback]
        self.inverted_edges = inverted_edges

        self._layer_all(roots, optimize)
        # add dummy vertex/edge for 'long' edges:
        for e in self.g.E():
            self.create_dummies(e)
        # precompute some layers values:
        for layer in self.layers:
            layer.setup(self)
        self.init_done = True

    @staticmethod
    def arrange(obj: typing.Union[pd.DataFrame, Graph], iteration_count = 1.5, source_column = "source", target_column = "target", layout_direction = 0):
        """
        Returns the positions from a Sugiyama layout iteration.

        :param layout_direction:
            - 0: top-to-bottom
            - 1: right-to-left
            - 2: bottom-to-top
            - 3: left-to-right

        :param     obj (nx.Graph|pd.DataFrame|sugiyama.Graph): can be a Sugiyama graph or a Pandas frame.
        :param     iteration_count: increase the value for diminished crossings
        :param     source_column: if a Pandas frame is given, the name of the column with the source of the edges
        :param     target_column: if a Pandas frame is given, the name of the column with the target of the edges

        Returns:
            a dictionary of positions.
        """
        if isinstance(obj, pd.DataFrame):
            gg = SugiyamaLayout.graph_from_pandas(obj, source_column, target_column)
        elif isinstance(obj, Graph):
            gg = obj
        else:
            raise TypeError

        for v in gg.V():
            v.view = Rectangle()

        sug = SugiyamaLayout(gg.C[0])
        sug.init_all()
        sug.layout(iteration_count)
        positions = SugiyamaLayout.get_positions(gg.C[0].verticesPoset, layout_direction)
        return positions

    @staticmethod
    def get_positions(poset, layout_direction = 0):
        if layout_direction == 0:  # top-to-bottom
            positions = {v.data: (v.view.xy[0], -v.view.xy[1]) for v in poset}
        elif layout_direction == 1:  # right-to-left
            positions = {v.data: (-v.view.xy[1], v.view.xy[0]) for v in poset}
        elif layout_direction == 2:  # bottom-to-top
            positions = {v.data: (v.view.xy[0], v.view.xy[1]) for v in poset}
        elif layout_direction == 3:  # left-to-right
            positions = {v.data: (v.view.xy[1], v.view.xy[0]) for v in poset}
        else:
            raise ValueError
        return positions

    @staticmethod
    def graph_from_pandas(df, source_column = "source", target_column = "target"):
        unique_ids = set(df[source_column].unique().tolist()).union(set(df[target_column].unique().tolist()))
        vertex_dic = {id: Vertex(id) for id in unique_ids}
        edges = [Edge(vertex_dic[u], vertex_dic[v]) for u, v in list(zip(df[source_column], df[target_column]))]
        g = Graph(vertex_dic.values(), edges)
        return g

    def layout(self, iteration_count = 1.5, topological_coordinates = False):
        """
            Compute every node coordinates after converging to optimal ordering by N
            rounds, and finally perform the edge routing.

            :param topological_coordinates: whether to use ( [0,1], layer index) coordinates
        """
        while iteration_count > 0.5:
            for (l, mvmt) in self.ordering_step():
                pass
            iteration_count -= 1
        if iteration_count > 0:
            for (l, mvmt) in self.ordering_step(oneway = True):
                pass
        if topological_coordinates:
            self.set_topological_coordinates()
        else:
            self.set_coordinates()
        self.layout_edges()

    def _edge_inverter(self):
        for e in self.inverted_edges:
            x, y = e.v
            e.v = (y, x)
        self.dag = not self.dag
        if self.dag:
            for e in self.g.degenerated_edges:
                e.detach()
                self.g.edgesPoset.remove(e)
        else:
            for e in self.g.degenerated_edges:
                self.g.add_edge(e)

    def _layer_all(self, roots, optimize = False):
        """
        Computes layer of all vertices from the roots onwards.        
        The initial layer is based on precedence relationships, optimal ranking may be derived from network flow (simplex).
        """
        self._edge_inverter()
        r = [x for x in self.g.verticesPoset if (len(x.e_in()) == 0 and x not in roots)]
        self._layer_init(roots + r)
        if optimize:
            self._layer_optimization()
        self._edge_inverter()

    def _layer_init(self, unranked):
        """
        Computes layer of provided unranked list of vertices and all their children.
        A vertex will be assigned a layer when all its inward edges have been scanned.
        When a vertex is assigned a layer, its outward edges are marked scanned.

        :param unranked: List of unassigned layer vertices
        """
        assert self.dag
        scanned = {}
        # set layer of unranked based on its in-edges vertices ranks:
        while len(unranked) > 0:
            coll = []
            for v in unranked:
                self._set_layer(v)
                # mark out-edges has scan-able:
                for e in v.e_out():
                    scanned[e] = True
                # check if out-vertices are layer-able:
                for x in v.neighbors(+1):
                    if not (False in [scanned.get(e, False) for e in x.e_in()]):
                        if x not in coll:
                            coll.append(x)
            unranked = coll

    def _layer_optimization(self):
        """
            Optimizes the layering by pushing long edges toward lower layers as much as possible.
        """
        assert self.dag
        for layer in reversed(self.layers):
            for v in layer:
                gv = self.layoutVertices[v]
                for x in v.neighbors(-1):
                    if all((self.layoutVertices[y].layer >= gv.layer for y in x.neighbors(+1))):
                        gx = self.layoutVertices[x]
                        self.layers[gx.layer].remove(x)
                        gx.layer = gv.layer - 1
                        self.layers[gv.layer - 1].append(x)

    def _set_layer(self, v):
        """
            Sets layer value for vertex v and add it to the corresponding layer.
           The Layer is created if it is the first vertex with this layer.
        """
        assert self.dag
        r = max([self.layoutVertices[x].layer for x in v.neighbors(-1)] + [-1]) + 1
        self.layoutVertices[v].layer = r
        # add it to its layer:
        try:
            self.layers[r].append(v)
        except IndexError:
            assert r == len(self.layers)
            self.layers.append(Layer([v]))

    def dummyctrl(self, r, control_vertices):
        """
        Creates a DummyVertex at layer r inserted in the ctrl dict of the associated edge and layer.

        **Arguments**
            - r (int): layer value
            - ctrl (dict): the edge's control vertices
           
        **Returns**
              sugiyama.DummyVertex : the created DummyVertex.
        """
        dv = DummyVertex(r)
        dv.view.w, dv.view.h = self.dw, self.dh
        self.layoutVertices[dv] = dv
        dv.control_vertices = control_vertices
        control_vertices[r] = dv
        self.layers[r].append(dv)
        return dv

    def create_dummies(self, e):
        """
            Creates and defines all dummy vertices for edge e.
        """
        source, target = e.v
        r0, r1 = self.layoutVertices[source].layer, self.layoutVertices[target].layer
        if r0 > r1:
            assert e in self.inverted_edges
            source, target = target, source
            r0, r1 = r1, r0
        if (r1 - r0) > 1:
            # "dummy vertices" are stored in the edge ctrl dict,
            # keyed by their layer in layers.
            ctrl = self.ctrls[e] = {}
            ctrl[r0] = source
            ctrl[r1] = target
            for r in range(r0 + 1, r1):
                self.dummyctrl(r, ctrl)

    def draw_step(self):
        """
        Iterator that computes all vertices coordinates and edge routing after
        just one step (one layer after the other from top to bottom to top).
        Use it only for "animation" or debugging purpose.
        """
        ostep = self.ordering_step()
        for s in ostep:
            self.set_coordinates()
            self.layout_edges()
            yield s

    def ordering_step(self, oneway = False):
        """iterator that computes all vertices ordering in their layers
           (one layer after the other from top to bottom, to top again unless
           oneway is True).
        """
        self.dirv = -1
        crossings = 0
        layer = None
        for layer in self.layers:
            mvmt = layer.order()
            crossings += mvmt
            yield (layer, mvmt)
        if oneway or (crossings == 0):
            return
        self.dirv = +1
        while layer:
            mvmt = layer.order()
            yield (layer, mvmt)
            layer = layer.nextlayer()

    def set_coordinates(self):
        """
        Computes all vertex coordinates using Brandes & Kopf algorithm.
        See https://www.semanticscholar.org/paper/Fast-and-Simple-Horizontal-Coordinate-Assignment-Brandes-Köpf/69cb129a8963b21775d6382d15b0b447b01eb1f8
        """
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float("infinity")

        # initialize layout vertex
        for layer in self.layers:
            for v in layer:
                self.layoutVertices[v].root = v
                self.layoutVertices[v].align = v
                self.layoutVertices[v].sink = v
                self.layoutVertices[v].shift = inf
                self.layoutVertices[v].X = None
                self.layoutVertices[v].x = [0.0] * 4
        current_h = self.dirvh
        for dirvh in range(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = current_h  # restore it

        # vertical coordinate assigment of all nodes:
        current_y = 0
        for layer in self.layers:
            dY = max([v.view.h / 2.0 for v in layer])
            for v in layer:
                vx = sorted(self.layoutVertices[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1] + vx[2]) / 2.0
                # final xy-coordinates :
                v.view.xy = (avgm, current_y + dY)
            current_y += 2 * dY + self.yspace
        self._edge_inverter()

    def set_topological_coordinates(self):
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float("infinity")
        bounds = [inf, -inf]
        # initialize layout vertex
        for layer in self.layers:
            for v in layer:
                self.layoutVertices[v].root = v
                self.layoutVertices[v].align = v
                self.layoutVertices[v].sink = v
                self.layoutVertices[v].shift = inf
                self.layoutVertices[v].X = None
                self.layoutVertices[v].x = [0.0] * 4
        current_h = self.dirvh
        for dirvh in range(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = current_h  # restore it

        # vertical coordinate assigment of all nodes:
        current_y = 0
        for layer in self.layers:
            dY = 1
            for v in layer:
                vx = sorted(self.layoutVertices[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1] + vx[2]) / 2.0
                bounds[0] = min(avgm, bounds[0])
                bounds[1] = max(avgm, bounds[1])
                v.view.xy = (avgm, current_y + dY)
            current_y += dY
            # rescale
        for layer in self.layers:
            for v in layer:
                v.view.xy = ((v.view.xy[0] - bounds[0]) / (bounds[1] - bounds[0]), v.view.xy[1])
        self._edge_inverter()

    def _detect_alignment_conflicts(self):
        """

        Inner edges are edges between dummy nodes

        - type 0 is regular crossing regular (or sharing vertex)
        - type 1 is inner crossing regular (targeted crossings)
        - type 2 is inner crossing inner (avoided by reduce_crossings phase)

        """
        curvh = self.dirvh  # save current dirvh value
        self.dirvh = 0
        self.conflicts = []
        for layer in self.layers:
            last = len(layer) - 1
            prev = layer.prevlayer()
            if not prev:
                continue
            k0 = 0
            k1_init = len(prev) - 1
            level = 0
            for l1, v in enumerate(layer):
                if not self.layoutVertices[v].dummy:
                    continue
                if l1 == last or v.inner(-1):
                    k1 = k1_init
                    if v.inner(-1):
                        k1 = self.layoutVertices[v.neighbors(-1)[-1]].pos
                    for vl in layer[level: l1 + 1]:
                        for vk in layer.neighbors(vl):
                            k = self.layoutVertices[vk].pos
                            if k < k0 or k > k1:
                                self.conflicts.append((vk, vl))
                    level = l1 + 1
                    k0 = k1
        self.dirvh = curvh  # restore it

    def _coord_vertical_alignment(self):
        """
            Vertical alignment according to current dirvh internal state.
        """
        dirh, dirv = self.dirh, self.dirv
        g = self.layoutVertices
        for layer in self.layers[::-dirv]:
            if not layer.prevlayer():
                continue
            r = None
            for vk in layer[::dirh]:
                for m in layer._median_index(vk):
                    # take the median node in dirv layer:
                    um = layer.prevlayer()[m]
                    # if vk is "free" align it with um's root
                    if g[vk].align is vk:
                        if dirv == 1:
                            vpair = (vk, um)
                        else:
                            vpair = (um, vk)
                        # if vk<->um link is used for alignment
                        if (vpair not in self.conflicts) and (
                                (r is None) or (dirh * r < dirh * m)
                        ):
                            g[um].align = vk
                            g[vk].root = g[um].root
                            g[vk].align = g[vk].root
                            r = m

    def _coord_horizontal_compact(self):
        limit = getrecursionlimit()
        layer_count = len(self.layers) + 10
        if layer_count > limit:
            setrecursionlimit(layer_count)
        dirh, dirv = self.dirh, self.dirv
        g = self.layoutVertices
        sub_layer = self.layers[::-dirv]
        # recursive placement of blocks:
        for layer in sub_layer:
            for v in layer[::dirh]:
                if g[v].root is v:
                    self.__place_block(v)
        setrecursionlimit(limit)
        # mirror all nodes if right-aligned:
        if dirh == -1:
            for layer in sub_layer:
                for v in layer:
                    x = g[v].X
                    if x:
                        g[v].X = -x
        # then assign x-coord of its root:
        inf = float("infinity")
        rb = inf
        for layer in sub_layer:
            for v in layer[::dirh]:
                g[v].x[self.dirvh] = g[g[v].root].X
                rs = g[g[v].root].sink
                s = g[rs].shift
                if s < inf:
                    g[v].x[self.dirvh] += dirh * s
                rb = min(rb, g[v].x[self.dirvh])
        # normalize to 0, and reinit root/align/sink/shift/X
        for layer in self.layers:
            for v in layer:
                # g[v].x[dirvh] -= rb
                g[v].root = g[v].align = g[v].sink = v
                g[v].shift = inf
                g[v].X = None

    def __place_block(self, v):
        g = self.layoutVertices
        if g[v].X is None:
            # every block is initially placed at x=0
            g[v].X = 0.0
            # place block in which v belongs:
            w = v
            while 1:
                j = g[w].pos - self.dirh  # predecessor in layer must be placed
                r = g[w].layer
                if 0 <= j < len(self.layers[r]):
                    wprec = self.layers[r][j]
                    delta = (self.xspace + (wprec.view.w + w.view.w) / 2.0)
                    # take root and place block:
                    u = g[wprec].root
                    self.__place_block(u)
                    # set sink as sink of prec-block root
                    if g[v].sink is v:
                        g[v].sink = g[u].sink
                    if g[v].sink != g[u].sink:
                        s = g[u].sink
                        newshift = g[v].X - (g[u].X + delta)
                        g[s].shift = min(g[s].shift, newshift)
                    else:
                        g[v].X = max(g[v].X, (g[u].X + delta))
                # take next node to align in block:
                w = g[w].align
                # quit if self aligned
                if w is v:
                    break

    def layout_edges(self):
        """
        Basic edge routing applied only for edges with dummy points.
        Enhanced edge routing can be performed by using the apropriate
        *route_with_xxx* functions from :ref:routing_ in the edges' view.
        """
        for e in self.g.E():
            if hasattr(e, "view"):
                coll = []
                if e in self.ctrls:
                    D = self.ctrls[e]
                    r0, r1 = self.layoutVertices[e.v[0]].layer, self.layoutVertices[e.v[1]].layer
                    if r0 < r1:
                        ranks = range(r0 + 1, r1)
                    else:
                        ranks = range(r0 - 1, r1, -1)
                    coll = [D[r].view.xy for r in ranks]
                coll.insert(0, e.v[0].view.xy)
                coll.append(e.v[1].view.xy)
                try:
                    self.route_edge(e, coll)
                except AttributeError:
                    pass
                e.view.setpath(coll)
