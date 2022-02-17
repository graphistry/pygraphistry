# -*- coding: utf-8 -*-
from typing import List

from .graphBase import GraphBase
from .poset import Poset


class Graph(object):
    """

       The graph is stored in disjoint-sets holding each connected component in C as a list of graph_core objects.

       Attributes:
          C (list[grandalf.graphBase.GraphBase]): list of graph_core components.

       Methods:
          add_vertex(v): add vertex v into the Graph as a new component
          add_edge(e): add edge e and its vertices into the Graph possibly merging the
            associated graph_core components
          get_vertices_count(): see order()
          V(): see graph_core
          E(): see graph_core
          remove_edge(e): remove edge e possibly spawning two new cores
            if the graph_core that contained e gets disconnected.
          remove_vertex(v): remove vertex v and all its edges.
          order(): the order of the graph (number of vertices)
          norm(): the norm of the graph (number of edges)
          deg_min(): the minimum degree of vertices
          deg_max(): the maximum degree of vertices
          deg_avg(): the average degree of vertices
          eps(): the graph epsilon value (norm/order), average number of edges per vertex. 
          connected(): returns True if the graph is connected (i.e. it has only one component).
          components(): returns self.C
    """

    component_class = GraphBase

    def __init__(self, V = None, E = None, directed = True):
        if V is None:
            V = []
        if E is None:
            E = []
        self.directed = directed
        # tag connex set of vertices:
        # at first, every vertex is its own component
        for v in V:
            v.c = Poset([v])
        CV = [v.c for v in V]
        # then pass through edges and union associated vertices such that
        # CV finally holds only connected sets:
        for e in E:
            x = e.v[0]
            y = e.v[1]
            assert x in V
            assert y in V
            assert x.c in CV
            assert y.c in CV
            e.attach()
            if x.c != y.c:
                # merge y.c into x.c :
                x.c.update(y.c)
                # update set list (MUST BE DONE BEFORE UPDATING REFS!)
                CV.remove(y.c)
                # update reference:
                for z in y.c:
                    z.c = x.c
        # now create edge sets from connected vertex sets and
        # make the GraphBase connected graphs for this component :
        self.C = []
        for c in CV:
            s = set()
            for v in c:
                s.update(v.e)
            self.C.append(self.component_class(c, s, directed))

    def add_vertex(self, v):
        for c in self.C:
            if v in c.verticesPoset:
                return c.verticesPoset.get(v)
        g = self.component_class(directed = self.directed)
        v = g.add_single_vertex(v)
        self.C.append(g)
        return v

    def add_edge(self, e):

        x = e.v[0]
        y = e.v[1]
        x = self.add_vertex(x)
        y = self.add_vertex(y)

        cx = x.c
        cy = y.c

        e = cy.add_edge(e)
        # connect (union) the graphs:
        if cx != cy:
            cx.union_update(cy)
            self.C.remove(cy)
        return e

    def add_edges(self, edges: List):
        if edges is None:
            raise ValueError
        for e in edges:
            self.add_edge(e)

    def get_vertices_count(self):
        return sum([c.order() for c in self.C])

    def V(self):
        for c in self.C:
            V = c.verticesPoset
            for v in V:
                yield v

    def E(self):
        for c in self.C:
            E = c.edgesPoset
            for e in E:
                yield e

    def remove_edge(self, e):
        # get the GraphBase:
        c = e.v[0].c
        assert c == e.v[1].c
        if c not in self.C:
            return None
        # remove edge in GraphBase and replace it with two new cores
        # if removing edge disconnects the GraphBase:
        try:
            e = c.remove_edge(e)
        except ValueError:
            e = c.edgesPoset.remove(e)
            e.detach()
            self.C.remove(c)
            tmpg = type(self)(c.verticesPoset, c.edgesPoset, self.directed)
            assert len(tmpg.C) == 2
            self.C.extend(tmpg.C)
        return e

    def remove_vertex(self, x):
        # get the GraphBase:
        c = x.c
        if c not in self.C:
            return None
        try:
            x = c.remove_vertex(x)
            if c.order() == 0:
                self.C.remove(c)
        except ValueError:
            for e in x.detach():
                c.edgesPoset.remove(e)
            x = c.verticesPoset.remove(x)
            self.C.remove(c)
            tmpg = type(self)(c.verticesPoset, c.edgesPoset, self.directed)
            assert len(tmpg.C) == 2
            self.C.extend(tmpg.C)
        return x

    def order(self):
        return sum([c.order() for c in self.C])

    def norm(self):
        return sum([c.norm() for c in self.C])

    def deg_min(self):
        return min([c.deg_min() for c in self.C])

    def deg_max(self):
        return max([c.deg_max() for c in self.C])

    def deg_avg(self):
        t = 0.0
        for c in self.C:
            t += sum([v.degree() for v in c.verticesPoset])
        return t / float(self.order())

    def eps(self):
        return float(self.norm()) / self.order()

    def path(self, x, y, f_io = 0, hook = None):
        if x == y:
            return []
        if x.c != y.c:
            return None
        # path:
        return x.c.path(x, y, f_io, hook)

    def N(self, v, f_io = 0):
        return v.neighbors(f_io)

    def __contains__(self, G):
        r = False
        for c in self.C:
            r |= G in c
        return r

    def connected(self):
        return len(self.C) == 1

    # returns connectivity (kappa)
    def connectivity(self):
        raise NotImplementedError

    # returns edge-connectivity (lambda)
    def e_connectivity(self):
        raise NotImplementedError

    # returns the list of graphs components
    def components(self):
        return self.C

    # derivated graphs:
    # -----------------

    # returns subgraph spanned by vertices V
    def spans(self, V):
        raise NotImplementedError

    # returns join of G (if disjoint)
    def __mul__(self, G):
        raise NotImplementedError

    # returns complement of a graph G
    def complement(self, G):
        raise NotImplementedError

    # contraction G\e
    def contract(self, e):
        raise NotImplementedError
