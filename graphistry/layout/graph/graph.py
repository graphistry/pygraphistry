# -*- coding: utf-8 -*-
# DEPRECRATED: Non-vector operators over non-vectorized data

from typing import List
from .graphBase import GraphBase
from graphistry.layout.utils import Poset
import pandas as np


class Graph(object):
    # """
    #     The graph is stored in disjoint-sets holding each connected component in `components` as a list of graph_core objects.

    #     **Attributes**
    #         C (list[GraphBase]): list of graph_core components.


    #     **add_edge(e):**
    #         add edge e and its vertices into the Graph possibly merging the associated graph_core components

    #     **get_vertices_count():** 
    #         see order()

    #     **vertices():** 
    #         see graph_core
        
    #     **edges():** 
    #         see graph_core

    #     **remove_edge(e):** 
    #         remove edge e possibly spawning two new cores if the graph_core that contained e gets disconnected.

    #     **remove_vertex(v):** 
    #         remove vertex v and all its edges.

    #     **order():** 
    #         the order of the graph (number of vertices)

    #     **norm():** 
    #         the norm of the graph (number of edges)

    #     **deg_min():** 
    #         the minimum degree of vertices
        
    #     **deg_max():** 
    #         the maximum degree of vertices

    #     **deg_avg():** 
    #         the average degree of vertices

    #     **eps():** 
    #         the graph epsilon value (norm/order), average number of edges per vertex.

    #     **connected():** 
    #         returns True if the graph is connected (i.e. it has only one component).

    #     **components():**
    #         returns the list of components
    # """

    component_class = GraphBase

    def __init__(self, vertices = None, edges = None, directed = True):
        if vertices is None:
            vertices = []
        if edges is None:
            edges = []
        self.directed = directed

        for v in vertices:
            v.component = Poset([v])  # at first, every vertex is its own component
        components = [v.component for v in vertices]
        # then pass through edges and union associated vertices such that
        # CV finally holds only connected sets:
        for e in edges:
            x = e.v[0]
            y = e.v[1]
            assert x in vertices
            assert y in vertices
            assert x.component in components
            assert y.component in components
            e.attach()
            if x.component != y.component:
                # merge y.component into x.component :
                x.component.update(y.component)
                # update set list (MUST BE DONE BEFORE UPDATING REFS!)
                components.remove(y.component)
                # update reference:
                for z in y.component:
                    z.component = x.component
        # create the components
        self.components = []
        for vertices in components:
            edge_set = set()
            for v in vertices:
                edge_set.update(v.e)
            self.components.append(self.component_class(vertices, edge_set, directed))

    def add_vertex(self, v):
        """
        add vertex v into the Graph as a new component
        """
        for c in self.components:
            if v in c.verticesPoset:
                return c.verticesPoset.get(v)
        g = self.component_class(directed = self.directed)
        v = g.add_single_vertex(v)
        self.components.append(g)
        print("add vertex v into the Graph as a new component")
        return v

    def add_edge(self, e):
        """
        add edge e and its vertices into the Graph possibly merging the associated graph_core components
        """
        x = e.v[0]
        y = e.v[1]
        x = self.add_vertex(x)
        y = self.add_vertex(y)

        cx = x.component
        cy = y.component

        e = cy.add_edge(e)
        # connect (union) the graphs:
        if cx != cy:
            cx.union_update(cy)
            self.components.remove(cy)
        return e

    def add_edges(self, edges: List):
        if edges is None:
            raise ValueError
        for e in edges:
            self.add_edge(e)

    def get_vertices_count(self):
        return sum([c.order() for c in self.components])

    def get_vertex_from_data(self, data):
        if data is None:
            return None
        for v in self.vertices():
            if v.data is not None and str(v.data) == str(data):
                return v
        return None

    def vertices(self):
        """
         see graph_core
        """
        for c in self.components:
            vertices = c.verticesPoset
            for v in vertices:
                yield v

    def edges(self):
        for c in self.components:
            edges = c.edgesPoset
            for e in edges:
                yield e

    def remove_edge(self, e):
        """
        remove edge e possibly spawning two new cores if the graph_core that contained e gets disconnected.
        """
        # get the GraphBase:
        c = e.v[0].component
        assert c == e.v[1].component
        if c not in self.components:
            return None
        # remove edge in GraphBase and replace it with two new cores
        # if removing edge disconnects the GraphBase:
        try:
            e = c.remove_edge(e)
        except ValueError:
            e = c.edgesPoset.remove(e)
            e.detach()
            self.components.remove(c)
            tmpg = type(self)(c.verticesPoset, c.edgesPoset, self.directed)
            assert len(tmpg.components) == 2
            self.components.extend(tmpg.components)
        return e

    def remove_vertex(self, x):
        """
        remove vertex v and all its edges.
        """
        c = x.component
        if c not in self.components:
            return None
        try:
            x = c.remove_vertex(x)
            if c.order() == 0:
                self.components.remove(c)
        except ValueError:
            for e in x.detach():
                c.edgesPoset.remove(e)
            x = c.verticesPoset.remove(x)
            self.components.remove(c)
            tmpg = type(self)(c.verticesPoset, c.edgesPoset, self.directed)
            assert len(tmpg.components) == 2
            self.components.extend(tmpg.components)
        return x

    def order(self):
        """
        the order of the graph (number of vertices)
        """
        return sum([c.order() for c in self.components])

    def norm(self):
        """
        the norm of the graph (number of edges)
        """
        return sum([c.norm() for c in self.components])

    def deg_min(self):
        """
         the minimum degree of vertices
        """
        return min([c.deg_min() for c in self.components])

    def deg_max(self):
        """
         the maximum degree of vertices
        """
        return max([c.deg_max() for c in self.components])

    def deg_avg(self):
        """
        the average degree of vertices
        """
        t = 0.0
        for c in self.components:
            t += sum([v.degree() for v in c.verticesPoset])
        return t / float(self.order())

    def eps(self):
        """
        the graph epsilon value (norm/order), average number of edges per vertex.
        """
        return float(self.norm()) / self.order()

    def path(self, x, y, f_io = 0, hook = None):
        if x == y:
            return []
        if x.component != y.component:
            return None
        # path:
        return x.component.path(x, y, f_io, hook)

    def N(self, v, f_io = 0):
        return v.neighbors(f_io)

    def __contains__(self, G):
        r = False
        for c in self.components:
            r |= G in c
        return r

    def connected(self):
        """
        returns the list of components
        """
        return len(self.components) == 1
