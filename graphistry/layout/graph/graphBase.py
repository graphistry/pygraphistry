# DEPRECRATED: Non-vector operators over non-vectorized data

from graphistry.layout.utils import Poset


class GraphBase(object):
    """
        A connected graph of Vertex/Edge objects. A GraphBase is a *component* of a Graph that contains a connected set of Vertex and Edges.

        Attributes:
            verticesPoset (Poset[Vertex]): the partially ordered set of vertices of the graph.
            edgesPoset (Poset[Edge]): the partially ordered set of edges of the graph.
            loops (set[Edge]): the set of *loop* edges (of degree 0).
            directed (bool): indicates if the graph is considered *oriented* or not.

    """

    def __init__(self, vertices = None, edges = None, directed = True):
        if vertices is None:
            vertices = []
        if edges is None:
            edges = []
        self.directed = directed
        self.verticesPoset = Poset(vertices)
        self.edgesPoset = Poset([])

        self.loops = set()

        if len(self.verticesPoset) == 1:
            v = self.verticesPoset[0]
            v.component = self
            for e in v.e:
                e.detach()
            return

        for e in edges:
            x = self.verticesPoset.get(e.v[0])
            y = self.verticesPoset.get(e.v[1])
            if x is None or y is None:
                raise ValueError("unknown Vertex (%s or %s)" % e.v)
            e.v = (x, y)
            if e.degree == 0:
                self.loops.add(e)
            e = self.edgesPoset.add(e)
            e.attach()
            if x.component is None:
                x.component = Poset([x])
            if y.component is None:
                y.component = Poset([y])
            if id(x.component) != id(y.component):
                x, y = (x, y) if len(x.component) > len(y.component) else (y, x)
                x.component.update(y.component)
                for v in y.component:
                    v.component = x.component
            s = x.component
        # check if graph is connected:
        for v in self.vertices():
            if v.component is None or (v.component != s):
                raise ValueError("unconnected Vertex %s" % v.data)
            else:
                v.component = self

    def roots(self):
        """
        returns the list of *roots* (vertices with no inward edges).
        """
        return list(filter(lambda v: len(v.e_in()) == 0, self.verticesPoset))

    def leaves(self):
        """
        returns the list of *leaves* (vertices with no outward edges).
        """
        return list(filter(lambda v: len(v.e_out()) == 0, self.verticesPoset))

    def add_single_vertex(self, v):
        """
        allow a GraphBase to hold a single vertex.
        """
        if len(self.edgesPoset) == 0 and len(self.verticesPoset) == 0:
            v = self.verticesPoset.add(v)
            v.component = self
            return v
        return None

    def add_edge(self, e):
        """
        add edge e. At least one of its vertex must belong to the graph, the other being added automatically.
        """
        if e in self.edgesPoset:
            return self.edgesPoset.get(e)
        x = e.v[0]
        y = e.v[1]
        if not ((x in self.verticesPoset) or (y in self.verticesPoset)):
            raise ValueError("unconnected edge")
        x = self.verticesPoset.add(x)
        y = self.verticesPoset.add(y)
        e.v = (x, y)
        e.attach()
        e = self.edgesPoset.add(e)
        x.component = self
        y.component = self
        if e.degree == 0:
            self.loops.add(e)
        return e

    def remove_edge(self, e):
        """
        remove Edge e, asserting that the resulting graph is still connex.
        """
        if e not in self.edgesPoset:
            return
        e.detach()
        # check if still connected (path is not oriented here):
        if e.degree == 1 and not self.path(e.v[0], e.v[1]):
            # return to initial state by reconnecting everything:
            e.attach()
            # exit with exception!
            raise ValueError(e)
        else:
            e = self.edgesPoset.remove(e)
            if e in self.loops:
                self.loops.remove(e)
            return e

    def remove_vertex(self, x):
        """
        remove Vertex x and all associated edges.
        """
        if x not in self.verticesPoset:
            return
        vertices = x.neighbors()  # get all neighbor vertices to check paths
        edges = x.detach()  # remove the edges from x and neighbors list
        # now we need to check if all neighbors are still connected,
        # and it is sufficient to check if one of them is connected to
        # all others:
        v0 = vertices.pop(0)
        for v in vertices:
            if not self.path(v0, v):
                # repair everything and raise exception if not connected:
                for e in edges:
                    e.attach()
                raise ValueError(x)
        # remove edges and vertex from internal sets:
        for e in edges:
            self.edgesPoset.remove(e)
        x = self.verticesPoset.remove(x)
        x.component = None
        return x

    def constant_function(self, value):
        return lambda x: value

    def vertices(self, cond = None):
        """
        generates an iterator over vertices, with optional filter
        """
        vertices = self.verticesPoset
        if cond is None:
            cond = self.constant_function(True)
        for v in vertices:
            if cond(v):
                yield v

    def edges(self, cond = None):
        """
        generates an iterator over edges, with optional filter
        """
        edges = self.edgesPoset
        if cond is None:
            cond = self.constant_function(True)
        for e in edges:
            if cond(e):
                yield e

    def matrix(self, cond = None):
        """
            This associativity matrix is like the adjacency matrix but antisymmetric. Returns the associativity matrix of the graph component

        :param cond: same a the condition function in vertices().
        :return: array
        """
        from array import array

        mat = []
        for v in self.vertices(cond):
            vec = array("b", [0] * self.order())
            mat.append(vec)
            for e in v.e_in():
                v0 = e.v[0]
                if v0.index == v.index:
                    continue
                vec[v0.index] = -e.w
            for e in v.e_out():
                v1 = e.v[1]
                vec[v1.index] = e.w
        return mat

    def order(self):
        """
        the order of the graph (number of vertices)
        """
        return len(self.verticesPoset)

    def norm(self):
        """
            The size of the edge poset (number of edges).
        """
        return len(self.edgesPoset)

    def deg_min(self):
        """
        the minimum degree of vertices
        """
        return min([v.degree() for v in self.verticesPoset])

    def deg_max(self):
        """
         the maximum degree of vertices
        """
        return max([v.degree() for v in self.verticesPoset])

    def deg_avg(self):
        """
         the average degree of vertices
        """
        return sum([v.degree() for v in self.verticesPoset]) / float(self.order())

    def eps(self):
        """
         the graph epsilon value (norm/order), average number of edges per vertex.
        """
        return float(self.norm()) / self.order()

    def path(self, x, y, f_io = 0, hook = None):
        """
        shortest path between vertices x and y by breadth-first descent, contrained by f_io direction if provided. The path is returned as a list of Vertex objects.
        If a *hook* function is provided, it is called at every vertex added to the path, passing the vertex object as argument.
        """
        assert x in self.verticesPoset
        assert y in self.verticesPoset
        x = self.verticesPoset.get(x)
        y = self.verticesPoset.get(y)
        if x == y:
            return []
        if f_io != 0:
            assert self.directed
        # path:
        p = None
        if hook is None:
            hook = self.constant_function(False)
        # apply hook:
        hook(x)
        # visisted:
        v = {x: None}
        # queue:
        q = [x]
        while (not p) and len(q) > 0:
            c = q.pop(0)
            for n in c.neighbors(f_io):
                if n not in v:
                    hook(n)
                    v[n] = c
                    if n == y:
                        p = [n]
                    q.append(n)
                if p:
                    break
        # now we fill the path p backward from y to x:
        while p and p[0] != x:
            p.insert(0, v[p[0]])
        return p

    def dijkstra(self, x, f_io = 0, hook = None):
        """
        shortest weighted-edges paths between x and all other vertices by dijkstra's algorithm with heap used as priority queue.
        """
        from collections import defaultdict
        from heapq import heappop, heappush

        if x not in self.verticesPoset:
            return None
        if f_io != 0:
            assert self.directed

        # initiate with path to itself...
        v = self.verticesPoset.get(x)

        # D is the returned vector of distances:
        dic = defaultdict(lambda: None)
        dic[v] = 0.0
        L = [(dic[v], v)]
        while len(L) > 0:
            l, u = heappop(L)
            for e in u.e_dir(f_io):
                v = e.v[0] if (u is e.v[1]) else e.v[1]
                dv = l + e.w
                if dic[v] is not None:
                    # check if heap/D needs updating:
                    # ignore if a shorter path was found already...
                    if dv < dic[v]:
                        for i, t in enumerate(L):
                            if t[1] is v:
                                L.pop(i)
                                break
                        dic[v] = dv
                        heappush(L, (dv, v))
                else:
                    dic[v] = dv
                    heappush(L, (dv, v))
        return dic

    def get_scs_with_feedback(self, roots = None):
        """
            Minimum FAS algorithm (feedback arc set) creating a DAG. Returns the set of strongly connected components
            ("scs") by using Tarjan algorithm. These are maximal sets of vertices such that there is a path from each vertex to every other vertex.
            The algorithm performs a DFS from the provided list of root vertices. A cycle is of course a strongly connected component,but a strongly connected component can include several cycles.
            The Feedback Acyclic Set of edge to be removed/reversed is provided by marking the edges with a "feedback" flag.
            Complexity is O(V+E).

        :param roots:
        :return:
        """

        from sys import getrecursionlimit, setrecursionlimit
        from .vertex import Vertex

        limit = getrecursionlimit()
        edge_poset_count = self.norm() + 10
        if edge_poset_count > limit:
            setrecursionlimit(edge_poset_count)

        def visitor(v, coll):
            v.ind = v.ncur
            v.low_link = v.ncur
            Vertex.ncur += 1
            self.stack.append(v)
            v.mark = True
            for e in v.e_out():
                w = e.v[1]
                if w.ind == 0:
                    visitor(w, coll)
                    v.low_link = min(v.low_link, w.low_link)
                elif w.mark:
                    e.feedback = True
                if w in self.stack:
                    v.low_link = min(v.low_link, w.ind)
            if v.low_link == v.ind:
                q = [self.stack.pop()]
                while q[0] != v:
                    q.insert(0, self.stack.pop())
                coll.append(q)
            v.mark = False

        if roots is None:
            roots = self.roots()
        self.stack = []
        flipped_edges = []
        Vertex.ncur = 1
        for v in self.verticesPoset:
            v.ind = 0
        # start exploring tree from roots:
        for v in roots:
            v = self.verticesPoset.get(v)
            if v.ind == 0:
                visitor(v, flipped_edges)
        # now possibly unvisited vertices:
        for v in self.verticesPoset:
            if v.ind == 0:
                visitor(v, flipped_edges)
        # clean up Tarjan-specific data:
        for v in self.verticesPoset:
            del v.ind
            del v.low_link
            del v.mark
        del Vertex.ncur
        del self.stack
        setrecursionlimit(limit)
        return flipped_edges

    def partition(self):
        vertices = self.verticesPoset.copy()
        roots = self.roots()
        for r in roots:
            vertices.remove(r)
        partitions = []
        while len(roots) > 0:
            v = roots.pop(0)
            p = Poset([v])
            nbs = v.neighbors(+1)
            while len(nbs) > 0:
                x = nbs.pop(0)
                if x in p:
                    continue
                if all([(y in p) for y in x.neighbors(-1)]):
                    p.add(x)
                    if x in roots:
                        roots.remove(x)
                    else:
                        vertices.remove(x)
                    nbs.extend(x.neighbors(+1))
                else:
                    if x in vertices:
                        vertices.remove(x)
                        roots.append(x)
            partitions.append(list(p))
        return partitions

    def N(self, v, f_io = 0):
        return v.neighbors(f_io)

    # general graph properties:
    # -------------------------

    # returns True iff
    #  - o is a subgraph of self, or
    #  - o is a vertex in self, or
    #  - o is an edge in self
    def __contains__(self, o):
        try:
            return o.verticesPoset.issubset(self.verticesPoset) and o.edgesPoset.issubset(self.edgesPoset)
        except AttributeError:
            return (o in self.verticesPoset) or (o in self.edgesPoset)

    # merge GraphBase G into self
    def union_update(self, G):
        for v in G.verticesPoset:
            v.component = self
        self.verticesPoset.update(G.verticesPoset)
        self.edgesPoset.update(G.edgesPoset)

    # derivated graphs:
    # -----------------

    # returns subgraph spanned by vertices vertices
    def spans(self, vertices):
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

    def __getstate__(self):
        vertices = [v for v in self.verticesPoset]
        edges = [e for e in self.edgesPoset]
        return (vertices, edges, self.directed)

    def __setstate__(self, state):
        vertices, edges, directed = state
        for e in edges:
            e.v = [vertices[x] for x in e._v]
            del e._v
        GraphBase.__init__(self, vertices, edges, directed)

    def dft(self, start_vertex = None):
        result = []

        if start_vertex is None:
            start_vertex = next(self.vertices())

        def recursive_helper(node):
            result.append(node)
            children = [e.v[1] for e in node.e_out()]
            for child in children:
                if child not in result:
                    recursive_helper(child)

        recursive_helper(start_vertex)
        return result
