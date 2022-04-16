from bisect import bisect
# DEPRECRATED: Non-vector operators over non-vectorized data


class Layer(list):
    """
    Layer is where Sugiyama layout organises vertices in hierarchical lists.
    The placement of a vertex is done by the Sugiyama class, but it highly relies on
    the *ordering* of vertices in each layer to reduce crossings.
    This ordering depends on the neighbors found in the upper or lower layers.

    Attributes:
        layout (SugiyamaLayout): a reference to the sugiyama layout instance that contains this layer
        upper (Layer): a reference to the *upper* layer (layer-1)
        lower (Layer): a reference to the *lower* layer (layer+1)
        crossings (int) : number of crossings detected in this layer

    Methods:
        setup (layout): set initial attributes values from provided layout
        nextlayer(): returns *next* layer in the current layout's direction parameter.
        prevlayer(): returns *previous* layer in the current layout's direction parameter.
        order(): compute *optimal* ordering of vertices within the layer.
    """

    __r = None
    layout = None
    upper = None
    lower = None
    __x = 1.0
    crossings = None

    def __eq__(self, other):
        return super().__eq__(other)

    def __str__(self):
        s = "<Layer %d" % self.__r
        s += ", len=%d" % len(self)
        xc = self.crossings or "?"
        s += ", crossings=%s>" % xc
        return s

    def setup(self, layout):
        self.layout = layout
        r = layout.layers.index(self)
        self.__r = r
        if len(self) > 1:
            self.__x = 1.0 / (len(self) - 1)
        for i, v in enumerate(self):
            assert layout.layoutVertices[v].layer == r
            layout.layoutVertices[v].pos = i
            layout.layoutVertices[v].bar = i * self.__x
        if r > 0:
            self.upper = layout.layers[r - 1]
        if r < len(layout.layers) - 1:
            self.lower = layout.layers[r + 1]

    def nextlayer(self):
        return self.lower if self.layout.dirv == -1 else self.upper

    def prevlayer(self):
        return self.lower if self.layout.dirv == +1 else self.upper

    def order(self):
        sug = self.layout
        sug._edge_inverter()
        c = self._cross_counting()
        if c > 0:
            for v in self:
                sug.layoutVertices[v].bar = self._meanvalueattr(v)
            # now resort layers l according to bar value:
            self.sort(key = lambda x: sug.layoutVertices[x].bar)
            # reduce & count crossings:
            c = self._ordering_reduce_crossings()
            # assign new position in layer l:
            for i, v in enumerate(self):
                sug.layoutVertices[v].pos = i
                sug.layoutVertices[v].bar = i * self.__x
        sug._edge_inverter()
        self.crossings = c
        return c

    def _meanvalueattr(self, v):
        """
        find new position of vertex v according to adjacency in prevlayer.
        position is given by the mean value of adjacent positions.
        experiments show that meanvalue heuristic performs better than median.
        """
        sug = self.layout
        if not self.prevlayer():
            return sug.layoutVertices[v].bar
        bars = [sug.layoutVertices[x].bar for x in self.neighbors(v)]
        return sug.layoutVertices[v].bar if len(bars) == 0 else float(sum(bars)) / len(bars)

    def _median_index(self, v):
        """
        Fetches the position of vertex v according to adjacency in layer l+dir.
        """
        assert self.prevlayer() is not None
        neighbor_count = self.neighbors(v)
        g = self.layout.layoutVertices
        pos = [g[x].pos for x in neighbor_count]
        lp = len(pos)
        if lp == 0:
            return []
        pos.sort()
        pos = pos[:: self.layout.dirh]
        i, j = divmod(lp - 1, 2)
        return [pos[i]] if j == 0 else [pos[i], pos[i + j]]

    def neighbors(self, v):
        """
        neighbors refer to upper/lower adjacent nodes.
        Note that v.neighbors() provides neighbors of v in the graph, while
        this method provides the Vertex and DummyVertex adjacent to v in the
        upper or lower layer (depending on layout.dirv state).
        """
        assert self.layout.dag
        direction = self.layout.dirv
        layout_vertex = self.layout.layoutVertices[v]
        layer_index = layout_vertex.layer
        if layout_vertex.nvs is not None and direction in layout_vertex.nvs:
            return layout_vertex.nvs[direction]
        else:
            above = [u for u in v.neighbors(0) if self.layout.layoutVertices[u].layer == layer_index - 1]
            below = [u for u in v.neighbors(0) if self.layout.layoutVertices[u].layer == layer_index + 1]
            layout_vertex.nvs = {-1: above, +1: below}
            # if layout_vertex.dummy:
            return layout_vertex.nvs[direction]
        # try:
        #     return layout_vertex.nvs[direction]
        # except AttributeError:
        #     layout_vertex.nvs = {-1: v.neighbors(-1), +1: v.neighbors(+1)}
        #     if layout_vertex.dummy:
        #         return layout_vertex.nvs[direction]
        #     # v is real, v.neighbors are graph neigbors but we need layers neighbors
        #     for d in (-1, +1):
        #         tr = layout_vertex.layer + d
        #         for i, x in enumerate(v.neighbors(d)):
        #             if self.layout.layoutVertices[x].layer == tr:
        #                 continue
        #             e = v.e_with(x)
        #             dum = self.layout.ctrls[e][tr]
        #             layout_vertex.nvs[d][i] = dum
        #     return layout_vertex.nvs[direction]

    def _crossings(self):
        """
        counts (inefficently but at least accurately) the number of
        crossing edges between layer l and l+dirv.
        P[i][j] counts the number of crossings from j-th edge of vertex i.
        The total count of crossings is the sum of flattened P:
        x = sum(sum(P,[]))
        """
        g = self.layout.layoutVertices
        P = []
        for v in self:
            P.append([g[x].pos for x in self.neighbors(v)])
        for i, p in enumerate(P):
            candidates = sum(P[i + 1:], [])
            for j, e in enumerate(p):
                p[j] = len(filter((lambda nx: nx < e), candidates))
            del candidates
        return P

    def _cross_counting(self):
        """
        Implementation of the efficient bilayer cross counting by insert-sort.
        See https://www.semanticscholar.org/paper/Simple-and-Efficient-Bilayer-Cross-Counting-Barth-Jünger/272d73edce86bcfac3c82945042cf6733ad281a0
        """
        g = self.layout.layoutVertices
        P = []
        for v in self:
            P.extend(sorted([g[x].pos for x in self.neighbors(v)]))
        # count inversions in P:
        s = []
        count = 0
        for i, p in enumerate(P):
            j = bisect(s, p)
            if j < i:
                count += i - j
            s.insert(j, p)
        return count

    def _ordering_reduce_crossings(self):
        assert self.layout.dag
        g = self.layout.layoutVertices
        layer_size = len(self)
        X = 0
        for i, j in zip(range(layer_size - 1), range(1, layer_size)):
            vi = self[i]
            vj = self[j]
            ni = [g[v].bar for v in self.neighbors(vi)]
            Xij = Xji = 0
            for nj in [g[v].bar for v in self.neighbors(vj)]:
                x = len([nx for nx in ni if nx > nj])
                Xij += x
                Xji += len(ni) - x
            if Xji < Xij:
                self[i] = vj
                self[j] = vi
                X += Xji
            else:
                X += Xij
        return X
