def to_networkx(g):
    """
        Converts a sugiyama graph to a networkx graph.
    Returns:
        object: A NetworkX graph.
    """
    from networkx import MultiDiGraph

    nxg = MultiDiGraph()
    for v in g.V():
        nxg.add_node(v.data)
    for e in g.E():
        # todo: this leads to issues when the data is more than an id
        nxg.add_edge(e.v[0].data, e.v[1].data)
    return nxg


def from_networkx(G):
    """
        Converts a networkx graph to a sugiyama graph.
    Returns:
        object: A Sugiyama graph.
    """

    from graphistry.layout import Edge, Vertex, Graph

    vertices = []
    data_to_v = {}
    for x in G.nodes():
        vertex = Vertex(x)
        vertices.append(vertex)
        data_to_v[x] = vertex
    E = [Edge(data_to_v[xy[0]], data_to_v[xy[1]], data = xy) for xy in G.edges()]
    g = Graph(vertices, E)
    return g
