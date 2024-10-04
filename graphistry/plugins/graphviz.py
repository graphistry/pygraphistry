from typing import Any, Dict, Optional, Set
import logging
import pandas as pd
  
from graphistry.Plottable import Plottable
from graphistry.plugins_types.graphviz_types import (
    AGraph,
    EDGE_ATTRS, FORMATS, GRAPH_ATTRS, NODE_ATTRS, PROGS, UNSANITARY_ATTRS,
    EdgeAttr, Format, GraphAttr, NodeAttr, Prog
)


logger = logging.getLogger(__name__)


############################################


def g_to_pgv(
    g: Plottable,
    directed: bool = True,
    strict: bool = False,
    drop_unsanitary: bool = False
) -> AGraph:

    import pygraphviz as pgv

    assert g._nodes is not None
    assert g._edges is not None

    graph = pgv.AGraph(directed=directed, strict=strict)

    node_attr_cols: Set[NodeAttr] = {
        c
        for c in NODE_ATTRS
        if c in g._nodes.columns
    }
    if drop_unsanitary:
        for c in node_attr_cols:
            if c in UNSANITARY_ATTRS:
                raise ValueError(f"Unsanitary node_attr {c} is not allowed")

    for _, row in g._nodes.iterrows():
        graph.add_node(
            row[g._node],
            **{c: row[c] for c in node_attr_cols if row[c] is not None}
        )

    edge_attr_cols: Set[EdgeAttr] = {
        c
        for c in EDGE_ATTRS
        if c in g._edges.columns
    }
    if drop_unsanitary:
        for d in edge_attr_cols:
            if d in UNSANITARY_ATTRS:
                raise ValueError(f"Unsanitary edge_attr {d} is not allowed")

    for _, row in g._edges.iterrows():
        graph.add_edge(
            row[g._source],
            row[g._destination],
            **{c: row[c] for c in edge_attr_cols if row[c] is not None}
        )

    return graph


def g_with_pgv_layout(g: Plottable, graph: AGraph) -> Plottable:

    node_positions = []
    for node in graph.nodes():
        # Get the position of the node
        pos = node.attr['pos'].split(',')
        x, y = float(pos[0]), float(pos[1])
        node_positions.append({g._node: node, 'x': x, 'y': y})
    positions_df = pd.DataFrame(node_positions)
    nodes_df = positions_df.merge(g._nodes, on=g._node, how='left')

    return g.nodes(nodes_df)

def pgv_styling(g: Plottable) -> Plottable:
    g2 = g.settings(url_params={
        'play': 0,
        'edgeCurvature': 0
    })
    return g2


def layout_graphviz_core(
    g: Plottable,
    prog: Prog = 'dot',
    args: Optional[str] = None,
    directed: bool = True,
    strict: bool = False,
    graph_attr: Optional[Dict[GraphAttr, Any]] = None,
    node_attr: Optional[Dict[NodeAttr, Any]] = None,
    edge_attr: Optional[Dict[EdgeAttr, Any]] = None,
    drop_unsanitary: bool = False,
) -> AGraph:

    graph = g_to_pgv(g, directed, strict, drop_unsanitary)

    if graph_attr is not None:
        for k, v in graph_attr.items():
            if k not in GRAPH_ATTRS:
                raise ValueError(f"Unknown graph_attr {k}, expected one of {GRAPH_ATTRS}")
            if drop_unsanitary and k in UNSANITARY_ATTRS:
                raise ValueError(f"Unsanitary graph_attr {k} is not allowed")
            graph.graph_attr[k] = v
    if node_attr is not None:
        for k2, v in node_attr.items():
            if k2 not in NODE_ATTRS:
                raise ValueError(f"Unknown node_attr {k2}, expected one of {NODE_ATTRS}")
            if drop_unsanitary and k in UNSANITARY_ATTRS:
                raise ValueError(f"Unsanitary node_attr {k2} is not allowed")
            graph.node_attr[k2] = v
    if edge_attr is not None:
        for k3, v in edge_attr.items():
            if k3 not in EDGE_ATTRS:
                raise ValueError(f"Unknown edge_attr {k3}, expected one of {EDGE_ATTRS}")
            if drop_unsanitary and k3 in UNSANITARY_ATTRS:
                raise ValueError(f"Unsanitary edge_attr {k3} is not allowed")
            graph.edge_attr[k3] = v
  
    if prog not in PROGS:
        raise ValueError(f"Unknown prog {prog}, expected one of {PROGS}")

    if args:
        #TODO: Security reasoning
        raise NotImplementedError("NotImplementedError: Passthrough of commandline arguments not implemented")

    graph.layout(prog=prog)

    return graph


def layout_graphviz(
    self: Plottable,
    prog: Prog = 'dot',
    args: Optional[str] = None,
    directed: bool = True,
    strict: bool = False,
    graph_attr: Optional[Dict[GraphAttr, Any]] = None,
    node_attr: Optional[Dict[NodeAttr, Any]] = None,
    edge_attr: Optional[Dict[EdgeAttr, Any]] = None,
    skip_styling: bool = False,
    render_to_disk: bool = False,  # unsafe in server settings
    path: Optional[str] = None,
    format: Optional[Format] = None,
    drop_unsanitary: bool = False,
) -> Plottable:
    """

    Use graphviz for layout, such as hierarchical trees and directed acycle graphs

    Requires pygraphviz Python bindings and graphviz native libraries to be installed, see https://pygraphviz.github.io/documentation/stable/install.html

    See PROGS for available layout algorithms

    To render image to disk, set render=True

    :param self: Base graph
    :type self: Plottable

    :param prog: Layout algorithm - "dot", "neato", ...
    :type prog: :py:data:`graphistry.plugins_types.graphviz_types.Prog`

    :param args: Additional arguments to pass to the graphviz commandline for layout
    :type args: Optional[str]

    :param directed: Whether the graph is directed (True, default) or undirected (False)
    :type directed: bool

    :param strict: Whether the graph is strict (True) or not (False, default)
    :type strict: bool

    :param graph_attr: Graphviz graph attributes, see https://graphviz.org/docs/graph/
    :type graph_attr: Optional[Dict[:py:data:`graphistry.plugins_types.graphviz_types.GraphAttr`, Any]]

    :param node_attr: Graphviz node attributes, see https://graphviz.org/docs/nodes/
    :type node_attr: Optional[Dict[:py:data:`graphistry.plugins_types.graphviz_types.NodeAttr`, Any]]

    :param edge_attr: Graphviz edge attributes, see https://graphviz.org/docs/edges/
    :type edge_attr: Optional[Dict[:py:data:`graphistry.plugins_types.graphviz_types.EdgeAttr`, Any]]

    :param skip_styling: Whether to skip applying default styling (False, default) or not (True)
    :type skip_styling: bool

    :param render_to_disk: Whether to render the graph to disk (False, default) or not (True)
    :type render_to_disk: bool

    :param path: Path to save the rendered image when render_to_disk=True
    :type path: Optional[str]

    :param format: Format of the rendered image when render_to_disk=True
    :type format: Optional[:py:data:`graphistry.plugins_types.graphviz_types.Format`]

    :param drop_unsanitary: Whether to drop unsanitary attributes (False, default) or not (True), recommended for sensitive settings
    :type drop_unsanitary: bool

    :return: Graph with layout and style settings applied, setting x/y
    :rtype: Plottable


    **Example: Dot layout for rigid hierarchical layout of trees and directed acyclic graphs**
        ::

            import graphistry
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g.layout_graphviz('dot').plot()

    **Example: Neato layout for organic layout of small graphs**

        ::

            import graphistry
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g.layout_graphviz('neato').plot()

    **Example: Set graphviz attributes at graph level**

        ::

            import graphistry
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g.layout_graphviz(
                prog='dot',
                graph_attr={
                    'ratio': 10
                }
            ).plot()

    **Example: Save rendered image to disk as a png**

        ::

            import graphistry
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g.layout_graphviz(
                'dot',
                render_to_disk=True,
                path='graph.png',
                format='png'
            )

    **Example: Save rendered image to disk as a png with passthrough of rendering styles**

        ::

            import graphistry
            edges = pd.DataFrame({
                's': ['a','b','c','d'],
                'd': ['b','c','d','e'],
                'color': ['red', None, None, 'yellow']
            })
            nodes = pd.DataFrame({
                'n': ['a','b','c','d','e'],
                'shape': ['circle', 'square', None, 'square', 'circle']
            })
            g = graphistry.edges(edges, 's', 'd')
            g.layout_graphviz(
                'dot',
                render_to_disk=True,
                path='graph.png',
                format='png'
            )

    """

    g = self
    assert g is not None
    assert g._edges is not None
    if g._nodes is None:
        g = g.materialize_nodes()
        assert g._nodes is not None

    graph = layout_graphviz_core(g, prog, args, directed, strict, graph_attr, node_attr, edge_attr, drop_unsanitary)

    if render_to_disk:
        if format not in FORMATS:
            raise ValueError(f"Unknown format {format}, expected one of {FORMATS}")
        # no prog because position already baked into graph
        graph.draw(path=path, format=format)

    g2 = g_with_pgv_layout(g, graph)

    if not skip_styling:
        g3 = pgv_styling(g2)
    else:
        g3 = g2

    return g3
