from typing import Any, Dict, Literal, Optional, TYPE_CHECKING
import logging
import pandas as pd
  
from graphistry.Plottable import Plottable


if TYPE_CHECKING:
    try:
        from pygraphviz import AGraph
    except:
        pgv: Any = None
else:
    pgv: Any = None


logger = logging.getLogger(__name__)

Prog = Literal[
    "acyclic",
    "ccomps",
    "circo",
    "dot",
    "fdp",
    "gc",
    "gvcolor",
    "gvpr",
    "neato",
    "nop",
    "osage",
    "patchwork",
    "sccmap",
    "sfdp",
    "tred",
    "twopi",
    "unflatten",
]
PROGS = [
    "acyclic",
    "ccomps",
    "circo",
    "dot",
    "fdp",
    "gc",
    "gvcolor",
    "gvpr",
    "neato",
    "nop",
    "osage",
    "patchwork",
    "sccmap",
    "sfdp",
    "tred",
    "twopi",
    "unflatten",
]

Format = Literal[
    "canon",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dia",
    "dot",
    "fig",
    "gd",
    "gd2",
    "gif",
    "hpgl",
    "imap",
    "imap_np",
    "ismap",
    "jpe",
    "jpeg",
    "jpg",
    "mif",
    "mp",
    "pcl",
    "pdf",
    "pic",
    "plain",
    "plain-ext",
    "png",
    "ps",
    "ps2",
    "svg",
    "svgz",
    "vml",
    "vmlz",
    "vrml",
    "vtx",
    "wbmp",
    "xdot",
    "xlib"
]

FORMATS = [
    "canon",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dia",
    "dot",
    "fig",
    "gd",
    "gd2",
    "gif",
    "hpgl",
    "imap",
    "imap_np",
    "ismap",
    "jpe",
    "jpeg",
    "jpg",
    "mif",
    "mp",
    "pcl",
    "pdf",
    "pic",
    "plain",
    "plain-ext",
    "png",
    "ps",
    "ps2",
    "svg",
    "svgz",
    "vml",
    "vmlz",
    "vrml",
    "vtx",
    "wbmp",
    "xdot",
    "xlib"
]

def g_to_pgv(
    g: Plottable,
    directed: bool = True,
    strict: bool = False,
) -> AGraph:

    graph = AGraph(directed=directed, strict=strict)

    for _, row in g._nodes.iterrows():
        graph.add_node(row[g._node], label=str(row[g._node]))


    for _, row in g._edges.iterrows():
        graph.add_edge(row[g._source], row[g._destination], label=str(row[g._source]))

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
    graph_attr: Optional[Dict[str, Any]] = None,
    node_attr: Optional[Dict[str, Any]] = None,
    edge_attr: Optional[Dict[str, Any]] = None,
) -> AGraph:

    graph = g_to_pgv(g, directed, strict)

    if graph_attr is not None:
        for k, v in graph_attr.items():
            graph.graph_attr[k] = v
    if node_attr is not None:
        for k, v in node_attr.items():
            graph.node_attr[k] = v
    if edge_attr is not None:
        for k, v in edge_attr.items():
            graph.edge_attr[k] = v
  
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
    graph_attr: Optional[Dict[str, Any]] = None,
    node_attr: Optional[Dict[str, Any]] = None,
    edge_attr: Optional[Dict[str, Any]] = None,
    skip_styling: bool = False,
    render_to_disk: bool = False,  # unsafe in server settings
    path: Optional[str] = None,
    format: Optional[Format] = None,
) -> Plottable:
    """

    Use graphviz for layout, such as hierarchical trees and directed acycle graphs

    Requires pygraphviz Python bindings and graphviz native libraries to be installed, see https://pygraphviz.github.io/documentation/stable/install.html

    See PROGS for available layout algorithms

    To render image to disk, set render=True

    :param self: Base graph
    :type self: Plottable

    :param prog: Layout algorithm - "dot", "neato", ...
    :type prog: Prog

    :param args: Additional arguments to pass to the graphviz commandline for layout
    :type args: Optional[str]

    :param directed: Whether the graph is directed (True, default) or undirected (False)
    :type directed: bool

    :param strict: Whether the graph is strict (True) or not (False, default)
    :type strict: bool

    :param graph_attr: Graphviz graph attributes
    :type graph_attr: Optional[Dict[str, Any]]

    :param node_attr: Graphviz node attributes
    :type node_attr: Optional[Dict[str, Any]]

    :param edge_attr: Graphviz edge attributes
    :type edge_attr: Optional[Dict[str, Any]]

    :param skip_styling: Whether to skip applying default styling (False, default) or not (True)
    :type skip_styling: bool

    :param render_to_disk: Whether to render the graph to disk (False, default) or not (True)
    :type render_to_disk: bool

    :param path: Path to save the rendered image when render_to_disk=True
    :type path: Optional[str]

    :param format: Format of the rendered image when render_to_disk=True
    :type format: Optional[Format]

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
            g.layout_graphviz('dot', render_to_disk=True, path='graph.png', format='png')
    """

    graph = layout_graphviz_core(self, prog, args, directed, strict, graph_attr, node_attr, edge_attr)

    if render_to_disk:
        # no prog because position already baked into graph
        graph.draw(path=path, format=format)

    g2 = g_with_pgv_layout(self, graph)

    if not skip_styling:
        g3 = pgv_styling(g2)
    else:
        g3 = g2

    return g3
