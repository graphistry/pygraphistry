import logging
import pandas as pd
import pytest

import graphistry
from graphistry import Plottable
from graphistry.plugins.graphviz import (
    g_to_pgv,
    layout_graphviz,
    render_graphviz
)


try:
    import pygraphviz
    has_pgv = True
except:
    has_pgv = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def chain_g() -> Plottable:
    return graphistry.edges(
        pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e']
        }),
        's', 'd'
    )


@pytest.fixture
def tree_g() -> Plottable:
    return graphistry.edges(
        pd.DataFrame({
            's': ['a', 'a', 'b1', 'c'],
            'd': ['b1', 'b2', 'c', 'd']
        }),
        's', 'd'
    )

@pytest.mark.skipif(not has_pgv, reason="Requires pygraphviz")
class Test_graphviz():

    def test_g_to_pgv(self, chain_g: Plottable) -> None:

        g = chain_g.materialize_nodes()

        pgv = g_to_pgv(g)
        assert pgv is not None
        assert isinstance(pgv, pygraphviz.AGraph)
        assert len(pgv.nodes()) == 5
        assert len(pgv.edges()) == len(chain_g._edges)

    def test_layout_graphviz_simple(self, chain_g: Plottable) -> None:

        g2 = layout_graphviz(chain_g, "dot")
        assert g2._nodes is not None
        assert 'x' in g2._nodes.columns
        assert 'y' in g2._nodes.columns

        n = g2._nodes.sort_values(g2._node, ascending=True)
        logger.debug('ids: %s', n[g2._node].tolist())
        logger.debug('ys: %s', n['y'].tolist())
        for i in range(1, len(n)):
            assert n.iloc[i]['y'] < n.iloc[i - 1]['y']

    def test_layout_graph_attr(self, tree_g: Plottable) -> None:

        g1 = layout_graphviz(tree_g, "dot")
        g2 = layout_graphviz(tree_g, "dot", graph_attr={'ratio': 10})

        g1_h = g1._nodes.y.max() - g1._nodes.y.min()
        g2_h = g2._nodes.y.max() - g2._nodes.y.min()
        assert g1_h * 2 < g2_h

    def test_render(self, tree_g: Plottable) -> None:

        base_path = "/tmp/"
        tree2_g = tree_g.materialize_nodes()

        id_to_shape = {
            'a': 'circle',
            'b1': 'star',
            'b2': 'star',
            'c': None,
            'd': 'circle'
        }
        tree2_g = tree2_g.nodes(lambda g: g._nodes.assign(
            color=g._nodes[g._node].apply(lambda x: 'red' if x in ['a', 'c'] else 'green'),
            shape=g._nodes[g._node].map(id_to_shape)
        ))

        layout_graphviz(
            tree2_g, "dot",
            render_to_disk=True,
            path=f'{base_path}graph.png',
            edge_attr={'color': 'red'},
            node_attr={'color': 'pink'},
            format='png'
        )
        
        import os
        assert os.path.exists(f'{base_path}graph.png')
        assert os.path.getsize(f'{base_path}graph.png') > 0

    def test_plot_static_layout(self, chain_g: Plottable) -> None:
        png = chain_g.plot_static(format='png', max_nodes=100, max_edges=200)
        assert isinstance(png, (bytes, bytearray))
        assert len(png) > 0

    def test_plot_static_reuse_positions(self, chain_g: Plottable) -> None:
        g_with_xy = chain_g.materialize_nodes()
        g_with_xy = g_with_xy.nodes(lambda g: g._nodes.assign(x=range(len(g._nodes)), y=range(len(g._nodes))))
        g_with_xy = g_with_xy.bind(point_x='x', point_y='y')
        svg = g_with_xy.plot_static(format='svg', reuse_layout=True, max_nodes=100, max_edges=200)
        assert b'<svg' in svg

    def test_plot_static_dot(self, chain_g: Plottable, tmp_path) -> None:
        dot_str = chain_g.plot_static(engine='graphviz-dot', reuse_layout=False)
        assert isinstance(dot_str, str)
        assert '->' in dot_str
        dot_path = tmp_path / "graph.dot"
        chain_g.plot_static(engine='graphviz-dot', reuse_layout=True, path=str(dot_path))
        assert dot_path.exists()
        with open(dot_path, 'r', encoding='utf-8') as f:
            text = f.read()
        assert 'pos' in text

    def test_plot_static_mermaid(self, chain_g: Plottable, tmp_path) -> None:
        mermaid = chain_g.plot_static(engine='mermaid-code', reuse_layout=False)
        assert isinstance(mermaid, str)
        assert 'graph LR' in mermaid
        assert '-->' in mermaid
        mmd_path = tmp_path / "graph.mmd"
        chain_g.plot_static(engine='mermaid-code', reuse_layout=True, path=str(mmd_path))
        assert mmd_path.exists()
        with open(mmd_path, 'r', encoding='utf-8') as f:
            text = f.read()
        assert 'graph LR' in text
    def test_render_graphviz_bytes(self, tree_g: Plottable) -> None:

        svg_bytes = render_graphviz(tree_g, "dot", format="svg", max_nodes=100, max_edges=200)
        assert b'<svg' in svg_bytes
