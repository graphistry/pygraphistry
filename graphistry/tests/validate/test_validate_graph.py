from graphistry.validate.validate_graph import validate_graph
import graphistry
import pandas as pd


def test_validate_graph_good():
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': ['a', 'b', 'c'], 'name': ['A', 'B', 'C']}), node='id')
    assert (validate_graph(g) is True)


def test_validate_graph_undefined_nodeid():
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': ['a', 'b', 'c'], 'name': ['A', 'B', 'C']}))
    assert (validate_graph(g) is False)


def test_validate_graph_duplicate_nodeid():
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': ['a','a', 'b', 'c'], 'name': ['A','A2', 'B', 'C']}), node='id')
    assert (validate_graph(g) is False)


def test_validate_graph_missing_nodes():
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}))
    assert (validate_graph(g) is False)


def test_validate_graph_nan_nodes():
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': [None, 'b', 'c'], 'name': ['A', 'B', 'C']}), node='id')
    assert (validate_graph(g) is False)


def test_validate_graph_missing_src_node():
    # Only returns warning
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': ['b', 'c'], 'name': ['B', 'C']}), node='id')
    assert (validate_graph(g) is True)


def test_validate_graph_missing_dst_node():
    # Only returns warning
    g = graphistry.edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd').nodes(
        pd.DataFrame({'id': ['a','b', ], 'name': ['A', 'B']}), node='id')
    assert (validate_graph(g) is True)
