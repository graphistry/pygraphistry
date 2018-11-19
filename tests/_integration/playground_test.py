import pytest
import graphistry
import networkx
import pyarrow

def test_plot():
    graph = networkx.random_lobster(100, 0.9, 0.9)
    uri = graphistry\
        .data(graph=graph)\
        .bind(
            node_id = graphistry.plotter.NODE_ID,
            edge_id = graphistry.plotter.EDGE_ID,
            edge_src = graphistry.plotter.EDGE_SRC,
            edge_dst = graphistry.plotter.EDGE_DST
        )\
        .plot()
    print(uri)
