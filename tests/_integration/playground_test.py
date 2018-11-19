import graphistry
import networkx

from ..pytest_util import skip_if_travis

@skip_if_travis
def test_plot():
    graph = networkx.random_lobster(100, 0.9, 0.9)
    uri = graphistry \
        .settings(
            protocol='http',
            server='nginx'
        ) \
        .data(graph=graph) \
        .plot()

    print(uri)
