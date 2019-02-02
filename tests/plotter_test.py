import graphistry
import pandas
from .pytest_util import skip_if_travis

@skip_if_travis
def test_pandas_with_edges_and_nodes():
    graphistry.register(
        protocol='http',
        server='nginx'
    )

    graphistry \
        .data(
            edges=pandas.DataFrame({
                's': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 100,
                'd': [1, 2, 3, 4, 5] * 200
            }),
            nodes=pandas.DataFrame({
                'n': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            })
        ) \
        .bind(
            nodeId='n',
            source='s',
            destination='d'
        ) \
        .plot()
