import graphistry
import pandas
import json

from os import path

def test_hypergraph():
    graphistry.register(
        protocol = 'http',
        server = 'localhost',
        bolt = { 'uri': 'bolt://localhost:7687' }
    )

    df = pandas \
        .read_csv(
            path.join(path.dirname(__file__), 'hypergraph_test.csv'),
            encoding='utf8'
        )
    
    hg = graphistry.hypergraph(df[:50])
    g = hg['graph']
