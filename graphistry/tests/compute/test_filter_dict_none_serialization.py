"""Pins for #1954: a ``None`` filter value must survive GFQL serialization.

``_filter_dict_to_json`` used to drop every entry whose value was ``None``. A filter the
local engine evaluates as "matches nothing" therefore serialized to *no filter at all*,
i.e. "matches everything" -- so ``chain_remote``/``gfql_remote``, saved query JSON, and any
``to_json``/``from_json`` round trip silently returned the whole graph where the in-process
call returned the empty graph.

The oracle here is the in-process answer: whatever ``filter_dict={'x': None}`` means locally,
the wire form must mean the same thing. Assertions are on row counts and id lists rather than
``to_dict('records')`` so that the pandas 3.13 ``None``/``nan`` cell rendering split cannot
make them vacuous.
"""
from typing import Any, Dict, List, Sequence, Tuple

import json
import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject, e, n
from graphistry.compute.chain import Chain


def _graph_with_nulls() -> Plottable:
    nodes = pd.DataFrame({'id': [0, 1, 2], 'x': [None, 'a', 'b']})
    edges = pd.DataFrame({'s': [0, 1], 'd': [1, 2], 'w': [None, 'k']})
    return graphistry.edges(edges, 's', 'd').nodes(nodes, 'id')


def _ids(g: Plottable) -> List[int]:
    return sorted(g._nodes['id'].tolist())


def _edge_pairs(g: Plottable) -> List[Tuple[int, int]]:
    return sorted(zip(g._edges['s'].tolist(), g._edges['d'].tolist()))


def test_node_filter_dict_none_value_is_serialized() -> None:
    assert n({'x': None}).to_json()['filter_dict'] == {'x': None}


def test_node_filter_dict_none_value_among_non_null_is_serialized() -> None:
    assert n({'x': None, 'y': 'a'}).to_json()['filter_dict'] == {'x': None, 'y': 'a'}


@pytest.mark.parametrize('key', ['edge_match', 'source_node_match', 'destination_node_match'])
def test_edge_match_family_none_value_is_serialized(key: str) -> None:
    op = e(**{key: {'w': None}})

    assert op.to_json()[key] == {'w': None}


def test_node_filter_dict_none_value_round_trips_through_json_text() -> None:
    op = n({'x': None})

    revived = ASTNode.from_json(json.loads(json.dumps(op.to_json())))

    assert revived.filter_dict == {'x': None}


@pytest.mark.parametrize('key', ['edge_match', 'source_node_match', 'destination_node_match'])
def test_edge_match_family_none_value_round_trips_through_json_text(key: str) -> None:
    op = e(**{key: {'w': None}})

    revived = ASTEdge.from_json(json.loads(json.dumps(op.to_json())))

    assert getattr(revived, key) == {'w': None}


_MATCH_ANYTHING_WOULD_RETURN_ALL_NODES = 3


@pytest.mark.parametrize(
    'label,ops',
    [
        ('node', [n({'x': None})]),
        ('node_mixed', [n({'x': None, 'y': 'a'})]),
        ('edge_match', [e(edge_match={'w': None})]),
        ('source_node_match', [e(source_node_match={'x': None})]),
        ('destination_node_match', [e(destination_node_match={'x': None})]),
    ],
)
def test_chain_with_none_filter_value_is_round_trip_equivalent(label: str, ops: Sequence[ASTObject]) -> None:
    g = _graph_with_nulls()
    if label == 'node_mixed':
        g = g.nodes(g._nodes.assign(y=['a', 'a', 'a']), 'id')

    local = g.gfql(Chain(ops))
    wire = g.gfql(Chain.from_json(json.loads(json.dumps(Chain(ops).to_json()))))

    assert _ids(wire) == _ids(local)
    assert _edge_pairs(wire) == _edge_pairs(local)
    assert len(local._nodes) != _MATCH_ANYTHING_WOULD_RETURN_ALL_NODES


@pytest.mark.parametrize(
    'label,ops',
    [
        ('node', [n({'x': None})]),
        ('edge_match', [e(edge_match={'w': None})]),
        ('source_node_match', [e(source_node_match={'x': None})]),
        ('destination_node_match', [e(destination_node_match={'x': None})]),
    ],
)
def test_chain_with_none_filter_value_does_not_widen_to_whole_graph(label: str, ops: Sequence[ASTObject]) -> None:
    g = _graph_with_nulls()

    wire = g.gfql(Chain.from_json(json.loads(json.dumps(Chain(ops).to_json()))))

    assert _ids(wire) == []
    assert _edge_pairs(wire) == []


def test_cypher_null_param_lowers_to_a_serializable_filter() -> None:
    from graphistry.compute.gfql.cypher.api import cypher_to_gfql

    g = _graph_with_nulls()
    params: Dict[str, Any] = {'p': None}
    compiled = cypher_to_gfql('MATCH (a {x: $p}) RETURN a', params=params)

    local = g.gfql(compiled)
    wire = g.gfql(Chain.from_json(json.loads(json.dumps(compiled.to_json()))))

    assert _ids(local) == []
    assert _ids(wire) == _ids(local)


@pytest.mark.parametrize(
    'query',
    [
        'MATCH (a {x: null}) RETURN a.id AS i',
        'MATCH (a) WHERE a.x = null RETURN a.id AS i',
        'MATCH (a {x: null})-[r]->(b) RETURN a.id AS i',
        'MATCH (a {x: null}), (b) RETURN a.id AS i',
    ],
)
def test_cypher_null_property_matches_nothing_on_every_pattern_shape(query: str) -> None:
    g = _graph_with_nulls()

    assert len(g.gfql(query)._nodes) == 0
