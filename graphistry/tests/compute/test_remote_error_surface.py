"""A user of the remote GFQL/Python APIs gets a typed GFQL error, never raw plumbing."""
import io
import json
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest
import requests

import graphistry
from graphistry.compute.ast import n
from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.exceptions import GFQLRemoteError, GFQLSchemaError, GFQLSyntaxError, GFQLTypeError
from graphistry.compute.predicates.numeric import gt
from graphistry.compute.python_remote import python_remote_generic


TASK = 'def task(g):\n    return g\n'
CREDS = {'api_token': 'tok', 'dataset_id': 'ds-1'}

NODES = pd.DataFrame({'id': [0, 1], 'x': ['a', 'b']})
EDGES = pd.DataFrame({'s': [0], 'd': [1]})


def resp(status: int, content: bytes, ctype: str) -> requests.Response:
    r = requests.models.Response()
    r.status_code = status
    r._content = content
    r.headers['content-type'] = ctype
    r.url = 'https://t/x'
    return r


def bound_graph():
    g = graphistry.edges(EDGES, 's', 'd').nodes(NODES, 'id')
    g._dataset_id = 'ds-1'
    return g


class Transport:
    """Patches requests at Session.send so real body preparation and real
    Response decoding both run."""

    def __init__(self, response=None):
        self.response = response
        self.bodies = []

    def __enter__(self):
        outer = self

        def send(self, request, **kwargs):
            body = request.body
            if isinstance(body, (bytes, bytearray)):
                body = body.decode('utf-8')
            outer.bodies.append(json.loads(body))
            return outer.response

        self._patch = patch('requests.sessions.Session.send', new=send)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def zip_of(members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, payload in members:
            z.writestr(name, payload)
    return buf.getvalue()


def gfql(g, **kwargs):
    kwargs.setdefault('format', 'json')
    kwargs.setdefault('output_type', 'all')
    return chain_remote_generic(g, kwargs.pop('chain', [n()]), **CREDS, **kwargs)


def pyrem(g, **kwargs):
    kwargs.setdefault('format', 'json')
    kwargs.setdefault('output_type', 'json')
    return python_remote_generic(g, TASK, **CREDS, **kwargs)


# --- #1956 1: a JSON content-type with a non-JSON payload -------------------


@pytest.mark.parametrize('call', [gfql, pyrem])
def test_json_ctype_non_json_error_body_is_typed_not_jsondecodeerror(call):
    with Transport(resp(500, b'<html>gateway error</html>', 'application/json')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph())
    assert not isinstance(ei.value, requests.exceptions.RequestException)
    assert 'gateway error' in str(ei.value)
    assert '500' in str(ei.value)
    assert ei.value.context['status_code'] == 500


# --- #1956 2: python_remote must not leak a raw HTTPError -------------------


@pytest.mark.parametrize('call', [gfql, pyrem])
def test_non_json_error_status_keeps_body_and_stays_typed(call):
    with Transport(resp(502, b'<html>bad gateway</html>', 'text/html')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph())
    assert not isinstance(ei.value, requests.exceptions.RequestException)
    assert 'bad gateway' in str(ei.value)
    assert '502' in str(ei.value)


# --- #1956 3: the server's own message survives the zip handler -------------


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'parquet', 'output_type': 'all'}),
    (pyrem, {'format': 'parquet', 'output_type': 'all'}),
])
def test_zip_path_error_body_keeps_server_message(call, kwargs):
    body = json.dumps({'error': "GFQL validation failed: unknown column 'foo'"}).encode()
    with Transport(resp(200, body, 'application/json')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph(), **kwargs)
    assert "unknown column 'foo'" in str(ei.value)


# --- #1956 4: a 200 whose JSON body is an error document --------------------


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'json', 'output_type': 'all'}),
    (pyrem, {'format': 'json', 'output_type': 'all'}),
])
def test_json_200_error_document_is_typed_not_keyerror(call, kwargs):
    with Transport(resp(200, json.dumps({'error': 'boom'}).encode(), 'application/json')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph(), **kwargs)
    assert 'boom' in str(ei.value)


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'json', 'output_type': 'all'}),
    (pyrem, {'format': 'json', 'output_type': 'all'}),
])
def test_json_200_missing_result_key_is_typed_not_keyerror(call, kwargs):
    with Transport(resp(200, json.dumps({'nodes': []}).encode(), 'application/json')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph(), **kwargs)
    assert 'edges' in str(ei.value)


# --- #1956 5: a zip missing an expected member ------------------------------


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'parquet', 'output_type': 'all'}),
    (pyrem, {'format': 'parquet', 'output_type': 'all'}),
])
def test_zip_missing_member_is_typed_not_indexerror(call, kwargs):
    payload = zip_of([('edges.parquet', parquet_bytes(EDGES))])
    with Transport(resp(200, payload, 'application/zip')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph(), **kwargs)
    assert 'nodes' in str(ei.value)


# --- #1956 6: substring member selection silently bound the WRONG table -----


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'parquet', 'output_type': 'all'}),
    (pyrem, {'format': 'parquet', 'output_type': 'all'}),
])
def test_zip_member_selection_binds_nodes_to_the_nodes_table(call, kwargs):
    payload = zip_of([
        ('nodes_and_edges.parquet', parquet_bytes(EDGES)),
        ('nodes.parquet', parquet_bytes(NODES)),
        ('edges.parquet', parquet_bytes(EDGES)),
    ])
    with Transport(resp(200, payload, 'application/zip')):
        out = call(bound_graph(), **kwargs)
    assert out._nodes.to_dict('records') == NODES.to_dict('records')
    assert out._edges.to_dict('records') == EDGES.to_dict('records')
    assert list(out._nodes.columns) == ['id', 'x']


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'parquet', 'output_type': 'all'}),
    (pyrem, {'format': 'parquet', 'output_type': 'all'}),
])
def test_zip_ambiguous_member_is_typed_not_arbitrary(call, kwargs):
    payload = zip_of([
        ('a/nodes_1.parquet', parquet_bytes(NODES)),
        ('a/nodes_2.parquet', parquet_bytes(NODES)),
        ('edges.parquet', parquet_bytes(EDGES)),
    ])
    with Transport(resp(200, payload, 'application/zip')):
        with pytest.raises(GFQLRemoteError) as ei:
            call(bound_graph(), **kwargs)
    assert 'nodes' in str(ei.value)


@pytest.mark.parametrize('call,kwargs', [
    (gfql, {'format': 'parquet', 'output_type': 'all'}),
    (pyrem, {'format': 'parquet', 'output_type': 'all'}),
])
def test_well_formed_zip_still_round_trips(call, kwargs):
    payload = zip_of([
        ('nodes.parquet', parquet_bytes(NODES)),
        ('edges.parquet', parquet_bytes(EDGES)),
    ])
    with Transport(resp(200, payload, 'application/zip')):
        out = call(bound_graph(), **kwargs)
    assert out._nodes.to_dict('records') == NODES.to_dict('records')
    assert out._edges.to_dict('records') == EDGES.to_dict('records')


# --- #1960 1: NaN/inf get the same typed decline as other non-JSON values ---


@pytest.mark.parametrize('flt', [[n({'x': float('nan')})],
                                 [n({'x': float('inf')})],
                                 [n({'x': float('-inf')})],
                                 [n({'x': gt(float('nan'))})]])
def test_non_finite_filter_value_is_typed_and_never_reaches_the_wire(flt):
    t = Transport(resp(200, json.dumps({'nodes': [], 'edges': []}).encode(), 'application/json'))
    with t:
        with pytest.raises(GFQLTypeError) as ei:
            gfql(bound_graph(), chain=flt)
    assert not isinstance(ei.value, requests.exceptions.RequestException)
    assert t.bodies == []


def test_finite_filter_value_still_goes_on_the_wire():
    t = Transport(resp(200, json.dumps({'nodes': [], 'edges': []}).encode(), 'application/json'))
    with t:
        gfql(bound_graph(), chain=[n({'x': 1.5})])
    assert t.bodies[0]['gfql_operations'][0]['filter_dict'] == {'x': 1.5}


# --- #1960 2: output= is honored on a Let, declined typed on a flat chain ---


def test_output_on_flat_chain_is_declined_typed_not_dropped():
    t = Transport(resp(200, json.dumps({'nodes': [], 'edges': []}).encode(), 'application/json'))
    with t:
        with pytest.raises(GFQLSyntaxError) as ei:
            gfql(bound_graph(), output='foo')
    assert 'output' in str(ei.value)
    assert t.bodies == []


def test_output_on_let_still_reaches_the_wire():
    let = {'type': 'Let', 'bindings': {'foo': {'type': 'Chain', 'chain': [{'type': 'Node'}]}}}
    t = Transport(resp(200, json.dumps({'nodes': [], 'edges': []}).encode(), 'application/json'))
    with t:
        gfql(bound_graph(), chain=let, output='foo')
    assert t.bodies[0]['gfql_output'] == 'foo'


# --- #1960 3: the shape variant accepts params/output -----------------------


def test_shape_variant_accepts_cypher_params():
    t = Transport(resp(200, json.dumps({'nodes': [1], 'edges': [1]}).encode(), 'application/json'))
    with t:
        out = bound_graph().gfql_remote_shape(
            "MATCH (a) WHERE a.x > $cut RETURN a", params={'cut': 1}, **CREDS)
    assert isinstance(out, pd.DataFrame)
    assert t.bodies[0]['gfql_operations']


# --- #1960 4: a column subset must not strand the result's own bindings -----


@pytest.mark.parametrize('fmt,payload_key', [('json', 'json'), ('parquet', 'zip')])
def test_node_col_subset_dropping_the_bound_id_is_typed(fmt, payload_key):
    if payload_key == 'json':
        r = resp(200, json.dumps({'nodes': [{'x': 'a'}], 'edges': [{'s': 0, 'd': 1}]}).encode(),
                 'application/json')
    else:
        r = resp(200, zip_of([('nodes.parquet', parquet_bytes(NODES[['x']])),
                              ('edges.parquet', parquet_bytes(EDGES))]), 'application/zip')
    with Transport(r):
        with pytest.raises(GFQLSchemaError) as ei:
            gfql(bound_graph(), format=fmt, node_col_subset=['x'])
    assert "'id'" in str(ei.value)


def test_edge_col_subset_dropping_the_bound_destination_is_typed():
    r = resp(200, json.dumps({'nodes': [{'id': 0}], 'edges': [{'s': 0}]}).encode(), 'application/json')
    with Transport(r):
        with pytest.raises(GFQLSchemaError) as ei:
            gfql(bound_graph(), edge_col_subset=['s'])
    assert "'d'" in str(ei.value)


def test_col_subset_keeping_the_bound_columns_is_accepted():
    r = resp(200, json.dumps({'nodes': [{'id': 0, 'x': 'a'}], 'edges': [{'s': 0, 'd': 1}]}).encode(),
             'application/json')
    with Transport(r):
        out = gfql(bound_graph(), node_col_subset=['id', 'x'], edge_col_subset=['s', 'd'])
    assert out._node == 'id'
    assert list(out._nodes.columns) == ['id', 'x']


def test_no_col_subset_leaves_server_supplied_bindings_alone():
    r = resp(200, json.dumps({
        'nodes': [{'id': 0}],
        'edges': [{'src': 0, 'dst': 1}],
        'metadata': {'bindings': {'source': 'new_src', 'destination': 'new_dst'}},
    }).encode(), 'application/json')
    with Transport(r):
        out = gfql(bound_graph())
    assert out._source == 'new_src'
