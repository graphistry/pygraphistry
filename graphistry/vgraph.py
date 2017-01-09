from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import next
from builtins import str

import sys
import random
import numpy
import pandas

from . import pygraphistry
from . import util
from . import graph_vector_pb2
from .graph_vector_pb2 import VectorGraph


EDGE = graph_vector_pb2.VectorGraph.EDGE
VERTEX = graph_vector_pb2.VectorGraph.VERTEX


# Creates the ETL2 protobuf vgraph from
#  - edge_df: the edge dataframe
#  - node_df: the node dataframe
#  - sources: a series of edge sources
#  - dests: a series of edge destinations
#  - nodeid: The name of the nodeId column in node_df
#  - node_map: A map from nodeId to a dense integer range [0, #nodes -1]
#  - name: The name of the dataset.
def create(edge_df, node_df, sources, dests, nodeid, node_map, name):
    vg = graph_vector_pb2.VectorGraph()
    vg.version = 1
    vg.type = VectorGraph.DIRECTED
    vg.vertexCount = len(node_map)
    vg.edgeCount = len(edge_df)
    if name is not None:
        vg.name = name

    addEdges(vg, sources, dests, node_map)
    edge_types = storeEdgeAttributes(vg, edge_df)
    node_types = storeNodeAttributes(vg, node_df, nodeid, node_map)

    return  {
        'name': name,
        'vgraph': vg,
        'attributes': {
            'nodes': node_types,
            'edges': edge_types
        },
    }


# Encode edges into protobuf using source/dest pairs in [0, #nodes-1] range.
def addEdges(vg, sources, dests, node_map):
    for s, d in zip(sources.tolist(), dests.tolist()):
        e = vg.edges.add()
        e.src = node_map[s]
        e.dst = node_map[d]


def storeEdgeAttributes(vg, df):
    edge_types = {}

    coltypes = df.columns.to_series().groupby(df.dtypes)
    for dtype, cols in list(coltypes.groups.items()):
        for col in cols:
            enc_type = storeValueVector(vg, df, col, dtype, EDGE)
            edge_types[col] = enc_type

    return edge_types


def storeNodeAttributes(vg, df, nodeid, node_map):
    ordercol = '__order__'
    node_types = {}

    # Sort values of node attributes based on assigned id in [0, #nodes-1] range
    df[ordercol] = df[nodeid].map(lambda n: node_map[n])
    df.sort_values(ordercol, inplace=True)
    df.drop(ordercol, axis=1, inplace=True)
    coltypes = df.columns.to_series().groupby(df.dtypes)

    for dtype, cols in list(coltypes.groups.items()):
        for col in cols:
            enc_type = storeValueVector(vg, df, col, dtype, VERTEX)
            node_types[col] = enc_type

    return node_types


# Create a protobuf vector of value (for storing node/edge attributes) given
#  - vg: A VGraph instance
#  - df: An input dataframe (nodes or edges)
#  - col: The name of the column to encode inside the input dataframe
#  - dtype: The numpy type of the column
#  - target: Whether attribute applies to nodes or edges (one of VERTEX/EDGE)
def storeValueVector(vg, df, col, dtype, target):
    encoders = {
        'object': objectEncoder,
        'bool': boolEncoder,
        'int8': numericEncoder,
        'int16': numericEncoder,
        'int32': numericEncoder,
        'int64': numericEncoder,
        'float16': numericEncoder,
        'float32': numericEncoder,
        'float64': numericEncoder,
        'datetime64[ns]': datetimeEncoder,
    }
    df_col = df[col]
    (vec, info) = encoders[dtype.name](vg, df_col, dtype)
    vec.name = str(col)
    vec.target = target

    if 'aggregations' not in info:
        info['aggregations'] = {}
    aggregations = info['aggregations']
    if 'valid' not in aggregations:
        aggregations['valid'] = nanGuard(df_col.count())
        aggregations['missing'] = nanGuard(df_col.size - aggregations['valid'])
    if 'distinct' not in aggregations:
        aggregations['distinct'] = nanGuard(df_col.nunique())
    if 'min' not in aggregations:
        aggregations['min'] = nanGuard(df_col.min())
    if 'max' not in aggregations:
        aggregations['max'] = nanGuard(df_col.max())

    return info


# returns tuple() of StringAttributeVector and object with type info.
def objectEncoder(vg, series, dtype):
    series.where(pandas.notnull(series), '\0', inplace=True)
    # vec is a string[] submessage within a repeated
    vec = vg.string_vectors.add()
    for val in series.astype('unicode'):
        vec.values.append(val)
    return (vec, {'ctype': 'utf8'})


# NaN (as well as Infinity and undefined) are valid JSON. Use this guard to filter
# them out when creating the json metadata.
def nanGuard(value):
    try:
        if numpy.isnan(value):
            return None
    except:
        pass
    return value


def numericEncoder(vg, series, dtype):
    def getBestRep(series, candidate_types):
        min = series.min()
        max = series.max()
        tinfo = [numpy.iinfo(t) for t in candidate_types]
        return next(i.dtype for i in tinfo if min >= i.min and max <= i.max)

    typemap = {
        'int8': vg.int32_vectors,
        'int16': vg.int32_vectors,
        'int32': vg.int32_vectors,
        'int64': vg.int64_vectors,
        'float16': vg.float_vectors,
        'float32': vg.float_vectors,
        'float64': vg.double_vectors
    }

    if dtype.name.startswith('int'):
        candidate_types = [numpy.int8, numpy.int16, numpy.int32, numpy.int64]
        rep_type = getBestRep(series, candidate_types)
    else:
        rep_type = dtype

    vec = typemap[rep_type.name].add()
    if sys.version_info < (3,):
        for val in series:
            vec.values.append(val)
    else:
        for val in series:
            vec.values.append(val.item()) # Cast to Python native int? Loss of precision?

    variance = series.var()
    stddev = series.std()
    info = {
        'ctype': rep_type.name,
        'originalType': dtype.name,
        'aggregations': {
            'mean': nanGuard(series.mean()),
            'variance': nanGuard(variance),
            'stddev': nanGuard(stddev)
        }
    }
    return (vec, info)


def boolEncoder(vg, series, dtype):
    vec = vg.bool_vectors.add()
    for val in series:
        vec.values.append(val.item())
    return (vec, {
        'ctype': 'bool'
    })


def datetimeEncoder(vg, series, dtype):
    vec = vg.int32_vectors.add()
    util.warn('Casting dates to UNIX epoch (resolution of 1 second)')
    series32 = series.astype('int64').map(lambda x: x / 1e9).astype(numpy.int32)
    for val in series32:
        vec.values.append(val.item())

    info = {
        'ctype': 'datetime32[s]',
        'userType': 'datetime',
        'aggregations': {
            'min': series32.min(),
            'max': series32.max(),
            'distinct': nanGuard(series32.nunique())
        }
    }
    return (vec, info)
