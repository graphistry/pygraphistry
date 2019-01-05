import random
import string

import pyarrow as arrow
import pytest

from graphistry.util.graph.rectify import rectify, _rectify_edge_ids, _rectify_node_ids

int32 = arrow.int32()

id_column_name = 'ids'
letters_column_name = 'letters'


def test_rectify_edge_ids_int64():
    edge_table = _create_simple_table()
    edge_table = _rectify_edge_ids(
        edge_table,
        id_column_name
    )
    assert edge_table.column(id_column_name).type == int32


def test_rectify_edge_ids_string():
    edge_table = _create_simple_table()
    edge_table = _rectify_edge_ids(
        edge_table,
        letters_column_name
    )
    assert edge_table.column(letters_column_name).type == int32


def test_rectify_edge_ids_missing():
    missing_column_name = '(missing)'
    edge_table = _create_simple_table()
    edge_table = _rectify_edge_ids(
        edge_table,
        missing_column_name
    )
    assert edge_table.column(missing_column_name).type == int32


def test_rectify_node_ids():
    pass


def _create_simple_table():
    letters = arrow.column(
        letters_column_name, [string.ascii_uppercase])
    indicies = arrow.column(
        id_column_name, [range(letters.length())])
    return arrow.Table.from_arrays([
        indicies,
        letters
    ])
