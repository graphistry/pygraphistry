# -*- coding: utf-8 -*-

import datetime as dt, graphistry, os, numpy as np, pandas as pd, pyarrow as pa, pytest
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

try:
    import neo4j
    has_neo4j = True
except (ImportError, ModuleNotFoundError):
    has_neo4j = False

from graphistry.bolt_util import (
    neo_df_to_pd_df,
    node_id_key,
    node_type_key,
    start_node_id_key,
    end_node_id_key,
    relationship_id_key,
    relationship_type_key,
)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_neo_df_to_pd_df_basics():
    rec = {"x": 1, "b": True, "s": "abc", "a": [1, 2, 3], "d": {"r": "v"}, "mt": None}
    df = pd.DataFrame([rec])
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    assert is_integer_dtype(dtypes["x"])
    assert is_bool_dtype(dtypes["b"])
    assert is_string_dtype(dtypes["s"])
    for col in ("a", "d", "mt"):
        assert is_object_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == rec
    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_neo_df_to_pd_df_basics_na():
    recs = {
        "x": [1, None],
        "b": [True, None],
        "s": ["abc", None],
        "a": [[1, 2, 3], None],
        "d": [{"r": "v"}, None],
        "mt": [None, None],
    }
    df = pd.DataFrame(recs)
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    assert is_float_dtype(dtypes["x"])
    assert is_object_dtype(dtypes["b"])
    assert is_string_dtype(dtypes["s"])
    for col in ("a", "d", "mt"):
        assert is_object_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == {k: recs[k][0] for k in recs.keys()}
    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_dates_homogeneous():
    rec = {
        "d": neo4j.time.Date(2020, 10, 20),
        "dt": neo4j.time.DateTime(2020, 10, 20, 3, 4, 5),
        "t": neo4j.time.Time(10, 20, 30),
        "dur": neo4j.time.Duration(1, 3, 2),
    }
    df = pd.DataFrame([rec])
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    assert is_datetime64_any_dtype(dtypes["d"])
    assert is_datetime64_any_dtype(dtypes["dt"])
    assert is_timedelta64_dtype(dtypes["t"])
    assert is_string_dtype(dtypes["dur"])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "d": dt.datetime(2020, 10, 20),
        "dt": dt.datetime(2020, 10, 20, 3, 4, 5),
        "t": pd.to_timedelta("10:20:30"),
        "dur": "P1Y3M14D",
    }
    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_dates_homogeneous_na():
    recs = {
        "d": [neo4j.time.Date(2020, 10, 20), None],
        "dt": [neo4j.time.DateTime(2020, 10, 20, 3, 4, 5), None],
        "t": [neo4j.time.Time(10, 20, 30), None],
        "dur": [neo4j.time.Duration(1, 3, 2), None],
    }
    df = pd.DataFrame(recs)
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    assert is_datetime64_any_dtype(dtypes["d"])
    assert is_datetime64_any_dtype(dtypes["dt"])
    assert is_timedelta64_dtype(dtypes["t"])
    assert is_string_dtype(dtypes["dur"])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "d": dt.datetime(2020, 10, 20),
        "dt": dt.datetime(2020, 10, 20, 3, 4, 5),
        "t": pd.to_timedelta("10:20:30"),
        "dur": "P1Y3M14D",
    }
    d2 = df2.to_dict(orient="records")[1]
    for col in ("d", "dt", "t", "dur"):
        assert pd.isna(d2[col])
    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_dates_heterogeneous():
    recs = {
        "d": [neo4j.time.Date(2020, 10, 20), 1],
        "dt": [neo4j.time.DateTime(2020, 10, 20, 3, 4, 5), 1],
        "t": [neo4j.time.Time(10, 20, 30), 1],
        "dur": [neo4j.time.Duration(1, 3, 2), 1],
    }
    df = pd.DataFrame(recs)
    df2 = neo_df_to_pd_df(df)
    dtypes = df.dtypes.to_dict()
    for col in ("d", "dt", "t", "dur"):
        assert is_object_dtype(dtypes[col])
    dtypes = df2.dtypes.to_dict()
    for col in ("d", "dt", "t", "dur"):
        assert is_object_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "d": dt.datetime(2020, 10, 20),
        "dt": dt.datetime(2020, 10, 20, 3, 4, 5),
        "t": pd.to_timedelta("10:20:30"),
        "dur": "P1Y3M14D",
    }
    assert df2.to_dict(orient="records")[1] == {"d": 1, "dt": 1, "t": 1, "dur": 1}
    with pytest.raises(pa.lib.ArrowTypeError):
        pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_spatial_homogenous():
    rec = {
        "p": neo4j.spatial.Point([1, 2, 3]),
        "c": neo4j.spatial.CartesianPoint([1, 2]),
        "c2": neo4j.spatial.CartesianPoint([1, 2, 3]),
        "w": neo4j.spatial.WGS84Point([4, 5, 6]),
    }
    df = pd.DataFrame([rec])
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    for col in ("p", "c", "c2", "w"):
        assert is_string_dtype(dtypes[col])
    for col in (
        "c_x",
        "c_y",
        "c2_x",
        "c2_y",
        "c2_z",
        "w_x",
        "w_y",
        "w_z",
        "w_latitude",
        "w_height",
    ):
        assert is_float_dtype(dtypes[col])
    for col in ("c_srid", "c2_srid", "w_srid"):
        assert is_integer_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "p": "POINT(1.0 2.0 3.0)",
        "c": "POINT(1.0 2.0)",
        "c_x": 1.,
        "c_y": 2.,
        "c_srid": 7203,
        "c2": "POINT(1.0 2.0 3.0)",
        "c2_x": 1.,
        "c2_y": 2.,
        "c2_z": 3.,
        "c2_srid": 9157,
        "w": "POINT(4.0 5.0 6.0)",
        "w_height": 6.,
        "w_x": 4.,
        "w_y": 5.,
        "w_z": 6.,
        #"w_longitude": 4,
        "w_latitude": 5.,
        "w_srid": 4979,
    }
    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_spatial_homogenous_na():
    recs = {
        "p": [neo4j.spatial.Point([1, 2, 3, 4]), None],
        "c": [neo4j.spatial.CartesianPoint([1, 2]), None],
        "c2": [neo4j.spatial.CartesianPoint([1, 2, 3]), None],
        "w": [neo4j.spatial.WGS84Point([4, 5]), None],
    }
    df = pd.DataFrame(recs)
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    for col in ("p", "c", "c2", "w"):
        assert is_string_dtype(dtypes[col])
    for col in (
        "c_x",
        "c_y",
        "c_srid",
        "c2_x",
        "c2_y",
        "c2_z",
        "c2_srid",
        "w_x",
        "w_y",
        "w_latitude",
        "w_srid",
    ):
        assert is_float_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "p": "POINT(1.0 2.0 3.0 4.0)",
        "c": "POINT(1.0 2.0)",
        "c_x": 1.,
        "c_y": 2.,
        "c_srid": 7203.,
        "c2": "POINT(1.0 2.0 3.0)",
        "c2_x": 1.,
        "c2_y": 2.,
        "c2_z": 3.,
        "c2_srid": 9157.,
        "w": "POINT(4.0 5.0)",
        "w_x": 4.,
        "w_y": 5.,
        #"w_longitude": 4.,
        "w_latitude": 5.,
        "w_srid": 4326.,
    }

    d2 = df2.to_dict(orient="records")[1]
    for k in d2.keys():
        if k not in ["p", "c", "c2", "w", "p_srid"]:
            assert pd.isna(d2[k])

    pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_spatial_heterogeneous():
    recs = {
        "p": [neo4j.spatial.Point([1, 2, 3, 4]), 1],
        "c": [neo4j.spatial.CartesianPoint([1, 2]), 1],
        "c2": [neo4j.spatial.CartesianPoint([1, 2, 3]), 1],
        "w": [neo4j.spatial.WGS84Point([4, 5]), 1],
    }
    df = pd.DataFrame(recs)
    df2 = neo_df_to_pd_df(df)
    dtypes = df2.dtypes.to_dict()
    for col in ("p", "c", "c2", "w"):
        assert is_object_dtype(dtypes[col])
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "p": "POINT(1.0 2.0 3.0 4.0)",
        "c": "POINT(1.0 2.0)",
        "c2": "POINT(1.0 2.0 3.0)",
        "w": "POINT(4.0 5.0)",
    }
    assert df2.to_dict(orient="records")[1] == {
        "p": 1,
        "c": 1,
        "c2": 1,
        "w": 1,
    }
    with pytest.raises(pa.lib.ArrowTypeError):
        pa.Table.from_pandas(df2)


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
@pytest.mark.skipif(
    not ("WITH_NEO4J" in os.environ) or os.environ["WITH_NEO4J"] != "1",
    reason="No WITH_NEO4J=1",
)
class Test_Neo4jConnector:
    @classmethod
    def setup_class(cls):
        from neo4j import GraphDatabase

        NEO4J_CREDS = {"uri": "bolt://neo4j4-test:7687", "auth": ("neo4j", "test")}
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=3, bolt=GraphDatabase.driver(**NEO4J_CREDS))

    def test_neo4j_conn_setup(self):
        assert True is True

    def test_neo4j_ready(self):

        g = graphistry.cypher("MATCH (a)-[b]-(c) WHERE a <> c RETURN a, b, c LIMIT 1")
        assert len(g._nodes) == 2

    def test_neo4j_no_edges(self):

        with pytest.warns(RuntimeWarning):
            g = graphistry.cypher("MATCH (a) RETURN a LIMIT 1")

        assert len(g._nodes) == 1
        assert len(g._edges) == 0
        for col in [node_id_key, node_type_key]:
            assert col in g._nodes
        for col in [
            start_node_id_key,
            end_node_id_key,
            relationship_id_key,
            relationship_type_key,
        ]:
            assert col in g._edges

    def test_neo4j_no_nodes(self):
        with pytest.warns(RuntimeWarning):
            g = graphistry.cypher("MATCH (a) WHERE a.x = 123 RETURN a LIMIT 1")

        assert len(g._nodes) == 0
        assert len(g._edges) == 0
        for col in [node_id_key, node_type_key]:
            assert col in g._nodes
        for col in [
            start_node_id_key,
            end_node_id_key,
            relationship_id_key,
            relationship_type_key,
        ]:
            assert col in g._edges

    def test_neo4j_some_edges(self):

        g = graphistry.cypher("MATCH (a)-[b]-(c) WHERE a <> c RETURN a, b, c LIMIT 1")
        assert len(g._nodes) == 2
        assert len(g._edges) == 1

        for col in [node_id_key, node_type_key]:
            assert col in g._nodes

        for col in [
            start_node_id_key,
            end_node_id_key,
            relationship_id_key,
            relationship_type_key,
        ]:
            assert col in g._edges


@pytest.mark.skipif(not has_neo4j, reason="No neo4j")
def test_stringify_spatial_unit():
    """Unit tests for stringify_spatial function format compatibility"""
    from graphistry.bolt_util import stringify_spatial

    # Test None and non-spatial inputs
    assert stringify_spatial(None) is None
    assert stringify_spatial("test") == "test"

    # Test with real Neo4j spatial objects to ensure our fix works
    point = neo4j.spatial.Point([1.0, 2.0, 3.0])
    cartesian = neo4j.spatial.CartesianPoint([4.0, 5.0])
    wgs84 = neo4j.spatial.WGS84Point([6.0, 7.0, 8.0])

    # All should be converted to old POINT(...) format with space-separated coords
    results = [stringify_spatial(obj) for obj in [point, cartesian, wgs84]]
    for result in results:
        assert result.startswith("POINT(") and result.endswith(")")
        assert "," not in result  # No commas in coordinates
