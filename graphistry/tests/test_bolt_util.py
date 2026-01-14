# -*- coding: utf-8 -*-

import datetime as dt, graphistry, os, numpy as np, pandas as pd, pyarrow as pa, pytest

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
    assert df2.dtypes.to_dict() == {
        "x": "int",
        "b": "bool",
        "s": "object",
        "a": "object",
        "d": "object",
        "mt": "object",
    }
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
    assert df2.dtypes.to_dict() == {
        "x": "float64",
        "b": "object",
        "s": "object",
        "a": "object",
        "d": "object",
        "mt": "object",
    }
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
    assert df2.dtypes.to_dict() == {
        "d": "datetime64[ns]",
        "dt": "datetime64[ns]",
        "t": "timedelta64[ns]",
        "dur": "object",
    }
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
    assert df2.dtypes.to_dict() == {
        "d": "datetime64[ns]",
        "dt": "datetime64[ns]",
        "t": "timedelta64[ns]",
        "dur": "object",
    }
    d = df2.to_dict(orient="records")[0]
    assert d == {
        "d": dt.datetime(2020, 10, 20),
        "dt": dt.datetime(2020, 10, 20, 3, 4, 5),
        "t": pd.to_timedelta("10:20:30"),
        "dur": "P1Y3M14D",
    }
    assert df2.to_dict(orient="records")[1] == {
        "d": pd.NaT,
        "dt": pd.NaT,
        "t": pd.NaT,
        "dur": None,
    }
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
    assert df.dtypes.to_dict() == {
        "d": "object",
        "dt": "object",
        "t": "object",
        "dur": "object",
    }
    assert df2.dtypes.to_dict() == {
        "d": "object",
        "dt": "object",
        "t": "object",
        "dur": "object",
    }
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
    assert df2.dtypes.to_dict() == {
        "p": np.dtype("object"),
        "c": np.dtype("object"),
        "c_x": np.dtype("float64"),
        "c_y": np.dtype("float64"),
        "c_srid": np.dtype("int64"),
        "c2": np.dtype("object"),
        "c2_x": np.dtype("float64"),
        "c2_y": np.dtype("float64"),
        "c2_z": np.dtype("float64"),
        "c2_srid": np.dtype("int64"),
        "w": np.dtype("object"),
        "w_x": np.dtype("float64"),
        "w_y": np.dtype("float64"),
        "w_z": np.dtype("float64"),
        #"w_longitude": np.dtype("float64"),
        "w_latitude": np.dtype("float64"),
        "w_height": np.dtype("float64"),
        "w_srid": np.dtype("int64"),
    }
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
    assert df2.dtypes.to_dict() == {
        "p": np.dtype("object"),
        #"p_srid": np.dtype("object"),
        "c": np.dtype("object"),
        "c_x": np.dtype("float64"),
        "c_y": np.dtype("float64"),
        "c_srid": np.dtype("float64"),
        "c2": np.dtype("object"),
        "c2_x": np.dtype("float64"),
        "c2_y": np.dtype("float64"),
        "c2_z": np.dtype("float64"),
        "c2_srid": np.dtype("float64"),
        "w": np.dtype("object"),
        "w_x": np.dtype("float64"),
        "w_y": np.dtype("float64"),
        #"w_longitude": np.dtype("float64"),
        "w_latitude": np.dtype("float64"),
        "w_srid": np.dtype("float64")
    }
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
    assert df2.dtypes.to_dict() == {
        "p": "object",
        "c": "object",
        "c2": "object",
        "w": "object",
    }
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
