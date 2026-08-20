"""Strictness levels for absent labels/properties (#1916).

Pins the three levels, the bool mapping, the validator/executor agreement matrix,
schema-declared typo-vs-narrow-instance disambiguation, and the remote wire field.
"""

from typing import Any, Dict, List, Optional
from unittest import mock
import warnings

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.exceptions import GFQLSchemaError, GFQLValidationError
from graphistry.compute.gfql.strictness import (
    DEFAULT_STRICT_LEVEL,
    UNSCOPED_STRICT_LEVEL,
    absent_column_matches,
    normalize_strict_level,
    resolve_strict_level,
    schema_declared_names,
    strict_level_to_bool,
)
from graphistry.compute.gfql_validate import gfql_validate
from graphistry.schema import EdgeType, GraphSchema, NodeType


ABSENT_LABEL = "MATCH (n:Nope) RETURN n.id AS id"
ABSENT_PROP_RETURN = "MATCH (n) RETURN n.nope_col AS c"
ABSENT_PROP_WHERE = "MATCH (n) WHERE n.nope_col = 1 RETURN n.id AS id"
ABSENT_PROP_PATTERN = "MATCH (n {nope_col: 1}) RETURN n.id AS id"
ABSENT_EDGE_LABEL = "MATCH (n)-[e:NOPE]->(m) RETURN n.id AS id"

FOUR_SHAPES = [ABSENT_LABEL, ABSENT_PROP_RETURN, ABSENT_PROP_WHERE, ABSENT_PROP_PATTERN]


ENGINES = ("pandas", "polars", "cudf")


def _graph(engine: str = "pandas") -> Plottable:
    nodes = pd.DataFrame({"id": [1, 2, 3], "t": ["a", "b", "a"]})
    edges = pd.DataFrame({"s": [1, 2], "d": [2, 3]})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "s", "d").nodes(nodes, "id")


def _norm(value: Any) -> Any:  # hygiene-ok: explicit-any -- row cells are heterogeneous
    # py3.13 pandas renders a null cell as float nan where 3.12 gave None
    if isinstance(value, float) and value != value:
        return None
    return None if value is pd.NA else value


def _rows(g: Plottable) -> List[Dict[str, Any]]:  # hygiene-ok: explicit-any -- row cells are heterogeneous
    if g._nodes is None:
        return []
    frame = g._nodes
    if hasattr(frame, "to_pandas") and not isinstance(frame, pd.DataFrame):
        frame = frame.to_pandas()
    return [{k: _norm(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _gfql_warnings(g: Plottable, query: str, **kwargs: Any) -> List[str]:  # hygiene-ok: explicit-any -- passthrough kwargs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.gfql(query, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning) and "GFQL" in str(w.message)]


# ---------------------------------------------------------------------------
# level resolution + bool mapping
# ---------------------------------------------------------------------------

def test_default_level_is_warn() -> None:
    assert DEFAULT_STRICT_LEVEL == "warn"
    assert resolve_strict_level(_graph()) == "warn"


def test_bool_true_maps_to_strict_and_false_to_quiet() -> None:
    assert normalize_strict_level(True) == "strict"
    assert normalize_strict_level(False) == "quiet"
    assert normalize_strict_level(None) is None
    assert strict_level_to_bool("strict") is True
    assert strict_level_to_bool("warn") is False
    assert strict_level_to_bool("quiet") is False


def test_unknown_level_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_strict_level("loose")


def test_precedence_explicit_beats_schema() -> None:
    g = _graph().bind(schema=GraphSchema(node_types=[NodeType("P", properties={"id": int})], strict=True))
    assert resolve_strict_level(g) == "strict"
    assert resolve_strict_level(g, strict="quiet") == "quiet"


def test_precedence_schema_metadata_tier() -> None:
    from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog

    catalog = GraphSchemaCatalog.from_schema_parts(
        node_columns=("id",), edge_columns=("s", "d"), metadata={"strict": "quiet"}
    )
    g = _graph().bind(schema=catalog)
    assert resolve_strict_level(g) == "quiet"


def test_unscoped_runtime_stays_strict() -> None:
    # a direct filter_nodes_by_dict is not a GFQL call and keeps raising
    assert UNSCOPED_STRICT_LEVEL == "strict"
    with pytest.raises(GFQLSchemaError):
        _graph().filter_nodes_by_dict({"nope_col": 1})


# ---------------------------------------------------------------------------
# execution semantics: absent name resolves to null (openCypher)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_absent_label_is_zero_rows(engine: str, level: str) -> None:
    assert _rows(_graph(engine).gfql(ABSENT_LABEL, strict=level)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_absent_edge_label_is_zero_rows(engine: str, level: str) -> None:
    assert _rows(_graph(engine).gfql(ABSENT_EDGE_LABEL, strict=level)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_absent_property_in_where_is_zero_rows(engine: str, level: str) -> None:
    assert _rows(_graph(engine).gfql(ABSENT_PROP_WHERE, strict=level)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_absent_property_in_pattern_is_zero_rows(engine: str, level: str) -> None:
    assert _rows(_graph(engine).gfql(ABSENT_PROP_PATTERN, strict=level)) == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_absent_property_in_return_is_null_column(engine: str, level: str) -> None:
    assert _rows(_graph(engine).gfql(ABSENT_PROP_RETURN, strict=level)) == [{"c": None}] * 3


@pytest.mark.parametrize("engine", ENGINES)
def test_absent_property_is_null_so_is_null_matches_every_row(engine: str) -> None:
    # 3VL: every comparison against null is null, but IS NULL on an absent property is TRUE
    rows = _rows(_graph(engine).gfql("MATCH (n) WHERE n.nope_col IS NULL RETURN n.id AS id", strict="quiet"))
    assert [r["id"] for r in rows] == [1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_absent_property_is_not_null_matches_no_row(engine: str) -> None:
    rows = _rows(_graph(engine).gfql("MATCH (n) WHERE n.nope_col IS NOT NULL RETURN n.id AS id", strict="quiet"))
    assert rows == []


def test_present_column_absent_value_is_unchanged() -> None:
    # scope discipline: only ABSENT names change; a present column keeps its semantics
    assert _rows(_graph().gfql("MATCH (n) WHERE n.t = 'zzz' RETURN n.id AS id", strict="warn")) == []
    assert len(_rows(_graph().gfql("MATCH (n) WHERE n.t = 'a' RETURN n.id AS id", strict="warn"))) == 2


def test_type_mismatch_still_raises_at_every_level() -> None:
    # E302 is not an absent name; leniency must not swallow it
    for level in ("strict", "warn", "quiet"):
        with pytest.raises(GFQLSchemaError):
            _graph().gfql("MATCH (n {id: 'not-a-number'}) RETURN n.id AS id", strict=level)


# ---------------------------------------------------------------------------
# strict is behavior-preserving; warn warns once; quiet is silent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [ABSENT_LABEL, ABSENT_PROP_WHERE, ABSENT_PROP_PATTERN, ABSENT_EDGE_LABEL])
@pytest.mark.parametrize("level", ["strict", True])
def test_strict_still_raises_the_same_error(query: str, level: Any) -> None:  # hygiene-ok: explicit-any -- level is bool | str by design
    with pytest.raises(GFQLSchemaError) as exc:
        _graph().gfql(query, strict=level)
    assert exc.value.code == "column-not-found"


def test_strict_raises_on_absent_return_property() -> None:
    # the validator already rejected this under strict; the executor now agrees
    with pytest.raises(GFQLSchemaError):
        _graph().gfql(ABSENT_PROP_RETURN, strict="strict")


@pytest.mark.parametrize("query", FOUR_SHAPES)
def test_warn_emits_exactly_one_warning(query: str) -> None:
    assert len(_gfql_warnings(_graph(), query, strict="warn")) == 1


@pytest.mark.parametrize("query", FOUR_SHAPES)
def test_quiet_emits_no_warning(query: str) -> None:
    assert _gfql_warnings(_graph(), query, strict="quiet") == []


@pytest.mark.parametrize("query", FOUR_SHAPES)
def test_bool_false_is_quiet_not_warn(query: str) -> None:
    assert _gfql_warnings(_graph(), query, strict=False) == []


def test_warn_once_per_distinct_name_not_per_row() -> None:
    nodes = pd.DataFrame({"id": list(range(50)), "t": ["a"] * 50})
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.edges(edges, "s", "d").nodes(nodes, "id")
    assert len(_gfql_warnings(g, "MATCH (n) RETURN n.nope_col AS c", strict="warn")) == 1


def test_warn_once_per_name_two_distinct_names_warn_twice() -> None:
    msgs = _gfql_warnings(
        _graph(), "MATCH (n) RETURN n.nope_a AS a, n.nope_b AS b", strict="warn"
    )
    assert len(msgs) == 2


def test_warning_names_the_absent_name() -> None:
    (msg,) = _gfql_warnings(_graph(), ABSENT_PROP_WHERE, strict="warn")
    assert "nope_col" in msg


def test_absent_label_warning_says_label() -> None:
    (msg,) = _gfql_warnings(_graph(), ABSENT_LABEL, strict="warn")
    assert "label" in msg and "Nope" in msg


# ---------------------------------------------------------------------------
# validator and executor agree at every level (#1889 pattern)
# ---------------------------------------------------------------------------

def _validator_verdict(g: Plottable, query: str, level: Any) -> str:  # hygiene-ok: explicit-any -- level is bool | str by design
    try:
        gfql_validate(g, query, strict=level)
        return "ok"
    except GFQLValidationError:
        return "raise"


def _executor_verdict(g: Plottable, query: str, level: Any) -> str:  # hygiene-ok: explicit-any -- level is bool | str by design
    try:
        g.gfql(query, strict=level)
        return "ok"
    except GFQLValidationError:
        return "raise"


@pytest.mark.parametrize("query", FOUR_SHAPES + [ABSENT_EDGE_LABEL])
@pytest.mark.parametrize("level", ["strict", "warn", "quiet", True, False, None])
def test_validator_and_executor_agree(query: str, level: Any) -> None:  # hygiene-ok: explicit-any -- level is bool | str by design
    g = _graph()
    assert _validator_verdict(g, query, level) == _executor_verdict(g, query, level)


@pytest.mark.parametrize("query", FOUR_SHAPES + [ABSENT_EDGE_LABEL])
def test_agreement_matrix_values(query: str) -> None:
    g = _graph()
    assert _validator_verdict(g, query, "strict") == "raise"
    assert _validator_verdict(g, query, "warn") == "ok"
    assert _validator_verdict(g, query, "quiet") == "ok"


def test_validate_true_on_gfql_uses_the_resolved_level() -> None:
    # gfql(validate=True) used to hardcode strict=True regardless of the caller's choice
    assert _rows(_graph().gfql(ABSENT_PROP_WHERE, validate=True, strict="quiet")) == []
    with pytest.raises(GFQLValidationError):
        _graph().gfql(ABSENT_PROP_WHERE, validate=True, strict="strict")


# ---------------------------------------------------------------------------
# schema-declared typo vs narrow instance
# ---------------------------------------------------------------------------

def _schema_graph(level: Any = "warn") -> Plottable:  # hygiene-ok: explicit-any -- level is bool | str by design
    schema = GraphSchema(
        node_types=[NodeType("Person", properties={"id": int, "t": str, "city": str})],
        edge_types=[EdgeType("KNOWS", source="Person", destination="Person",
                             properties={"s": int, "d": int})],
        strict=level,
    )
    return _graph().bind(schema=schema)


def test_schema_declared_names_collects_properties_and_labels() -> None:
    names = schema_declared_names(_schema_graph())
    assert names is not None
    assert {"city", "id", "t", "Person", "label__Person"} <= names


def test_schema_declared_names_none_without_a_schema() -> None:
    assert schema_declared_names(_graph()) is None


@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_name_in_schema_absent_from_instance_is_served(level: str) -> None:
    # the narrow-subgraph case the owner ruled for
    assert _rows(_schema_graph(level).gfql("MATCH (n) WHERE n.city = 'x' RETURN n.id AS id", strict=level)) == []


@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_name_absent_from_schema_is_a_typo_and_still_raises(level: str) -> None:
    with pytest.raises(GFQLValidationError):
        _schema_graph(level).gfql("MATCH (n) WHERE n.ciyt = 'x' RETURN n.id AS id", strict=level)


@pytest.mark.parametrize("level", ["warn", "quiet"])
def test_label_absent_from_schema_is_a_typo_and_still_raises(level: str) -> None:
    with pytest.raises(GFQLValidationError):
        _schema_graph(level).gfql("MATCH (n:Persno) RETURN n.id AS id", strict=level)


def test_declared_label_absent_from_instance_is_served() -> None:
    assert _rows(_schema_graph("quiet").gfql("MATCH (n:Person) RETURN n.id AS id", strict="quiet")) == []


# ---------------------------------------------------------------------------
# chain() surface
# ---------------------------------------------------------------------------

def test_chain_honors_the_level() -> None:
    from graphistry.compute.ast import n

    g = _graph()
    assert len(g.chain([n(filter_dict={"nope_col": 1})], strict="quiet")._nodes) == 0
    with pytest.raises(GFQLSchemaError):
        g.chain([n(filter_dict={"nope_col": 1})], strict="strict")


def test_chain_default_is_warn() -> None:
    from graphistry.compute.ast import n

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _graph().chain([n(filter_dict={"nope_col": 1})])
    assert len(out._nodes) == 0
    assert any("GFQL" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# the warn level needs the process's warning filters intact
# ---------------------------------------------------------------------------

def test_lazy_cudf_import_leaves_the_global_warning_filters_intact() -> None:
    pytest.importorskip("cudf")
    from graphistry.utils.lazy_import import lazy_cudf_import

    before = list(warnings.filters)
    lazy_cudf_import()
    assert list(warnings.filters) == before


# ---------------------------------------------------------------------------
# 3VL helper
# ---------------------------------------------------------------------------

def test_absent_column_matches_only_is_na() -> None:
    from graphistry.compute.predicates.comparison import isna, notna, gt

    assert absent_column_matches(isna()) is True
    assert absent_column_matches(notna()) is False
    assert absent_column_matches(gt(1)) is False
    assert absent_column_matches(1) is False


# ---------------------------------------------------------------------------
# remote: honor the level in preflight + ship it on the wire
# ---------------------------------------------------------------------------

class _FakeResponse:
    ok = True
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = '{"nodes": [], "edges": []}'

    def json(self) -> Dict[str, Any]:  # hygiene-ok: explicit-any -- JSON payload
        return {"nodes": [], "edges": []}


def _remote_body(level: Any, query: str = "MATCH (n) RETURN n", validate: bool = False,  # hygiene-ok: explicit-any -- level is bool | str by design
                 g: Optional[Plottable] = None) -> Dict[str, Any]:  # hygiene-ok: explicit-any -- JSON payload
    captured: Dict[str, Any] = {}

    def _post(url: str, headers: Any = None, json: Any = None, verify: Any = None, **kwargs: Any) -> _FakeResponse:  # hygiene-ok: explicit-any -- requests passthrough
        captured["body"] = json
        return _FakeResponse()

    graph = g if g is not None else _graph()
    with mock.patch("graphistry.compute.chain_remote.requests.post", _post):
        try:
            chain_remote_generic(graph, query, api_token="t", dataset_id="ds",
                                 validate=validate, strict=level)
        except AttributeError:
            pass  # the stub response carries no body to deserialize; the request is what is pinned
    return captured["body"]


@pytest.mark.parametrize("level,expected", [
    (None, "warn"), ("strict", "strict"), ("warn", "warn"), ("quiet", "quiet"),
    (True, "strict"), (False, "quiet"),
])
def test_remote_sends_strictness_field(level: Any, expected: str) -> None:  # hygiene-ok: explicit-any -- level is bool | str by design
    assert _remote_body(level)["strictness"] == expected


def test_remote_default_sends_no_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _remote_body(None)
    assert [w for w in caught if "strictness" in str(w.message)] == []


@pytest.mark.parametrize("level", ["strict", "quiet", True, False])
def test_remote_non_default_level_warns_once(level: Any) -> None:  # hygiene-ok: explicit-any -- level is bool | str by design
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _remote_body(level)
    assert len([w for w in caught if "strictness" in str(w.message)]) == 1


def test_remote_preflight_no_longer_hardcodes_loose() -> None:
    # master preflighted with strict=False regardless of the caller's choice
    with pytest.raises(GFQLValidationError):
        _remote_body("strict", query=ABSENT_PROP_WHERE, validate=True)


def test_remote_preflight_serves_a_declared_schema_without_local_frames() -> None:
    # a dataset_id-only client holds no frames, but bind(schema=...) is names without data
    schema = GraphSchema(node_types=[NodeType("Person", properties={"id": int, "city": str})],
                         strict="strict")
    g = graphistry.bind().bind(schema=schema)
    g._dataset_id = "ds"
    with pytest.raises(GFQLValidationError):
        _remote_body("strict", query="MATCH (n) WHERE n.ciyt = 1 RETURN n.id", validate=True, g=g)


def test_remote_preflight_accepts_declared_name_absent_from_the_instance() -> None:
    schema = GraphSchema(node_types=[NodeType("Person", properties={"id": int, "city": str})],
                         strict="strict")
    g = graphistry.bind().bind(schema=schema)
    g._dataset_id = "ds"
    body = _remote_body("strict", query="MATCH (n) WHERE n.city = 1 RETURN n.id", validate=True, g=g)
    assert body["strictness"] == "strict"
