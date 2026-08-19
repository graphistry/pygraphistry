"""#1889 validate-vs-execute agreement matrix.

``gfql_validate`` sold itself as preflight ("validate without executing") while returning
``{ok: True, diagnostics: []}`` on graph shapes whose execution then died with a bare
``ValueError: Missing edges`` (pandas/cuDF, ``ComputeMixin.materialize_nodes``) or an
empty-message ``AssertionError`` (polars ``ensure_nodes_polars``). Verified live at master
``0c3f3a1fa`` for the both-frames-None-after-bind shape on both query languages.

The contract pinned here, per combo and per engine: the validator verdict and the
execution outcome AGREE. Either
  * both admit -- validator ok AND execution serves values, or
  * both diagnose -- validator raises a typed GFQL diagnostic AND execution declines typed.
No combo may validate clean and then raise a bare (untyped / empty-message) error.

Master reds (the drift): both-None x {cypher, chain} -- validator ok, execution bare.
Master greens kept red-proof here: nodes-only x {cypher, chain} answers on pandas (#1942),
and an edge pattern against unbound edges declines typed on BOTH surfaces.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

try:
    import polars  # noqa: F401
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]

CYPHER = "MATCH (a) RETURN a"
CHAIN = [n({"v": 20})]

# Bare = the crash classes #1889 filed: no GFQL code, no remedy (and often no message).
BARE_ERRORS = (ValueError, AssertionError, TypeError, AttributeError, KeyError, IndexError)


def _both_none():
    """Bindings set, frames never attached -- graphistry.bind() only names columns."""
    return graphistry.bind(source="s", destination="d", node="id")


def _nodes_only():
    return graphistry.nodes(pd.DataFrame({"id": [0, 1], "v": [10, 20]}), "id")


def _norm(value: Any) -> Any:
    """py3.13 keeps NaN out of record equality: compare at VALUE level, never via notna()."""
    return None if isinstance(value, float) and math.isnan(value) else value


def _records(df) -> List[Dict[str, Any]]:
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return [{k: _norm(v) for k, v in row.items()} for row in pdf.to_dict("records")]


def _validator_verdict(g, query) -> Dict[str, Any]:
    try:
        out = g.gfql_validate(query)
    except GFQLValidationError as e:
        return {"verdict": "diagnose", "code": e.code}
    assert out["ok"] is True and out["diagnostics"] == [], out
    return {"verdict": "admit", "code": None}


def _execution_verdict(g, query, engine) -> Dict[str, Any]:
    try:
        out = g.gfql(query, engine=engine)
    except GFQLValidationError as e:
        return {"verdict": "diagnose", "code": e.code, "typed": True}
    except NotImplementedError as e:
        # Honest engine-capability decline (named limitation + remedy), not a crash.
        assert str(e).strip() != "", "NotImplementedError with no message is a bare crash"
        return {"verdict": "diagnose", "code": None, "typed": True}
    except BARE_ERRORS as e:  # pragma: no cover - the #1889 defect; red at master only
        return {"verdict": "bare", "error": f"{type(e).__name__}: {e}", "typed": False}
    return {"verdict": "admit", "nodes": None if out._nodes is None else _records(out._nodes)}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "shape,query",
    [
        ("both-none", CYPHER),
        ("both-none", CHAIN),
        ("nodes-only", CYPHER),
        ("nodes-only", CHAIN),
    ],
    ids=["bothnone-cypher", "bothnone-chain", "nodesonly-cypher", "nodesonly-chain"],
)
def test_validate_execute_agreement_matrix(shape, query, engine):
    """The #1889 matrix: no cell validates clean and then crashes raw."""
    g = _both_none() if shape == "both-none" else _nodes_only()

    validation = _validator_verdict(g, query)
    execution = _execution_verdict(g, query, engine)

    assert execution["verdict"] != "bare", (
        f"{shape}/{engine}: execution raised a bare error ({execution.get('error')}) "
        f"while the validator said {validation['verdict']}"
    )
    if validation["verdict"] == "admit":
        assert execution["verdict"] in ("admit", "diagnose")
    else:
        assert execution["verdict"] == "diagnose", (
            f"{shape}/{engine}: validator diagnosed {validation['code']} but execution admitted"
        )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [CYPHER, CHAIN], ids=["cypher", "chain"])
def test_both_none_diagnoses_typed_on_both_surfaces(query, engine):
    """Master red: validator returned ok:True; execution raised bare ValueError/AssertionError."""
    g = _both_none()

    with pytest.raises(GFQLValidationError) as validate_exc:
        g.gfql_validate(query)
    assert validate_exc.value.code == ErrorCode.E305
    assert validate_exc.value.context.get("suggestion")

    with pytest.raises(GFQLValidationError) as exec_exc:
        g.gfql(query, engine=engine)
    assert exec_exc.value.code == ErrorCode.E305
    # Same shape, same verdict, same words on both surfaces.
    assert exec_exc.value.message == validate_exc.value.message


@pytest.mark.parametrize("engine", ENGINES)
def test_edge_pattern_without_edges_diagnoses_on_both_surfaces(engine):
    """Execution already declined typed (#1942); the validator now says the same thing."""
    g = _nodes_only()

    for query in ([n(), e_forward(), n()], "MATCH (a)-[r]->(b) RETURN a"):
        with pytest.raises(GFQLValidationError) as validate_exc:
            g.gfql_validate(query)
        assert validate_exc.value.code == ErrorCode.E304

        with pytest.raises(GFQLValidationError) as exec_exc:
            g.gfql(query, engine=engine)
        assert exec_exc.value.code == ErrorCode.E304


@pytest.mark.parametrize("query,expected", [(CYPHER, [{"a.id": 0, "a.v": 10}, {"a.id": 1, "a.v": 20}]),
                                            (CHAIN, [{"id": 1, "v": 20}])],
                         ids=["cypher", "chain"])
def test_nodes_only_admits_on_both_surfaces_with_values(query, expected):
    """Anti-vacuity: the shape the validator MAY admit must really answer, with these values."""
    g = _nodes_only()

    assert g.gfql_validate(query) == {
        "ok": True,
        "query_type": "chain",
        "language": "cypher" if isinstance(query, str) else "gfql",
        "diagnostics": [],
        **({"compiled_kind": "query"} if isinstance(query, str) else {}),
    }
    assert _records(g.gfql(query, engine="pandas")._nodes) == expected


def test_graph_with_data_still_validates_and_executes_clean():
    """Anti-vacuity: the shape guard must not fire on an ordinary bound graph."""
    g = (
        graphistry
        .nodes(pd.DataFrame({"id": [0, 1], "v": [10, 20]}), "id")
        .edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    )

    assert g.gfql_validate([n(), e_forward(), n()])["ok"] is True
    assert _records(g.gfql([n(), e_forward(), n()], engine="pandas")._nodes) == [
        {"id": 0, "v": 10}, {"id": 1, "v": 20}
    ]


@pytest.mark.parametrize("query", [CYPHER, CHAIN], ids=["cypher", "chain"])
def test_schema_false_skips_the_shape_guard_for_unbound_graphs(query):
    """Remote preflight validates with schema=False against a graph whose frames live server-side."""
    assert _both_none().gfql_validate(query, schema=False)["ok"] is True


def test_edges_only_graph_is_not_flagged_by_the_shape_guard():
    """Nodes are synthesizable from edges, so an edges-only graph stays answerable."""
    g = graphistry.edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")

    assert g.gfql_validate([n()])["ok"] is True
    assert _records(g.gfql([n()], engine="pandas")._nodes) == [{"id": 0}, {"id": 1}]
